
# -*- coding: UTF-8 -*-
# @Author : Reimplementation by GPT-5 Pro (based on FinalMLP AAAI'23)
# NOTE: This is a clean reimplementation for ReChorus, not using the original ReChorus FinalMLP.py.

"""
FinalMLP (AAAI 2023): An enhanced two-stream MLP with
 - stream-specific Feature Selection (gating) and
 - Multi-head Bilinear Fusion for stream-level interaction aggregation.

Paper: "FinalMLP: An Enhanced Two-Stream MLP Model for CTR Prediction", AAAI 2023.
This re-implementation follows the description in the paper and the reference MindSpore/FuxiCTR code,
but is written from scratch to fit the ReChorus API.

We provide two modes:
  - FinalMLPReImplCTR   : CTR prediction with BCE loss (ContextCTRModel + CTRRunner)
  - FinalMLPReImplTopK  : top-k ranking (ContextModel + BaseRunner, using ranking loss defined by the base class)

Key notations (matching the paper):
  e    : concatenated feature embeddings from all fields, shape (B, N, F*D)
  Gate : a small MLP or a parameter that outputs element-wise gating weights g in R^{F*D}
  h1,h2: gated inputs for Stream-1/2, h = 2*sigmoid(g) ⊙ e
  o1,o2: outputs of the two MLP streams
  Fusion: ŷ = σ(b + w1^T o1 + w2^T o2 + sum_{j=1..k} o1_j^T W_j o2_j)

Assumptions in this ReChorus version:
- Feature embeddings follow the ContextReader rules:
  categorical fields (ending with "_c" or "_id") use nn.Embedding,
  numeric fields (ending with "_f") use a linear projection to vec_size.
- For gating:
  * "global" makes g a learnable parameter shared by all instances;
  * "user" uses only the user_id embedding as the gating-condition input;
  * "item" uses only the item_id embedding as the gating-condition input;
- For multi-head fusion:
  we project o1 and o2 to the same dimension if needed, then split into k heads.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.BaseContextModel import ContextModel, ContextCTRModel

# ----------------------------
# Utilities
# ----------------------------
def mlp(in_dim, hidden_list, dropout=0.0, out_dim=None, out_activation=None):
    layers = []
    prev = in_dim
    for h in hidden_list:
        layers.extend([nn.Linear(prev, h), nn.ReLU()])
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    if out_dim is not None:
        layers.append(nn.Linear(prev, out_dim))
        if out_activation == 'relu':
            layers.append(nn.ReLU())
        elif out_activation == 'tanh':
            layers.append(nn.Tanh())
        elif out_activation == 'sigmoid':
            layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)

# ----------------------------
# Core FinalMLP (base)
# ----------------------------
class FinalMLPReImplBase(object):
    @staticmethod
    def parse_model_args_FinalMLP(parser):
        # embedding + architecture
        parser.add_argument('--emb_size', type=int, default=64, help='Embedding dimension per field.')
        parser.add_argument('--mlp1_layers', type=str, default='[256,256,128]', help='Hidden sizes for MLP stream #1.')
        parser.add_argument('--mlp2_layers', type=str, default='[256,256,128]', help='Hidden sizes for MLP stream #2.')
        parser.add_argument('--dropout', type=float, default=0.0, help='Dropout for MLP streams.')
        # feature selection (gating)
        parser.add_argument('--fs1', type=str, default='global', choices=['global','user','item'], help='Feature selection type for stream #1.')
        parser.add_argument('--fs2', type=str, default='item',   choices=['global','user','item'], help='Feature selection type for stream #2.')
        parser.add_argument('--gate_hidden', type=int, default=128, help='Hidden size of gating MLPs for user/item gating.')
        # fusion
        parser.add_argument('--num_heads', type=int, default=8, help='Number of heads for multi-head bilinear fusion.')
        return parser

    def _define_init(self, args, corpus):
        # --------- store hyper-params ---------
        self.vec_size     = args.emb_size
        self.mlp1_layers  = eval(args.mlp1_layers) if isinstance(args.mlp1_layers, str) else args.mlp1_layers
        self.mlp2_layers  = eval(args.mlp2_layers) if isinstance(args.mlp2_layers, str) else args.mlp2_layers
        self.dropout      = args.dropout
        self.fs1_type     = args.fs1
        self.fs2_type     = args.fs2
        self.gate_hidden  = args.gate_hidden
        self.num_heads    = args.num_heads

        # --------- build per-field embeddings as in Context models ---------
        self.context_embedding = nn.ModuleDict()
        for f in self.context_features:
            if f.endswith('_c') or f.endswith('_id'):
                self.context_embedding[f] = nn.Embedding(self.feature_max[f], self.vec_size)
            else:
                # numeric feature: 1 -> vec_size projection
                self.context_embedding[f] = nn.Linear(1, self.vec_size, bias=False)

        # total feature dim after concatenation
        self.num_fields = len(self.context_features)
        self.concat_dim = self.num_fields * self.vec_size

        # --------- gating (feature selection) ---------
        # We implement three variants: global / user / item
        # Each gate outputs a vector in R^{F*D} that will be transformed by 2*sigmoid and multiplied element-wise with e
        def build_gate(gtype):
            if gtype == 'global':
                # one parameter per element (broadcast to batch & items)
                return nn.Parameter(torch.zeros(self.concat_dim), requires_grad=True)
            elif gtype == 'user':
                # MLP(user_embed -> gating vector)
                return mlp(self.vec_size, [self.gate_hidden], dropout=0.0, out_dim=self.concat_dim, out_activation=None)
            elif gtype == 'item':
                # MLP(item_embed -> gating vector)
                return mlp(self.vec_size, [self.gate_hidden], dropout=0.0, out_dim=self.concat_dim, out_activation=None)
            else:
                raise ValueError('Unknown gating type: {}'.format(gtype))

        self.gate1 = build_gate(self.fs1_type)
        self.gate2 = build_gate(self.fs2_type)

        # --------- two MLP streams ---------
        self.stream1 = mlp(self.concat_dim, self.mlp1_layers, dropout=self.dropout)
        self.stream2 = mlp(self.concat_dim, self.mlp2_layers, dropout=self.dropout)

        # --------- stream output dims and alignment ---------
        d1 = self.mlp1_layers[-1] if len(self.mlp1_layers)>0 else self.concat_dim
        d2 = self.mlp2_layers[-1] if len(self.mlp2_layers)>0 else self.concat_dim
        # project to common dim if needed
        fuse_dim = min(d1, d2)
        if d1 != fuse_dim:
            self.proj1 = nn.Linear(d1, fuse_dim, bias=False)
        else:
            self.proj1 = nn.Identity()
        if d2 != fuse_dim:
            self.proj2 = nn.Linear(d2, fuse_dim, bias=False)
        else:
            self.proj2 = nn.Identity()
        self.fuse_dim = fuse_dim

        # --------- multi-head bilinear fusion params ---------
        assert self.num_heads >= 1 and self.fuse_dim % self.num_heads == 0, \
            "fuse_dim must be divisible by num_heads. Got fuse_dim={}, num_heads={}".format(self.fuse_dim, self.num_heads)
        self.head_dim = self.fuse_dim // self.num_heads

        # Bilinear for each head: W_j in R^{d_h x d_h}
        self.W_heads = nn.ParameterList([nn.Parameter(torch.randn(self.head_dim, self.head_dim)*0.01)
                                         for _ in range(self.num_heads)])
        # First-order terms
        self.w1 = nn.Linear(d1, 1, bias=False)
        self.w2 = nn.Linear(d2, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(1), requires_grad=True)

        # init weights
        self.apply(self.init_weights)

    def _embed_all_fields(self, feed_dict):
        """
        Build the concatenated embedding e \in R^{B x N x (F*D)} from all fields.
        Follows ContextReader conventions on feature types.
        """
        item_ids = feed_dict['item_id']
        if isinstance(item_ids, int) or (hasattr(item_ids, 'dim') and item_ids.dim()==1):  # handle degenerate case
            item_ids = item_ids.view(-1,1)
        B, N = item_ids.shape

        vectors = []
        for f in self.context_features:
            if f.endswith('_c') or f.endswith('_id'):
                v = self.context_embedding[f](feed_dict[f])
            else:
                v = self.context_embedding[f](feed_dict[f].float().unsqueeze(-1))
            # make sure shape is (B, N, D)
            if v.dim() == 2:
                v = v.unsqueeze(1).repeat(1, N, 1)
            vectors.append(v)
        # stack features and flatten: (B, N, F, D) -> (B, N, F*D)
        feat_tensor = torch.stack(vectors, dim=-2)   # (B, N, F, D)
        e = feat_tensor.flatten(start_dim=-2)        # (B, N, F*D)
        return e, B, N

    def _gate_weights(self, feed_dict, e, which='gate1'):
        """
        Compute gating weights g in R^{B x N x (F*D)} for a stream.
        """
        B, N, CD = e.shape
        gate = getattr(self, which)
        if isinstance(gate, nn.Parameter):
            g = gate.view(1,1,-1).expand(B,N,-1)
        else:
            if which == 'gate1' and self.fs1_type == 'user':
                cond = self.context_embedding['user_id'](feed_dict['user_id'])      # (B, D)
            elif which == 'gate2' and self.fs2_type == 'user':
                cond = self.context_embedding['user_id'](feed_dict['user_id'])
            elif which == 'gate1' and self.fs1_type == 'item':
                # use the target item embedding (first candidate) as condition; average over N if multiple
                item_emb = self.context_embedding['item_id'](feed_dict['item_id'])  # (B, N, D)
                cond = item_emb.mean(dim=1)                                         # (B, D)
            elif which == 'gate2' and self.fs2_type == 'item':
                item_emb = self.context_embedding['item_id'](feed_dict['item_id'])
                cond = item_emb.mean(dim=1)
            else:
                cond = torch.zeros(B, self.vec_size, device=e.device)               # safe default
            g_vec = gate(cond)                                                      # (B, CD)
            g = g_vec.view(B,1,CD).repeat(1, N, 1)
        # transform to [0,2] with mean ~1 as in the paper
        return 2.0 * torch.sigmoid(g)

    def _forward_base(self, feed_dict):
        """
        Common forward: embeddings -> gated inputs -> two MLPs -> multi-head fusion score.
        Return a dict with key 'prediction' (B, N).
        """
        e, B, N = self._embed_all_fields(feed_dict)
        # Stream-specific feature selection
        g1 = self._gate_weights(feed_dict, e, which='gate1')
        g2 = self._gate_weights(feed_dict, e, which='gate2')
        h1 = g1 * e
        h2 = g2 * e

        # Two-Stream MLPs
        o1 = self.stream1(h1.view(B*N, -1))   # (B*N, d1)
        o2 = self.stream2(h2.view(B*N, -1))   # (B*N, d2)
        d1 = o1.shape[-1]; d2 = o2.shape[-1]
        o1_2d = o1.view(B, N, d1)
        o2_2d = o2.view(B, N, d2)

        # Linear terms
        lin = self.bias + self.w1(o1_2d) + self.w2(o2_2d)   # (B,N,1)

        # Align dims and multi-head bilinear fusion
        p1 = self.proj1(o1_2d)               # (B,N,fuse_dim)
        p2 = self.proj2(o2_2d)               # (B,N,fuse_dim)
        # split into heads
        p1_heads = p1.view(B, N, self.num_heads, self.head_dim)   # (B,N,H,Dh)
        p2_heads = p2.view(B, N, self.num_heads, self.head_dim)
        # bilinear per head: sum over j [ (o1_j^T W_j o2_j) ]
        bilinear_list = []
        for j in range(self.num_heads):
            # (B,N,1) via batch vector-matrix-vector product
            a = p1_heads[:,:,j,:]                     # (B,N,Dh)
            b = p2_heads[:,:,j,:]                     # (B,N,Dh)
            W = self.W_heads[j]                       # (Dh,Dh)
            # compute element-wise: for each (B,N), a[b,:]^T W b[b,:]
            # -> (B,N,Dh) x (Dh,Dh) -> (B,N,Dh) then dot with b -> (B,N)
            Wa = torch.matmul(a, W)                   # (B,N,Dh)
            ab = (Wa * b).sum(dim=-1, keepdim=True)   # (B,N,1)
            bilinear_list.append(ab)
        bilinear = torch.stack(bilinear_list, dim=2).sum(dim=2)   # sum over heads -> (B,N,1)

        score = lin + bilinear                           # (B,N,1)
        return {'prediction': score.squeeze(-1)}         # (B,N)

# ----------------------------
# CTR variant
# ----------------------------
class FinalMLPReImplCTR(ContextCTRModel, FinalMLPReImplBase):
    reader, runner = 'ContextReader', 'CTRRunner'
    extra_log_args = ['emb_size','mlp1_layers','mlp2_layers','fs1','fs2','num_heads','dropout','loss_n']

    @staticmethod
    def parse_model_args(parser):
        parser = FinalMLPReImplBase.parse_model_args_FinalMLP(parser)
        return ContextCTRModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        ContextCTRModel.__init__(self, args, corpus)
        self._define_init(args, corpus)

    def forward(self, feed_dict):
        out = self._forward_base(feed_dict)
        out['prediction'] = out['prediction'].view(-1).sigmoid()
        out['label'] = feed_dict['label'].view(-1)
        return out

# ----------------------------
# Top-K variant
# ----------------------------
class FinalMLPReImplTopK(ContextModel, FinalMLPReImplBase):
    reader, runner = 'ContextReader', 'BaseRunner'
    extra_log_args = ['emb_size','mlp1_layers','mlp2_layers','fs1','fs2','num_heads','dropout','loss_n']

    @staticmethod
    def parse_model_args(parser):
        parser = FinalMLPReImplBase.parse_model_args_FinalMLP(parser)
        return ContextModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        ContextModel.__init__(self, args, corpus)
        self._define_init(args, corpus)

    def forward(self, feed_dict):
        return self._forward_base(feed_dict)

