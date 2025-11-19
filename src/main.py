# -*- coding: UTF-8 -*-

import os
import sys
import pickle
import logging
import argparse
import pandas as pd
import torch

from helpers import *
from models.general import *
from models.sequential import *
from models.developing import *
from models.context import *
from models.context_seq import *
from models.reranker import *
from utils import utils


def parse_global_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--gpu', type=str, default='0',
                        help='Set CUDA_VISIBLE_DEVICES, pass "" for CPU.')
    parser.add_argument('--verbose', type=int, default=logging.INFO,
                        help='Logging level: 0,10,...,50')
    parser.add_argument('--log_file', type=str, default='',
                        help='Logging file path (auto if empty).')
    parser.add_argument('--random_seed', type=int, default=0,
                        help='Random seed of numpy and pytorch')
    parser.add_argument('--load', type=int, default=0,
                        help='Whether to load checkpoint before training')
    parser.add_argument('--train', type=int, default=1,
                        help='Whether to train the model')
    parser.add_argument('--save_final_results', type=int, default=1,
                        help='Save prediction csv after training')
    parser.add_argument('--regenerate', type=int, default=0,
                        help='Regenerate intermediate corpus cache (.pkl)')
    return parser


def _resolve_name_to_class(name: str):
    """
    将 'X' 或 'X.X' 解析成类对象。
    helpers/* 与 models/* 已通过 from ... import * 导入到当前命名空间。
    """
    try:
        # 例如 'ContextReader.ContextReader'
        return eval(f"{name}.{name}")
    except Exception:
        # 例如 'ContextReader'
        return eval(name)


def main():
    logging.info('-' * 45 + ' BEGIN: ' + utils.get_time() + ' ' + '-' * 45)

    exclude = ['check_epoch', 'log_file', 'model_path', 'path', 'pin_memory',
               'load', 'regenerate', 'sep', 'train', 'verbose', 'metric',
               'test_epoch', 'buffer']
    logging.info(utils.format_arg_str(args, exclude_lst=exclude))

    # ----- Seed -----
    utils.init_seed(args.random_seed)

    # ----- Device -----
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cpu')
    if args.gpu != '' and torch.cuda.is_available():
        args.device = torch.device('cuda')
    logging.info('Device: {}'.format(args.device))

    # ----- Read corpus (with cache) -----
    corpus_tag = reader_str + args.data_appendix
    corpus_path = os.path.join(args.path, args.dataset, corpus_tag + '.pkl')
    if not args.regenerate and os.path.exists(corpus_path):
        logging.info('Load corpus from {}'.format(corpus_path))
        corpus = pickle.load(open(corpus_path, 'rb'))
    else:
        corpus = reader_name(args)
        utils.check_dir(corpus_path)
        logging.info('Save corpus to {}'.format(corpus_path))
        pickle.dump(corpus, open(corpus_path, 'wb'))

    # ----- Build model -----
    model = model_name(args, corpus).to(args.device)
    logging.info('#params: {}'.format(model.count_variables()))
    logging.info(model)

    # ----- Build datasets -----
    data_dict = {}
    for phase in ['train', 'dev', 'test']:
        data_dict[phase] = model_name.Dataset(model, corpus, phase)
        data_dict[phase].prepare()

    # ----- Train / Eval -----
    runner = runner_name(args)
    logging.info('Test Before Training: ' + runner.print_res(data_dict['test']))

    if args.load > 0:
        model.load_model()
    if args.train > 0:
        runner.train(data_dict)

    dev_res = runner.print_res(data_dict['dev'])
    logging.info(os.linesep + 'Dev  After Training: ' + dev_res)
    test_res = runner.print_res(data_dict['test'])
    logging.info(os.linesep + 'Test After Training: ' + test_res)

    # ----- Save predictions -----
    if args.save_final_results == 1:
        save_rec_results(data_dict['dev'], runner, 100)
        save_rec_results(data_dict['test'], runner, 100)

    model.actions_after_train()
    logging.info(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)


def save_rec_results(dataset, runner, topk):
    """
    将预测保存为 csv：
      - CTR: user_id, item_id, pCTR, label
      - TopK: user_id, rec_items, rec_predictions
      - Impression/General/Sequential: user_id, pos/neg items & scores
    """
    model_full_name = '{0}{1}'.format(init_args.model_name, init_args.model_mode)
    result_path = os.path.join(runner.log_path, runner.save_appendix,
                               'rec-{}-{}.csv'.format(model_full_name, dataset.phase))
    utils.check_dir(result_path)

    if init_args.model_mode == 'CTR':
        logging.info('Saving CTR prediction results to: {}'.format(result_path))
        predictions, labels = runner.predict(dataset)
        users, items = [], []
        for i in range(len(dataset)):
            info = dataset[i]
            users.append(info['user_id'])
            # CTR 每条样本是单 item，通常包装为 1-D/1-list
            items.append(info['item_id'][0] if hasattr(info['item_id'], '__len__') else info['item_id'])
        rec_df = pd.DataFrame({
            'user_id': users,
            'item_id': items,
            'pCTR': predictions,
            'label': labels
        })
        rec_df.to_csv(result_path, sep=args.sep, index=False)

    elif init_args.model_mode in ['TopK', '']:
        logging.info('Saving top-{} recommendation results to: {}'.format(topk, result_path))
        predictions = runner.predict(dataset)  # shape: [n_users, n_candidates]
        users, rec_items, rec_scores = [], [], []
        for i in range(len(dataset)):
            info = dataset[i]
            users.append(info['user_id'])
            item_ids = info['item_id']
            scores = predictions[i]
            # zip & sort
            sorted_pairs = sorted(zip(item_ids, scores), key=lambda x: x[1], reverse=True)[:topk]
            rec_items.append([x[0] for x in sorted_pairs])
            rec_scores.append([x[1] for x in sorted_pairs])
        rec_df = pd.DataFrame({
            'user_id': users,
            'rec_items': rec_items,
            'rec_predictions': rec_scores
        })
        rec_df.to_csv(result_path, sep=args.sep, index=False)

    elif init_args.model_mode in ['Impression', 'General', 'Sequential']:
        logging.info('Saving all recommendation results to: {}'.format(result_path))
        predictions = runner.predict(dataset)  # list-wise
        users, pos_items, pos_scores, neg_items, neg_scores = [], [], [], [], []
        for i in range(len(dataset)):
            info = dataset[i]
            users.append(info['user_id'])
            pos_items.append(info['pos_items'])
            neg_items.append(info['neg_items'])
            pos_len = dataset.pos_len
            neg_len = dataset.neg_len
            pos_scores.append(predictions[i][:pos_len])
            neg_scores.append(predictions[i][pos_len:pos_len + neg_len])
        rec_df = pd.DataFrame({
            'user_id': users,
            'pos_items': pos_items,
            'pos_predictions': pos_scores,
            'neg_items': neg_items,
            'neg_predictions': neg_scores
        })
        rec_df.to_csv(result_path, sep=args.sep, index=False)

    else:
        # 未知模式，不保存
        return 0

    logging.info("{} Prediction results saved!".format(dataset.phase))


if __name__ == '__main__':
    # ------ First-stage: model + mode ------
    init_parser = argparse.ArgumentParser(description='Model Entry')
    init_parser.add_argument('--model_name', type=str, default='SASRec',
                             help='Model class name, e.g., BPRMF/NeuMF/FM/DeepFM/FinalMLPReImpl')
    init_parser.add_argument('--model_mode', type=str, default='',
                             help='Task mode: ""/TopK/CTR/Impression/General/Sequential')
    init_args, _ = init_parser.parse_known_args()

    # ------ Robust model class resolution ------
    mode = (init_args.model_mode or '').strip()
    name = (init_args.model_name or '').strip()

    def _resolve_model_cls(_name: str, _mode: str):
        # 我们重写的 FinalMLP：显式映射 TopK/CTR
        if _name == 'FinalMLPReImpl':
            from models.context.FinalMLP_ReImpl import FinalMLPReImplTopK, FinalMLPReImplCTR
            return FinalMLPReImplTopK if _mode.lower() == 'topk' else FinalMLPReImplCTR

        # 经典模型优先尝试：模块.类(带后缀) -> 模块.类 -> 直接类名
        tried = []
        for cand in (f'{_name}.{_name}{_mode}', f'{_name}.{_name}', _name):
            try:
                return eval(cand)
            except Exception as e:
                tried.append((cand, repr(e)))
        raise RuntimeError(f'Cannot resolve model class for name="{_name}", mode="{_mode}". Tried: {tried}')

    model_name = _resolve_model_cls(name, mode)

    # ------ Resolve reader / runner (class names or defaults) ------
    reader_str = getattr(model_name, 'reader', None)
    runner_str = getattr(model_name, 'runner', None)
    if not reader_str:
        reader_str = 'ContextReader' if mode.lower() in ('topk', 'ctr') else 'BaseReader'
    if not runner_str:
        if mode.lower() == 'topk':
            runner_str = 'TopKRunner'
        elif mode.lower() == 'ctr':
            runner_str = 'CTRRunner'
        else:
            runner_str = 'BaseRunner'

    reader_name = _resolve_name_to_class(reader_str)
    runner_name = _resolve_name_to_class(runner_str)

    # ------ Second-stage: merge args from reader / runner / model ------
    parser = argparse.ArgumentParser(description='')
    parser = parse_global_args(parser)
    parser = reader_name.parse_data_args(parser)
    parser = runner_name.parse_runner_args(parser)
    parser = model_name.parse_model_args(parser)
    args, _ = parser.parse_known_args()

    # 为 ContextReader 生成 data_appendix（如 _context000）
    args.data_appendix = ''
    if 'Context' in reader_str:
        # include_item_features/include_user_features/include_situation_features 通常由 reader 的 args 提供
        args.data_appendix = '_context%d%d%d' % (getattr(args, 'include_item_features', 0),
                                                 getattr(args, 'include_user_features', 0),
                                                 getattr(args, 'include_situation_features', 0))

    # ------ Build log/model path ------
    extra_log_args = getattr(model_name, 'extra_log_args', [])
    if not isinstance(extra_log_args, list):
        extra_log_args = []

    log_args = [init_args.model_name + init_args.model_mode, args.dataset + args.data_appendix, str(args.random_seed)]
    for arg in ['lr', 'l2'] + extra_log_args:
        if hasattr(args, arg):
            log_args.append(arg + '=' + str(getattr(args, arg)))
    log_file_name = '__'.join(log_args).replace(' ', '__')

    if getattr(args, 'log_file', '') == '':
        args.log_file = '../log/{}/{}.txt'.format(init_args.model_name + init_args.model_mode, log_file_name)
    if getattr(args, 'model_path', '') == '':
        args.model_path = '../model/{}/{}.pt'.format(init_args.model_name + init_args.model_mode, log_file_name)

    utils.check_dir(args.log_file)
    logging.basicConfig(filename=args.log_file, level=args.verbose)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(init_args)

    # keep for save_rec_results
    reader_str_global = reader_str
    # 让 main() 里能访问
    globals().update({
        'reader_str': reader_str,
        'reader_name': reader_name,
        'runner_name': runner_name,
        'model_name': model_name,
        'args': args,
        'init_args': init_args
    })

    main()
