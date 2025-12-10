建议直接看Code,不要看Preview
ReChorus-FinalMLP-Reproduction 实验全流程（Windows + PowerShell +conda）

适用于 Windows + PowerShell + Conda 环境，项目：ReChorus-FinalMLP-Reproduction（含 MovieLens_1M、Amazon Grocery & Gourmet Food 两个数据集的 TopK/CTR 任务）。

0. 环境准备

1) 已安装 Anaconda/Miniconda，创建好环境（例）：

创建虚拟环境
conda create -n chorus python=3.10.4
conda activate chorus

第一步：安装 PyTorch (1.12.1) 为了确保 GPU 支持最稳定，使用 PyTorch 官方源：
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch

第二步：使用 conda-forge 安装其余依赖 我们指定 -c conda-forge 来安装其他库，这能保证下载到精确的旧版本：
conda install -c conda-forge numpy=1.22.3 pandas=1.4.4 scikit-learn=1.1.3 scipy=1.7.3 tqdm matplotlib seaborn ipython jupyter pyyaml

另外也可以参考使用requirements.txt配置环境

2) 确保 CUDA 驱动 / 显卡可用（如需 GPU）。

3) 仓库根目录记为：<Root>（例如你本机是 E:\ReChorus-FinalMLP-Reproduction）。之后均通过参数 -Root "<Root>" 传入。

1. 数据目录要求

<Root>\data\ 下应已经存在如下子目录（前期准备好的）：
MovieLens_1M/
  ├─ ML_1MTOPK/
  │    ├─ train.csv  dev.csv  test.csv
  └─ ML_1MCTR/
       ├─ train.csv  dev.csv  test.csv

Grocery_and_Gourmet_Food/
  ├─ GGFTOPK/
  │    ├─ train.csv  dev.csv  test.csv
  └─ GGFCTR/
       ├─ train.csv  dev.csv  test.csv

TopK 文件必须包含 user_id, item_id, time, neg_items；
CTR 文件必须包含 user_id, item_id, label。

2. 一键跑全量实验（多 seed）

使用 scripts\exp_full.ps1。不使用硬编码盘符，通过参数指定路径。
python解释器使用自己的路径，下面用D:\Anaconda3\envs\chorus\python.exe作为示例
例：用 5 个种子（0..4），GPU 0，不重新生成语料（--regenerate 0）：
powershell -ExecutionPolicy Bypass -File scripts\exp_full.ps1 `
  -Root "." `
  -Py "D:\Anaconda3\envs\chorus\python.exe" `
  -Seeds 0,1,2,3,4 `
  -Gpu 0 `
  -Regenerate 0

可调公共/任务超参（均有默认值）：
公共：-EmbSize、-NumWorkers
TopK：-EpochTopK、-EarlyStopTopK、-BatchTopK、-EvalBatchTopK
CTR ：-EpochCTR、-EarlyStopCTR、-BatchCTR、-EvalBatchCTR

脚本会依次运行 6 个模型×2 数据集×2 任务×N 个 seed。
日志、模型与预测结果默认写到 <Root>\log、<Root>\model 下（由 main.py/模型保存逻辑决定）。

3. 汇总日志（Dev / Test 两套）

使用「两套汇总」：
Dev 只取 Dev After Training 行，用于选最佳模型/超参/seed；
Test 只取 Test After Training 行，用于最终报告。

3.1 解析原始日志 → 结构化 CSV

# 解析 Dev
python tools\summarize_all.py `
  --log_dir  "<Root>\log" `
  --out_dir  "<Root>\log\DevAfterTraining" `
  --project_root "<Root>" `
  --line_pattern "Dev After Training"

# 解析 Test
python tools\summarize_all.py `
  --log_dir  "<Root>\log" `
  --out_dir  "<Root>\log\TestAfterTraining" `
  --project_root "<Root>" `
  --line_pattern "Test After Training"

会生成 topk_wide.csv / ctr_wide.csv 等。

3.2 规范化/清洗

把数据集别名统一（例如把 Grocery_and_Gourm... 纠正成完整路径），并确保 seed 字段可聚合。

python tools\fix_wide_csv_v2.py `
  --dev_dir  "<Root>\log\DevAfterTraining" `
  --test_dir "<Root>\log\TestAfterTraining"

如仍提示某些分组种子数 ≠ 期望值（例如不是 5），请先清空历史旧日志后重跑训练，或核对是否所有 seed 都完成。

3.3 统计均值±标准差（可选）

将宽表转为按（模型×数据集）聚合的 mean/std 表（同时保留实验目录与 checkpoint 路径列，便于追溯）：
# Dev
python tools\agg_meanstd.py `
  --in_dir  "<Root>\log\DevAfterTraining" `
  --out_dir "<Root>\log\Devmeanstd"

# Test
python tools\agg_meanstd.py `
  --in_dir  "<Root>\log\TestAfterTraining" `
  --out_dir "<Root>\log\Testmeanstd"

会生成：topk_meanstd.csv、ctr_meanstd.csv。

3.4 Dev/Test 对齐（可选）

将 Dev 的“最佳”与 Test 的结果对齐，用于最终对照表：
python tools\merge_dev_test.py `
  --dev_dir  "<Root>\log\DevAfterTraining" `
  --test_dir "<Root>\log\TestAfterTraining" `
  --out_dir  "<Root>\log\merged" `
  --topk_metric "NDCG@10" `
  --ctr_metric  "AUC"      # 或 "LOG_LOSS"

这一步会生成 merged_topk.csv / merged_ctr.csv，用于“先在 Dev 选，再看 Test 的固定结果”。

4. 生成论文/报告用素材

推荐分别生成 AUC 和 LOG_LOSS 两个版本（CTR 主指标不同）。

AUC 版
python tools\make_report_assets.py `
  --dev_dir    "<Root>\log\Devmeanstd" `
  --test_dir   "<Root>\log\Testmeanstd" `
  --merged_dir "<Root>\log\merged" `
  --out_dir    "<Root>\log\report_auc" `
  --topk_metric "NDCG@10" `
  --ctr_metric  "AUC"

LOG_LOSS 版
python tools\make_report_assets.py `
  --dev_dir    "<Root>\log\Devmeanstd" `
  --test_dir   "<Root>\log\Testmeanstd" `
  --merged_dir "<Root>\log\merged" `
  --out_dir    "<Root>\log\report_logloss" `
  --topk_metric "NDCG@10" `
  --ctr_metric  "LOG_LOSS"

输出目录将包含：
tables_topk/TopK_summary_pretty.tex
tables_ctr/CTR_summary_pretty.tex
以及 CSV 版本供检查。

说明：如均值±标准差的 std == 0（统一种子或样本数不足导致），可在报告中仅展示均值，省略“±std”。

5. 拟合概率的校准评估（ECE/可靠性图）（可选）

仅对 CTR 预测文件（包含 label,prediction 列）有效；TopK 会被跳过（提示 [SKIP] 非 CTR 预测 属正常）。

python tools\compute_calibration.py `
  --log_dir "<Root>\log" `
  --out_dir "<Root>\log\calib" `
  --bins 15

输出：
calibration_summary.csv：分模型/数据集/切分的 ECE、Brier 分数等；
curves/：分箱曲线数据；
plots/：可靠性图（PNG/PDF）。

6. 一键命令清单
假设已经在 <Root> 目录，且 Conda 环境已激活。

# 1) 跑全量（seed 0..4；GPU 0；不重建语料）
powershell -ExecutionPolicy Bypass -File scripts\exp_full.ps1 `
  -Root "." -Py "D:\Anaconda3\envs\chorus\python.exe" `
  -Seeds 0,1,2,3,4 -Gpu 0 -Regenerate 0

# 2) 结构化日志（Dev/Test 各一份）
python tools\summarize_all.py --log_dir ".\log" --out_dir ".\log\DevAfterTraining" --project_root "." --line_pattern "Dev After Training"
python tools\summarize_all.py --log_dir ".\log" --out_dir ".\log\TestAfterTraining" --project_root "." --line_pattern "Test After Training"

# 3) 清洗/规范化
python tools\fix_wide_csv_v2.py --dev_dir ".\log\DevAfterTraining" --test_dir ".\log\TestAfterTraining"

# 4) 统计均值±标准差（生成 topk_meanstd.csv / ctr_meanstd.csv）
python tools\agg_meanstd.py --in_dir ".\log\DevAfterTraining"  --out_dir ".\log\Devmeanstd"
python tools\agg_meanstd.py --in_dir ".\log\TestAfterTraining" --out_dir ".\log\Testmeanstd"

# 5) Dev/Test 对齐（可选）
python tools\merge_dev_test.py --dev_dir ".\log\DevAfterTraining" --test_dir ".\log\TestAfterTraining" --out_dir ".\log\merged" --topk_metric "NDCG@10" --ctr_metric "AUC"

# 6) 报告素材（AUC 版 / LOG_LOSS 版各一套）
python tools\make_report_assets.py --dev_dir ".\log\Devmeanstd" --test_dir ".\log\Testmeanstd" --merged_dir ".\log\merged" --out_dir ".\log\report_auc"     --topk_metric "NDCG@10" --ctr_metric "AUC"
python tools\make_report_assets.py --dev_dir ".\log\Devmeanstd" --test_dir ".\log\Testmeanstd" --merged_dir ".\log\merged" --out_dir ".\log\report_logloss" --topk_metric "NDCG@10" --ctr_metric "LOG_LOSS"

# 7) 校准评估（ECE/可靠性图）（可选）
python tools\compute_calibration.py --log_dir ".\log" --out_dir ".\log\calib" --bins 15

7. 完整项目目录

<Root>/
├─ README.md                         
├─ README_exp.md                     
│
├─ scripts/
│  ├─ exp_full.ps1                   # 一键全量实验（多 seed；TopK+CTR）
│  └─ exp_smoke.ps1                  # 冒烟测试（可选）
│  
│
├─ src/
│  ├─ main.py                        # 训练入口（--model_name / --model_mode / --dataset / --seed 等）
│  ├─ helpers/
│  │  ├─ BaseReader.py
│  │  ├─ ContextReader.py
│  │  ├─ BaseRunner.py
│  │  └─ ...（其余 reader/runner 工具）
│  └─ models/
│     ├─ general/
│     │  ├─ BPRMF.py
│     │  ├─ FM.py
│     │  └─ DeepFM.py
│     └─ context/
│        ├─ FinalMLP_ReImpl.py       # FinalMLPReImplTopK / FinalMLPReImplCTR 实现在这
│        └─ ...（如需的其他上下文模型）
│
├─ tools/
│  ├─ summarize_all.py               # 从原始训练日志抽取 “Dev/Test After Training” 行生成 wide 表
│  ├─ fix_wide_csv_v2.py             # 规范化/清洗 wide 表（修正截断数据集名、标准化 seed 等）
│  ├─ agg_meanstd.py                 # 对 wide 表按（模型×数据集）聚合出 mean/std（含 exp_dir/ckpt 列）
│  ├─ merge_dev_test.py              # Dev 选优与 Test 对齐（生成 merged_topk.csv / merged_ctr.csv）
│  ├─ make_report_assets.py          # 产出 \input 即用的 .tex 表（AUC 版 / LOG_LOSS 版）
│  ├─ compute_calibration.py         # 仅 CTR：生成校准汇总、曲线与可靠性图（有 label/prediction 才会处理）
│  ├─ audit_wide.py                  # 自检 wide 表的 seed 覆盖情况（可选）
│  ├─ summarize_logs.py              # 辅助日志解析（保留）
│  ├─ analyze_package.py             # 打包后数据自检/分析（可选）
│  └─ analyze_all_final.py           # 合并分析脚本（可选）
│
├─ data/
│  ├─ MovieLens_1M/
│  │  ├─ ML_1MTOPK/                  # TopK 数据（三件套）
│  │  │  ├─ train.csv
│  │  │  ├─ dev.csv
│  │  │  └─ test.csv
│  │  └─ ML_1MCTR/                   # CTR 数据（三件套）
│  │     ├─ train.csv
│  │     ├─ dev.csv
│  │     └─ test.csv
│  └─ Grocery_and_Gourmet_Food/
│     ├─ GGFTOPK/                    # TopK 数据（三件套，含 neg_items）
│     │  ├─ train.csv
│     │  ├─ dev.csv
│     │  └─ test.csv
│     ├─ GGFCTR/                     # CTR 数据（三件套，含 label）
│     │  ├─ train.csv
│     │  ├─ dev.csv
│     │  └─ test.csv
│     └─ item_meta.csv               # 若使用了 meta（可选）
│
├─ model/                             # 训练得到的 checkpoint（按模型/数据集/seed 分组）
│  ├─ FinalMLPReImplTopK/
│  │  ├─ FinalMLPReImplTopK__MovieLens_1M/
│  │  │  └─ ML_1MTOPK_context000__{seed}__lr=...__emb_size=...__.pt
│  │  └─ FinalMLPReImplTopK__Grocery_and_Gourmet_Food/
│  │     └─ GGFTOPK_context000__{seed}__lr=...__emb_size=...__.pt
│  ├─ BPRMFTopK/
│  │  └─ BPRMFTopK__MovieLens_1M / BPRMFTopK__Grocery_and_Gourmet_Food / ...
│  ├─ NeuMFTopK/
│  ├─ FinalMLPReImplCTR/
│  ├─ FMCTR/
│  └─ DeepFMCTR/
│
└─ log/                               # 训练日志与预测结果（rec-*.csv）
   ├─ FinalMLPReImplTopK/
   │  ├─ FinalMLPReImplTopK__MovieLens_1M/
   │  │  └─ ML_1MTOPK_context000__{seed}__lr=.../
   │  │     ├─ train.log
   │  │     ├─ rec-FinalMLPReImplTopK-dev.csv
   │  │     └─ rec-FinalMLPReImplTopK-test.csv
   │  └─ FinalMLPReImplTopK__Grocery_and_Gourmet_Food/
   │     └─ GGFTOPK_context000__{seed}__lr=.../
   │        ├─ train.log
   │        ├─ rec-FinalMLPReImplTopK-dev.csv
   │        └─ rec-FinalMLPReImplTopK-test.csv
   ├─ BPRMFTopK/ ...                  # 同上（TopK 类同）
   ├─ NeuMFTopK/ ...
   ├─ FinalMLPReImplCTR/
   │  ├─ FinalMLPReImplCTR__MovieLens_1M/
   │  │  └─ ML_1MCTR_context000__{seed}__lr=.../
   │  │     ├─ train.log
   │  │     ├─ rec-FinalMLPReImplCTR-dev.csv       # 含 label/prediction（CTR）
   │  │     └─ rec-FinalMLPReImplCTR-test.csv
   │  └─ FinalMLPReImplCTR__Grocery_and_Gourmet_Food/
   │     └─ GGFCTR_context000__{seed}__lr=.../
   │        ├─ train.log
   │        ├─ rec-FinalMLPReImplCTR-dev.csv
   │        └─ rec-FinalMLPReImplCTR-test.csv
   ├─ FMCTR/ ...
   ├─ DeepFMCTR/ ...
   │
   ├─ DevAfterTraining/               # summarize_all.py（Dev）产物
   │  ├─ topk_wide.csv
   │  └─ ctr_wide.csv
   ├─ TestAfterTraining/              # summarize_all.py（Test）产物
   │  ├─ topk_wide.csv
   │  └─ ctr_wide.csv
   ├─ Devmeanstd/                     # agg_meanstd.py（Dev）产物
   │  ├─ topk_meanstd.csv
   │  └─ ctr_meanstd.csv
   ├─ Testmeanstd/                    # agg_meanstd.py（Test）产物
   │  ├─ topk_meanstd.csv
   │  └─ ctr_meanstd.csv
   ├─ merged/                         # merge_dev_test.py 产物
   │  ├─ merged_topk.csv
   │  └─ merged_ctr.csv
   ├─ report_auc/                     # make_report_assets.py（AUC 版）
   │  ├─ tables_topk/TopK_summary_pretty.tex
   │  └─ tables_ctr/CTR_summary_pretty.tex
   ├─ report_logloss/                 # make_report_assets.py（LOG_LOSS 版）
   │  ├─ tables_topk/TopK_summary_pretty.tex
   │  └─ tables_ctr/CTR_summary_pretty.tex
   └─ calib/                          # compute_calibration.py 产物（仅 CTR 有）
      ├─ calibration_summary.csv
      ├─ curves/
      └─ plots/
         ├─ reliability_FMCTR__MovieLens_1M__dev__ML_1MCTR_context000__0__lr=0.png
         ├─ reliability_FMCTR__MovieLens_1M__test__ML_1MCTR_context000__0__lr=0.png
         ├─ reliability_FinalMLPReImplCTR__Grocery_and_Gourmet_Food__dev__GGFCTR_context000__0__lr=0.png
         └─ ...（共 ~60 张）
 








