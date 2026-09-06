# 项目代码模块总览

这个仓库包含两层内容：

1. 原始 `phygnn` 通用物理引导神经网络框架。
2. 针对 `3D BN / Al2O3` 多保真 PGNN 实验方案新增的材料实验工作流。

## 一、核心库源码

核心库位于 `phygnn/` 目录。

| 路径 | 作用 |
| --- | --- |
| `phygnn/__init__.py` | 包入口文件，导出常用类，例如 `PhysicsGuidedNeuralNetwork` 和 `PhygnnModel`。 |
| `phygnn/base.py` | 底层神经网络封装，负责 Keras 网络构建、权重管理、预测、保存和加载。 |
| `phygnn/phygnn.py` | PGNN 核心类。这里实现物理损失、普通监督损失、梯度下降、多保真 LF/HF 训练、残差基线 `k0` 等核心逻辑。 |
| `phygnn/layers/custom_layers.py` | 自定义层定义。需要特殊神经网络层时放在这里。 |
| `phygnn/layers/handlers.py` | 根据配置字典构建 Keras 层，负责把 `hidden_layers` 等配置转换为实际网络结构。 |
| `phygnn/model_interfaces/base_model.py` | 模型接口基类，负责特征列、标签列、标准化、DataFrame/ndarray 转换等通用数据处理。 |
| `phygnn/model_interfaces/tf_model.py` | 通用 TensorFlow 模型接口，不限定物理损失，适合普通神经网络训练和保存。 |
| `phygnn/model_interfaces/phygnn_model.py` | PGNN 高层接口。针对 BN/Al2O3 新增了 `build_bn_al2o3()`、`train_two_stage_multifidelity()`、`predict_k()` 等入口。 |
| `phygnn/utilities/loss_metrics.py` | 损失函数和评价指标工具。 |
| `phygnn/utilities/pre_processing.py` | 数据预处理工具。 |
| `phygnn/utilities/tf_utilities.py` | TensorFlow 相关工具函数。 |

### BN/Al2O3 领域子包 `phygnn/bn_al2o3/`

把 BN/Al2O3 工作流共享的常量、特征变换、评价指标、物理模型和数据流水线
集中到一个随安装包分发的子包里，作为单一事实来源，避免各脚本重复实现。

| 模块 | 作用 |
| --- | --- |
| `bn_al2o3/constants.py` | 原始/对数特征列名、标签名、各类自动探测候选列（观测热导率、预测热导率、分组、合并键、元数据、可选输入列）。 |
| `bn_al2o3/features.py` | 单一来源的 `format_features()` 对数变换；`make_bo_feature_matrix()`、`make_feature_keys()`、`infer_column()`。 |
| `bn_al2o3/metrics.py` | `compute_regression_metrics()`，统一计算 R2/RMSE/MAE/MAPE/Bias 等回归指标。 |
| `bn_al2o3/physics.py` | 纯数学：Dinger-Funk 目标级配、Lewis-Nielsen 热导率、phi_max 估计、体积分数采样与裁剪。 |
| `bn_al2o3/data.py` | pandas/CSV 数据流水线：粉体库读写、DF 三峰筛选、LF 数据生成、装填数据合并、训练表保存。 |
| `bn_al2o3/paths.py` | `repo_root()` 与 `data_path()/models_path()/outputs_path()`，统一默认路径定位到 `data/models/outputs`。 |

## 二、BN/Al2O3 实验工作流

新增工作流脚本位于 `workflow/` 目录。推荐按下面顺序使用：

```text
workflow/select_df_particle_sizes.py
    -> workflow/generate_lf_bn_al2o3.py
    -> workflow/train_bn_al2o3.py
    -> workflow/plot_bn_al2o3_metrics.py
    -> workflow/bayesian_optimize_bn_al2o3.py
    -> workflow/suggest_next_experiments.py
    -> workflow/generate_experiment_recipes.py
    -> workflow/prepare_hf_data.py
    -> workflow/train_bn_al2o3.py
```

| 脚本 | 作用 | 主要输入 | 主要输出 |
| --- | --- | --- | --- |
| `workflow/select_df_particle_sizes.py` | 用 Dinger-Funk 理论筛选小/中/大三峰 Al2O3 粒径组合和配比。 | 粉体库 CSV：`powder_id,D10,D50,D90` | `df_particle_candidates.csv` |
| `workflow/generate_lf_bn_al2o3.py` | 根据 DF 候选组合生成低保真数据，并用 Lewis-Nielsen 模型计算 `k_LF`。 | `df_particle_candidates.csv`，可选粉体库和装填数据 | `lf_bn_al2o3_data.csv` |
| `workflow/train_bn_al2o3.py` | 训练 BN/Al2O3 PGNN，并输出导热系数预测。支持 LF 8:2 训练/测试划分、早停、loss 曲线。 | LF 数据，可选 HF 数据 | `models/bn_al2o3_model/`、`outputs/bn_al2o3_predictions.csv`、loss 图 |
| `workflow/plot_bn_al2o3_metrics.py` | 读取预测结果，计算 R2、RMSE、MAE、MAPE、Bias 等指标，并生成一张综合评价图片。 | `lf_test_predictions.csv` 或其他包含真实值和预测值的 CSV | `bn_al2o3_metrics.png`、`bn_al2o3_metrics_summary.csv` |
| `workflow/bayesian_optimize_bn_al2o3.py` | 以 PGNN Top 10 作为初始观测，用 GP+EI/UCB 贝叶斯优化在候选池中二次筛选。有 HF 数据时用真实 `k_HF` 训练 GP，没有时用 `k_pred` 伪观测筛选。 | `bn_al2o3_predictions.csv`，可选模型和 HF 数据 | `bo_experiment_suggestions.csv` |
| `workflow/suggest_next_experiments.py` | 根据训练好的模型或已有 `k_pred` 推荐下一批实验配方。 | 候选预测表，可选训练好的模型和 HF 数据 | `next_experiment_suggestions.csv` |
| `workflow/generate_experiment_recipes.py` | 把模型配方转换成实验室可称量的质量配方。 | 推荐实验表或预测表 | `experiment_recipes.csv` |
| `workflow/prepare_hf_data.py` | 整理真实高保真实验数据，合并配方信息、重复实验取均值、划分 dev/blind。 | 实验测量 CSV，可选实验配方表 | `hf_bn_al2o3_data.csv` |

## 三、数据文件分层

建议把正式实验数据放到 `data/`，把训练模型放到 `models/`，避免和示例脚本混在一起。

```text
data/
  al2o3_powders.csv              原始 Al2O3 粉体库
  df_particle_candidates.csv      DF 筛选候选
  lf_bn_al2o3_data.csv            低保真数据
  raw_hf_measurements.csv         实验原始测量数据
  hf_bn_al2o3_data.csv            整理后的高保真数据

models/
  bn_al2o3_pgnn/                  保存的 PGNN 模型

outputs/
  bn_al2o3_predictions.csv        模型预测结果
  bo_experiment_suggestions.csv   贝叶斯优化二次筛选结果
  next_experiment_suggestions.csv 下一批实验推荐
  experiment_recipes.csv          可称量实验配方
  bn_al2o3_loss_curve.png         train_loss / val_loss 曲线
  bn_al2o3_metrics.png            R2/RMSE 等模型评价图
  bn_al2o3_metrics_summary.csv    R2/RMSE 等模型评价表
```

脚本默认路径已通过 `phygnn.bn_al2o3.paths` 路由到仓库根的 `data/`、`models/`、`outputs/` 分层目录：数据落到 `data/`、模型落到 `models/`、预测与图表落到 `outputs/`。也可以继续用命令行参数显式覆盖到任意路径。

## 四、模型输入和标签

当前第一版 BN/Al2O3 PGNN 使用 6 个输入特征：

```text
D_s, D_m, D_l, phi_s, phi_m, E
```

其中：

| 字段 | 含义 |
| --- | --- |
| `D_s` | 小粒径 Al2O3 的代表粒径，当前用 D50。 |
| `D_m` | 中粒径 Al2O3 的代表粒径，当前用 D50。 |
| `D_l` | 大粒径 Al2O3 的代表粒径，当前用 D50。 |
| `phi_s` | 小粒径 Al2O3 在 Al2O3 总体积中的占比。 |
| `phi_m` | 中粒径 Al2O3 在 Al2O3 总体积中的占比。 |
| `E` | 混合 PSD 与 Dinger-Funk 目标级配曲线之间的误差。 |

`phi_l = 1 - phi_s - phi_m`，所以当前没有作为神经网络输入。

训练标签统一用：

```text
k
```

低保真数据里 `k = k_LF`，高保真数据里 `k = k_HF`。

## 五、训练逻辑

`train_bn_al2o3.py` 现在包含这些训练控制：

| 功能 | 参数 |
| --- | --- |
| LF 外部训练/测试划分 | `--lf-test-size 0.2` |
| 训练集内部验证集 | `--validation-split 0.2` |
| LF 预训练 epoch | `--stage1-epochs` |
| HF 校正 epoch | `--stage2-epochs` |
| 早停 | `--patience`、`--early-stop-monitor`、`--early-stop-min-delta` |
| 保存 loss 曲线 | `--loss-plot-output` |
| 保存历史表 | `--history-output` |

没有 HF 数据时，只会执行 LF-only 预训练。有 HF 数据时，会执行 LF 预训练加 LF/HF 校正两阶段训练。

## 六、测试和验证

| 目录 | 作用 |
| --- | --- |
| `tests/` | 单元测试和接口测试。 |
| `tests/test_phygnn.py` | 底层 PGNN 功能测试。 |
| `tests/test_phygnn_interface.py` | 高层 `PhygnnModel` 接口测试，包含 BN/Al2O3 多保真接口测试。 |
| `tests/test_tf_model.py` | 通用 TensorFlow 模型接口测试。 |
| `tests/test_layers.py` | 层构建和自定义层测试。 |

常用验证命令：

```powershell
D:\anaconda\ANaconda\envs\phygnn\python.exe -m pytest tests -q
D:\anaconda\ANaconda\envs\phygnn\python.exe -m ruff check phygnn workflow tests
```

## 七、改进记录 / Changelog

本次重构（完整重构 + 脚本迁移）：

- 新增领域子包 `phygnn/bn_al2o3/`，集中常量、特征变换、评价指标、物理模型、数据流水线和路径定位。
- `PhygnnModel.format_bn_al2o3_features()` 改为委托 `phygnn.bn_al2o3.features.format_features()`，消除对数变换的重复实现；`BN_AL2O3_FEATURE_NAMES` 仍保持兼容。
- 8 个 BN/Al2O3 脚本从 `examples/` 迁移到 `workflow/`，改写为薄 CLI，只导入 `phygnn.bn_al2o3.*` 和 `phygnn`，消除了脚本间的脆弱互导入（`from generate_lf_bn_al2o3 import ...` 等）。
- 默认输出路径由 `Path(__file__).with_name(...)` 改为 `data_path()/models_path()/outputs_path()`，输出不再污染 `examples/`。
- 修正 `pyproject.toml` 的 `[tool.setuptools] packages`，显式列出全部子包（含 `phygnn.bn_al2o3`）。
