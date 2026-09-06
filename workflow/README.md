# BN/Al2O3 PGNN 实验工作流

这个目录包含 BN/Al2O3 多保真实验的 8 个命令行脚本。原始的 `phygnn` 示例
notebook 仍在 `examples/` 目录，这里只放 BN/Al2O3 工作流脚本。

所有脚本都是薄 CLI：只导入 `phygnn.bn_al2o3.*` 和 `phygnn`，共享逻辑集中在
`phygnn/bn_al2o3/` 子包里。默认输出路径通过 `phygnn.bn_al2o3.paths` 自动路由到
仓库根的 `data/`、`models/`、`outputs/`，也可以用命令行参数显式覆盖。

## 一、脚本职责

| 文件 | 什么时候用 | 功能 |
| --- | --- | --- |
| `select_df_particle_sizes.py` | 实验开始前 | 用 Dinger-Funk 理论从粉体库中筛选三峰粒径组合。 |
| `generate_lf_bn_al2o3.py` | 生成训练数据前 | 从 DF 候选中生成低保真数据，并计算 `k_LF`。 |
| `train_bn_al2o3.py` | 模型训练和预测 | 训练 PGNN、保存模型、预测导热系数、保存 loss 曲线。 |
| `plot_bn_al2o3_metrics.py` | 模型训练后 | 从预测结果 CSV 计算 R2、RMSE、MAE、MAPE，并保存评价指标图片。 |
| `bayesian_optimize_bn_al2o3.py` | PGNN 预测后 | 先取 PGNN Top 10，再用 GP+EI 贝叶斯优化二次筛选。 |
| `suggest_next_experiments.py` | 选择下一批实验 | 用模型预测值推荐 Top N 实验配方。 |
| `generate_experiment_recipes.py` | 做实验前 | 把体积分数配方转换成克级称量配方。 |
| `prepare_hf_data.py` | 实验测完后 | 整理真实 `k_HF`，重复实验求均值，生成可训练 HF 数据。 |

## 二、推荐运行顺序

> 下面命令从仓库根目录 `D:\PINN\phygnn-main\phygnn` 运行。若不加显式路径，
> 脚本会默认把数据写到 `data/`、模型写到 `models/`、预测和图表写到 `outputs/`。

### 1. 准备粉体库

粉体库最少需要这些列：

```csv
powder_id,D10,D50,D90
Al2O3_A,0.28,0.50,0.90
Al2O3_B,1.80,3.00,5.20
```

也可以先生成模板：

```powershell
D:\anaconda\ANaconda\envs\phygnn\python.exe workflow\select_df_particle_sizes.py `
  --write-template data\al2o3_powder_template.csv
```

### 2. 用 DF 理论筛选粒径组合

```powershell
D:\anaconda\ANaconda\envs\phygnn\python.exe workflow\select_df_particle_sizes.py `
  --powders data\al2o3_powders.csv `
  --output data\df_particle_candidates.csv `
  --top-n 100
```

输出核心列：

```text
D_s,D_m,D_l,phi_s,phi_m,phi_l,E
```

其中 `E` 是混合 PSD 与 Dinger-Funk 理想级配曲线的积分平方误差，不是随机数。

### 3. 生成低保真数据

```powershell
D:\anaconda\ANaconda\envs\phygnn\python.exe workflow\generate_lf_bn_al2o3.py `
  --df-candidates data\df_particle_candidates.csv `
  --powders data\al2o3_powders.csv `
  --n-samples 3000 `
  --ratio-concentration 80 `
  --output data\lf_bn_al2o3_data.csv
```

输出的 `k_LF` 是 Lewis-Nielsen 半经验模型计算的低保真标签。

### 4. 训练 PGNN 并预测导热系数

只有 LF 数据时：

```powershell
D:\anaconda\ANaconda\envs\phygnn\python.exe workflow\train_bn_al2o3.py `
  --lf-data data\lf_bn_al2o3_data.csv `
  --model-output models\bn_al2o3_pgnn `
  --predictions-output outputs\bn_al2o3_predictions.csv `
  --history-output outputs\bn_al2o3_training_history.csv `
  --loss-plot-output outputs\bn_al2o3_loss_curve.png `
  --stage1-epochs 1000 `
  --patience 50 `
  --early-stop-min-delta 0.0001
```

有 HF 数据时：

```powershell
D:\anaconda\ANaconda\envs\phygnn\python.exe workflow\train_bn_al2o3.py `
  --lf-data data\lf_bn_al2o3_data.csv `
  --hf-data data\hf_bn_al2o3_data.csv `
  --hf-label-col k_HF `
  --model-output models\bn_al2o3_pgnn `
  --predictions-output outputs\bn_al2o3_predictions.csv `
  --stage1-epochs 500 `
  --stage2-epochs 1000
```

训练脚本会自动：

```text
LF 数据 8:2 划分
保存 LF train/test
保存 training_history.csv
保存 train_loss / val_loss 图片
输出 LF test MAE/RMSE
```

### 4.1 生成 R2/RMSE 等评价指标图片

训练结束后，可以直接用测试集预测结果画评价图：

```powershell
D:\anaconda\ANaconda\envs\phygnn\python.exe workflow\plot_bn_al2o3_metrics.py `
  --input outputs\lf_test_predictions.csv `
  --output outputs\bn_al2o3_metrics.png `
  --summary-output outputs\bn_al2o3_metrics_summary.csv
```

如果使用脚本默认路径，也可以直接运行：

```powershell
D:\anaconda\ANaconda\envs\phygnn\python.exe workflow\plot_bn_al2o3_metrics.py
```

生成的图片会包含：

```text
observed k vs predicted k
residuals
error distribution
R2 / RMSE / MAE / MAPE / Bias / MaxAE
```

### 5. 用贝叶斯优化二次筛选 PGNN Top 10

```powershell
D:\anaconda\ANaconda\envs\phygnn\python.exe workflow\bayesian_optimize_bn_al2o3.py `
  --candidates outputs\bn_al2o3_predictions.csv `
  --model models\bn_al2o3_pgnn `
  --pgnn-top-n 10 `
  --bo-top-n 5 `
  --output outputs\bo_experiment_suggestions.csv
```

如果已经有高保真实验数据，建议把 HF 数据也传进去：

```powershell
D:\anaconda\ANaconda\envs\phygnn\python.exe workflow\bayesian_optimize_bn_al2o3.py `
  --candidates outputs\bn_al2o3_predictions.csv `
  --model models\bn_al2o3_pgnn `
  --hf-data data\hf_bn_al2o3_data.csv `
  --hf-label-col k_HF `
  --pgnn-top-n 10 `
  --bo-top-n 5 `
  --output outputs\bo_experiment_suggestions.csv
```

没有 HF 数据时，BO 会使用 PGNN 的 `k_pred` 作为伪观测进行二次筛选。这可以用于第一轮实验排序，但不等同于真实实验反馈的贝叶斯优化。

默认设置下，脚本把 PGNN Top 10 作为 GP 的初始观测，然后在全部候选池中计算 acquisition score。若只想在 PGNN Top 10 内部二次排序，可加：

```powershell
--candidate-pool top-pgnn
```

### 6. 普通排序推荐下一批实验

```powershell
D:\anaconda\ANaconda\envs\phygnn\python.exe workflow\suggest_next_experiments.py `
  --candidates outputs\bn_al2o3_predictions.csv `
  --model models\bn_al2o3_pgnn `
  --hf-data data\hf_bn_al2o3_data.csv `
  --output outputs\next_experiment_suggestions.csv `
  --top-n 5 `
  --max-per-triplet 2
```

如果暂时没有保存模型，但候选表里已有 `k_pred`，可以加：

```powershell
--no-model
```

### 7. 转成实验可称量配方

```powershell
D:\anaconda\ANaconda\envs\phygnn\python.exe workflow\generate_experiment_recipes.py `
  --input outputs\bo_experiment_suggestions.csv `
  --output outputs\experiment_recipes.csv `
  --repeat-count 3 `
  --batch-mass-g 20 `
  --hardener-per-resin 0.3
```

输出会包含：

```text
al2o3_s_mass_g
al2o3_m_mass_g
al2o3_l_mass_g
resin_mass_g
hardener_mass_g
bn_mass_g
```

### 8. 整理真实 HF 实验数据

实验测完后，准备原始测量表：

```csv
experiment_id,k_HF,measurement_notes
EXP_00001_R01,3.21,ok
EXP_00001_R02,3.18,ok
EXP_00001_R03,3.24,ok
```

整理成模型训练可用表：

```powershell
D:\anaconda\ANaconda\envs\phygnn\python.exe workflow\prepare_hf_data.py `
  --raw-measurements data\raw_hf_measurements.csv `
  --recipes outputs\experiment_recipes.csv `
  --output data\hf_bn_al2o3_data.csv `
  --repeat-output outputs\hf_repeats_checked.csv `
  --blind-count 3
```

然后回到第 4 步，用 LF + HF 重新训练。

## 三、当前模型输入

当前第一版模型输入是 6 维：

```text
D_s,D_m,D_l,phi_s,phi_m,E
```

`phi_l` 通过 `1 - phi_s - phi_m` 得到。

如果后续实验中总填料体积分数 `V_f` 也要变化，需要把模型输入扩展为：

```text
D_s,D_m,D_l,phi_s,phi_m,E,V_f
```

这需要同步修改 `PhygnnModel.BN_AL2O3_FEATURE_NAMES`、LF 生成脚本、训练脚本和测试。
