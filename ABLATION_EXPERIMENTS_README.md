# 消融实验使用指南

本文档说明如何使用 `run_ablation_experiments.sh` 脚本运行三种消融实验配置。

## 实验配置说明

### (a) 无采样和长尾损失 (`no_sampling_no_loss`)
- **采样策略**: 均匀随机采样（禁用 `WeightedRandomSampler`）
- **损失函数**: 标准交叉熵损失（禁用 `Focal Loss` 和 `Logit Adjustment`）
- **目的**: 作为baseline实验，移除了过采样和特定于长尾的损失
- **命令**: `./run_ablation_experiments.sh a` 或 `./run_ablation_experiments.sh no_sampling_no_loss`

### (b) 仅无长尾损失 (`no_loss_only`)
- **采样策略**: `WeightedRandomSampler`（启用）
- **损失函数**: 标准交叉熵损失（禁用 `Focal Loss` 和 `Logit Adjustment`）
- **目的**: 验证采样器单独对长尾问题的处理能力
- **命令**: `./run_ablation_experiments.sh b` 或 `./run_ablation_experiments.sh no_loss_only`

### (c) 仅无采样 (`no_sampling_only`)
- **采样策略**: 均匀随机采样（禁用 `WeightedRandomSampler`）
- **损失函数**: `Focal Loss` + `Logit Adjustment`（启用）
- **目的**: 验证损失函数单独对长尾问题的处理能力
- **命令**: `./run_ablation_experiments.sh c` 或 `./run_ablation_experiments.sh no_sampling_only`

## 快速开始

### 基本使用

```bash
# 运行实验 (a): 无采样和长尾损失
./run_ablation_experiments.sh a

# 运行实验 (b): 仅无长尾损失
./run_ablation_experiments.sh b

# 运行实验 (c): 仅无采样
./run_ablation_experiments.sh c
```

### 带额外参数的使用

您可以在实验类型后添加任何 `main.py` 支持的参数：

```bash
# 使用自定义训练轮数
./run_ablation_experiments.sh a --epochs 50

# 使用自定义学习率和权重衰减
./run_ablation_experiments.sh b --lr 0.0001 --wd 1e-3

# 实验(c)中自定义Focal Loss参数
./run_ablation_experiments.sh c --focal_gamma 2.5 --logit_adj_tau 1.0

# 使用自定义数据路径
./run_ablation_experiments.sh a --data_dir /path/to/data --source_domain cold_zone --target_domain hot_13.31

# 跳过Stage 1并使用已有模型
./run_ablation_experiments.sh b --skip_stage1 --stage1_model_path output/xxx/best_model_stage1_audio.pth

# Stage 2相关参数
./run_ablation_experiments.sh c --finetune_epoch 30 --finetune_lr 5e-5 --distill_weight 0.8
```

## 输出目录

脚本会自动创建带时间戳的输出目录：

- 实验 (a): `output/ablation_a_no_sampling_no_loss_<timestamp>/`
- 实验 (b): `output/ablation_b_no_loss_only_<timestamp>/`
- 实验 (c): `output/ablation_c_no_sampling_only_<timestamp>/`

其中 `<timestamp>` 格式为 `月-日_时-分`，例如 `11-16_10-45`。

## 实验对比

| 实验 | WeightedRandomSampler | Focal Loss | Logit Adjustment | 交叉熵 |
|------|----------------------|------------|------------------|--------|
| (a)  | ❌ 禁用               | ❌ 禁用     | ❌ 禁用           | ✅ 启用 |
| (b)  | ✅ 启用               | ❌ 禁用     | ❌ 禁用           | ✅ 启用 |
| (c)  | ❌ 禁用               | ✅ 启用     | ✅ 启用           | ❌ 禁用 |

## 参数映射

脚本自动处理以下参数映射：

### 实验 (a)
```python
--no_weighted_sampler
# 不添加 --use_focal_loss 和 --use_logit_adjustment（默认禁用）
```

### 实验 (b)
```python
# 不添加 --no_weighted_sampler（默认启用采样器）
# 不添加 --use_focal_loss 和 --use_logit_adjustment（默认禁用）
```

### 实验 (c)
```python
--no_weighted_sampler
--use_focal_loss
--use_logit_adjustment
```

## 常用参数参考

### Stage 1 参数（源域训练）
- `--epochs`: 训练轮数（默认：100）
- `--lr`: 学习率（默认：0.0001）
- `--wd`: 权重衰减（默认：1e-3）
- `--batch_size`: 批次大小（默认：32）
- `--early_stopping_patience`: 早停耐心值（默认：20）

### Stage 2 参数（目标域微调和蒸馏）
- `--finetune_epoch`: Stage 2训练轮数（默认：25）
- `--finetune_lr`: Stage 2学习率（默认：3e-5）
- `--finetune_wd`: Stage 2权重衰减（默认：1e-4）
- `--distill_weight`: 蒸馏损失权重（默认：1.0）
- `--distill_temperature`: 蒸馏温度（默认：4.0）

### 长尾损失参数（仅实验c）
- `--focal_gamma`: Focal Loss的gamma参数（默认：2.0）
- `--focal_alpha`: Focal Loss的alpha参数（默认：None）
- `--logit_adj_tau`: Logit Adjustment的tau参数（默认：1.0）

### 数据集参数
- `--data_dir`: 数据根目录（默认：'data/'）
- `--source_domain`: 源域文件夹名（默认：'cold_zone'）
- `--target_domain`: 目标域文件夹名（默认：'hot_13.31'）

## 完整示例

### 示例 1: 运行所有三个实验（默认参数）

```bash
# 实验 (a)
./run_ablation_experiments.sh a

# 实验 (b)
./run_ablation_experiments.sh b

# 实验 (c)
./run_ablation_experiments.sh c
```

### 示例 2: 快速测试（少量轮数）

```bash
./run_ablation_experiments.sh a --epochs 10 --finetune_epoch 5
./run_ablation_experiments.sh b --epochs 10 --finetune_epoch 5
./run_ablation_experiments.sh c --epochs 10 --finetune_epoch 5
```

### 示例 3: 使用已有Stage 1模型

```bash
STAGE1_MODEL="output/cold_zone_to_hot_13.31_11-15_14-30/best_model_stage1_audio.pth"

./run_ablation_experiments.sh a --skip_stage1 --stage1_model_path $STAGE1_MODEL
./run_ablation_experiments.sh b --skip_stage1 --stage1_model_path $STAGE1_MODEL
./run_ablation_experiments.sh c --skip_stage1 --stage1_model_path $STAGE1_MODEL
```

### 示例 4: 调整实验(c)的长尾损失参数

```bash
# 调整Focal Loss的gamma值
./run_ablation_experiments.sh c --focal_gamma 1.5

# 同时调整Focal Loss和Logit Adjustment
./run_ablation_experiments.sh c --focal_gamma 2.5 --logit_adj_tau 0.8

# 完全禁用Logit Adjustment（只用Focal Loss）
./run_ablation_experiments.sh c --logit_adj_tau 0
```

## 批量运行脚本示例

如果您想依次运行所有实验，可以创建以下批处理脚本：

```bash
#!/bin/bash
# run_all_ablations.sh

echo "开始运行所有消融实验..."

# 共享参数
COMMON_ARGS="--epochs 100 --finetune_epoch 25 --batch_size 32"

# 运行实验 (a)
echo "================================"
echo "运行实验 (a)"
echo "================================"
./run_ablation_experiments.sh a $COMMON_ARGS

# 运行实验 (b)
echo "================================"
echo "运行实验 (b)"
echo "================================"
./run_ablation_experiments.sh b $COMMON_ARGS

# 运行实验 (c)
echo "================================"
echo "运行实验 (c)"
echo "================================"
./run_ablation_experiments.sh c $COMMON_ARGS

echo "所有实验已完成！"
```

保存为 `run_all_ablations.sh`，然后运行：
```bash
chmod +x run_all_ablations.sh
./run_all_ablations.sh
```

## 结果分析

训练完成后，您可以在各自的输出目录中找到：

- `train_log.txt`: 训练日志
- `best_model_stage1_audio.pth`: Stage 1最佳模型
- `best_model_stage2_audio.pth`: Stage 2最佳模型（如果有）
- `*.png`: 训练过程可视化图表

比较三个实验的结果，分析：
- 各实验在源域和目标域上的性能
- 不同配置对长尾类别的影响
- 采样策略 vs 损失函数在处理类别不平衡时的效果

## 故障排除

### 权限错误
```bash
chmod +x run_ablation_experiments.sh
```

### Python环境
确保已安装所有依赖：
```bash
pip install torch torchaudio torch-audiomentations scikit-learn matplotlib tqdm
```

### 查看完整的main.py参数
```bash
python main.py --help
```

## 更多信息

详细的模型架构和训练流程，请参考：
- `main.py`: 主训练脚本
- `audio_cross_domain_trainer.py`: 跨域训练器实现
- `losses/long_tail_losses.py`: 长尾损失函数实现
