# 课程学习功能使用说明

## 概述

本项目已经实现了基于置信度的课程学习（Curriculum Learning）策略，用于Stage 2训练阶段。该策略通过教师模型的置信度来逐步选择训练样本，从最有把握的样本开始，逐渐放宽标准，直到所有目标域数据都被纳入训练。

## 新增的命令行参数

### `--use_curriculum_learning`
- **类型**: 布尔标志 (action='store_true')
- **默认值**: False (禁用)
- **说明**: 启用基于置信度的课程学习策略

### `--initial_confidence_threshold`
- **类型**: float
- **默认值**: 0.98
- **说明**: 课程学习的初始置信度阈值。在训练初期，只有教师模型置信度超过此阈值的样本会被用于训练

### `--final_confidence_threshold`
- **类型**: float
- **默认值**: 0.70
- **说明**: 课程学习的最终置信度阈值。随着训练进行，阈值会从初始值线性衰减到此值，逐步包含更多样本

## 使用示例

### 基本用法（启用课程学习）

```bash
python main.py \
    --source_domain cold_zone \
    --target_domain hot_zone \
    --use_curriculum_learning \
    --initial_confidence_threshold 0.98 \
    --final_confidence_threshold 0.70
```

### 自定义阈值

```bash
python main.py \
    --source_domain cold_zone \
    --target_domain hot_zone \
    --use_curriculum_learning \
    --initial_confidence_threshold 0.95 \
    --final_confidence_threshold 0.60
```

### 不使用课程学习（默认行为）

```bash
python main.py \
    --source_domain cold_zone \
    --target_domain hot_zone
```

## 工作原理

### 1. 预计算阶段
在 Stage 2 训练开始前，系统会：
- 使用 Stage 1 训练的最佳模型（教师模型）
- 对所有目标域训练数据进行一次推理
- 保存每个样本的索引和教师模型的预测置信度

### 2. 动态筛选阶段
在每个 epoch 中：
- 根据当前 epoch，计算当前的置信度阈值（线性衰减）
  ```
  current_threshold = initial_threshold - (progress × (initial_threshold - final_threshold))
  ```
  其中 `progress = current_epoch / (total_epochs - 1)`

- 筛选出置信度 ≥ 当前阈值的样本
- 创建只包含这些样本的临时数据加载器
- 使用筛选后的样本进行当前 epoch 的训练

### 3. 日志输出
训练过程中会输出：
- 置信度统计信息（最小值、最大值、平均值、中位数）
- 每个 epoch 使用的样本数量和当前阈值

## 实现细节

### 修改的文件

1. **main.py**
   - 添加了三个新的命令行参数

2. **audio_cross_domain_trainer.py**
   - `train_stage2()`: 添加了预计算和动态筛选逻辑
   - `_train_stage2_epoch()`: 修改为接受动态数据加载器
   - `_precompute_pseudo_labels()`: 新增，用于预计算置信度
   - `_compute_current_threshold()`: 新增，计算当前epoch的阈值
   - `_create_filtered_loader()`: 新增，创建筛选后的数据加载器

### 核心优势

1. **渐进式学习**: 从简单样本到困难样本，符合人类学习规律
2. **噪声抑制**: 初期排除低置信度样本，减少伪标签噪声
3. **灵活配置**: 通过调整阈值参数，可以适应不同的数据集和场景
4. **无缝集成**: 可以与现有的消融实验配置（Focal Loss, Logit Adjustment等）完美结合

## 参数调优建议

- **高质量数据集**: 可以降低初始阈值（如0.95），让更多样本参与训练
- **低质量数据集**: 可以提高初始阈值（如0.99），确保初期只使用高质量样本
- **快速收敛**: 设置较低的最终阈值（如0.60），在训练后期包含更多样本
- **保守策略**: 设置较高的最终阈值（如0.80），保持对样本质量的严格要求

## 注意事项

1. 课程学习只在 Stage 2 训练阶段生效
2. 如果某个 epoch 没有样本满足阈值要求，系统会自动使用全部数据
3. 预计算过程会在 Stage 2 开始前执行一次，可能需要几分钟时间
4. 置信度阈值采用线性衰减策略，未来可以扩展为其他衰减策略（如指数衰减）

## 与其他功能的兼容性

课程学习功能可以与以下功能同时使用：
- WeightedRandomSampler
- Focal Loss
- Logit Adjustment
- 各种消融实验配置 (combo1-combo5)

## 示例完整命令

```bash
python main.py \
    --data_dir data/ \
    --source_domain cold_zone \
    --target_domain hot_zone \
    --epochs 100 \
    --finetune_epoch 50 \
    --batch_size 16 \
    --lr 0.0001 \
    --finetune_lr 0.001 \
    --use_curriculum_learning \
    --initial_confidence_threshold 0.98 \
    --final_confidence_threshold 0.70 \
    --use_weighted_sampler \
    --use_focal_loss \
    --focal_gamma 2.0
```

这个命令会同时启用课程学习、加权采样和Focal Loss，实现多种技术的组合优化。
