# MobileNetV4 主干网络迁移说明

## 概述

已成功将主干网络从 VGGish 迁移到 MobileNetV4。新架构具有以下优势：
- 更轻量级的模型（参数量：~250万）
- 更快的推理速度
- 保留了预训练权重的迁移学习能力

## 文件修改

### 1. 新增文件
- `models/mobilenetv4_model.py`: MobileNetV4 模型实现

### 2. 修改文件
- `audio_distill_los_system.py`: 更新 `_create_audio_backbone()` 方法以支持两种主干网络
- `main.py`: 添加 `--backbone` 参数，更新默认预训练模型路径

## 使用方法

### 使用 MobileNetV4（默认）

```bash
python main.py \
    --backbone mobilenetv4 \
    --pretrained_model models/model.safetensors \
    --data_dir data/ \
    --source_domain cold_zone \
    --target_domain hot_zone_fine
```

### 使用 VGGish（保持向后兼容）

```bash
python main.py \
    --backbone vggish \
    --pretrained_model models/vggish-10086976.pth \
    --data_dir data/ \
    --source_domain cold_zone \
    --target_domain hot_zone_fine
```

### 自动检测（基于预训练模型路径）

如果预训练模型路径包含 "vggish"，系统会自动使用 VGGish；否则使用 MobileNetV4。

```bash
# 自动使用 MobileNetV4
python main.py --pretrained_model models/model.safetensors

# 自动使用 VGGish
python main.py --pretrained_model models/vggish-10086976.pth
```

## 技术细节

### 预训练权重适配

MobileNetV4 的预训练权重来自 ImageNet（3通道RGB图像），而音频频谱图是单通道的。系统会自动处理这一差异：

1. 第一层卷积：对RGB三通道权重取平均，转换为单通道
2. 其他层：直接加载预训练权重
3. 分类头：随机初始化（适配具体任务）

加载日志示例：
```
Using timm MobileNetV4
Detected feature dimension: 1280
Loading pretrained MobileNetV4 weights from models/model.safetensors
Adapted first conv layer: torch.Size([32, 3, 3, 3]) -> torch.Size([32, 1, 3, 3])
Successfully loaded 276 pretrained parameters
```

### 模型架构

- **Backbone**: MobileNetV4 (timm 实现)
- **输入**: 单通道 Mel 频谱图 (1, 96, time_frames)
- **特征维度**: 1280
- **分类头**: Dropout(0.2) + Linear(1280 -> num_classes)

### 依赖项

确保安装以下依赖：
```bash
pip install torch torchvision
pip install timm  # 用于 MobileNetV4 实现
pip install safetensors  # 用于加载 .safetensors 格式的权重
```

如果没有安装 timm，系统会自动回退到简化版的 MobileNetV4 实现（基于深度可分离卷积）。

## 性能对比

### 模型参数量
- VGGish: ~1.4M 参数（估计）
- MobileNetV4: ~2.5M 参数

### 特征维度
- VGGish: 128-dim embeddings
- MobileNetV4: 1280-dim features

## 向后兼容性

所有原有的训练脚本和配置文件都保持兼容：
- `train_all.sh`: 只需添加 `--backbone mobilenetv4` 参数
- `train_stage2_only.sh`: 同上
- Stage 1 和 Stage 2 的训练流程完全不变

## 故障排除

### 问题 1: CUDA 内存不足
如果使用 MobileNetV4 时遇到 CUDA 内存问题，可以：
- 减小 batch_size
- 或者回退到 VGGish

### 问题 2: timm 不可用
如果 timm 库不可用，系统会自动使用简化版实现。建议安装 timm 以获得最佳性能：
```bash
pip install timm
```

### 问题 3: 预训练权重加载失败
确保预训练权重文件路径正确：
- MobileNetV4: `models/model.safetensors`
- VGGish: `models/vggish-10086976.pth`

## 下一步

1. 使用新的主干网络运行完整的训练流程
2. 比较 VGGish 和 MobileNetV4 的性能
3. 根据实验结果选择最佳的主干网络

## 联系方式

如有问题或建议，请联系项目维护者。

