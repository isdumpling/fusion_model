# FLOPs 指标使用说明

## 📊 功能概述

已成功为你的项目添加 **FLOPs（浮点运算次数）** 指标计算功能，用于评估模型的计算复杂度。

## ✨ 主要特点

1. ✅ **自动计算**: 训练开始时自动计算模型的 FLOPs 和参数量
2. ✅ **完整记录**: 将指标记录到控制台和训练日志文件中
3. ✅ **易读格式**: 使用 M (百万) 和 G (十亿) 等单位显示结果
4. ✅ **兼容性**: 支持动态初始化的模型（如本项目的 VGGish）

## 📝 修改内容

### 1. 新增工具函数 (`utils/common.py`)

新增了 `calculate_flops()` 函数，用于计算模型的 FLOPs 和参数量：

```python
calculate_flops(model, input_shape=(1, 1, 96, 64), device='cuda')
```

**功能特点：**
- 自动处理动态初始化的模型
- 计算失败时至少返回参数量统计
- 提供格式化和精确的两种输出

### 2. 集成到训练器 (`audio_cross_domain_trainer.py`)

在 `CrossDomainAudioTrainer` 类的 `__init__()` 方法中：
- 自动调用 `_compute_and_log_flops()` 方法
- 在模型初始化后立即计算 FLOPs
- 将结果记录到日志文件

## 🚀 使用方法

### 方法一：直接运行训练（推荐）

```bash
conda activate fusion_model
python main.py --ablation_config combo4
```

训练开始时会自动显示并记录 FLOPs 指标。

### 方法二：运行示例脚本

```bash
conda activate fusion_model
python example_with_flops.py
```

### 方法三：在代码中调用

```python
from utils.common import calculate_flops
from models.vggish_model import create_vggish_model

model = create_vggish_model(num_classes=2)
model = model.cuda()

flops, params, flops_fmt, params_fmt = calculate_flops(
    model, 
    input_shape=(1, 1, 96, 64),
    device='cuda'
)

print(f"FLOPs: {flops_fmt}")
print(f"参数量: {params_fmt}")
```

## 📊 输出示例

### 控制台输出

```
============================================================
计算模型复杂度...
============================================================
FLOPs: 863.896M
Parameters: 72.141M
============================================================
```

### 训练日志 (training_log.txt)

```
============================================================
MODEL COMPLEXITY
============================================================
FLOPs: 863.896M
Parameters: 72.141M
FLOPs (exact): 863895808
Parameters (exact): 72141442
============================================================
```

## 📈 当前模型复杂度

**VGGish 模型（2分类）:**

| 指标 | 数值 | 说明 |
|------|------|------|
| **FLOPs** | 863.896M | 约 0.864 GFLOPs |
| **参数量** | 72.141M | 约 72.14 百万参数 |
| **输入尺寸** | (1, 1, 96, 64) | batch × channels × n_mels × time_frames |

## 🔧 技术细节

### 依赖库

- **thop** (torch-ops-counter): 用于计算 FLOPs
  - 已安装在 `fusion_model` 环境中
  - 版本: 0.1.1.post2209072238

### 计算逻辑

1. 在模型上执行一次前向传播（初始化动态层）
2. 使用 `thop.profile()` 分析模型计算量
3. 使用 `thop.clever_format()` 格式化输出
4. 如果计算失败，至少统计参数量

### 关键代码位置

- **工具函数**: `utils/common.py` → `calculate_flops()`
- **训练器集成**: `audio_cross_domain_trainer.py` → `_compute_and_log_flops()`
- **调用位置**: `audio_cross_domain_trainer.py` → `__init__()` 第 52 行

## 💡 FLOPs 说明

**FLOPs (Floating Point Operations)** 是衡量模型计算复杂度的标准指标：

- **1 FLOP** = 1 次浮点运算
- **1 MFLOPs** = 100万次浮点运算 (10⁶)
- **1 GFLOPs** = 10亿次浮点运算 (10⁹)

**实际意义：**
- ⚡ 更低的 FLOPs → 更快的推理速度
- 🔋 更低的 FLOPs → 更低的能耗
- 📱 更低的 FLOPs → 更适合移动端部署

## 📂 相关文件

```
fusion_model/
├── utils/
│   └── common.py                    # FLOPs计算函数
├── audio_cross_domain_trainer.py    # 训练器（已集成FLOPs）
├── example_with_flops.py            # 使用示例脚本
├── README_FLOPS.md                  # 英文说明
└── FLOPS使用说明.md                 # 本文档
```

## ✅ 验证测试

所有功能已通过测试：

```bash
# 测试1: 导入测试
✓ 函数导入成功

# 测试2: 独立计算测试
✓ FLOPs计算成功: 863.896M
✓ 参数量计算成功: 72.141M

# 测试3: 训练器集成测试
✓ 训练器导入成功
✓ FLOPs自动计算功能正常
```

## 🎯 下一步

现在你可以：

1. **运行完整训练**，FLOPs 指标会自动记录
   ```bash
   python main.py --ablation_config combo4
   ```

2. **查看训练日志**，在日志开头会看到模型复杂度信息
   ```bash
   cat output/cold_zone_to_hot_zone_combo4_*/training_log.txt
   ```

3. **对比不同配置**的模型复杂度（如果修改了模型结构）

## ❓ 常见问题

**Q: 为什么需要先做一次前向传播？**  
A: VGGish 模型的 embeddings 层是动态初始化的，需要先运行一次前向传播来确定输入维度。

**Q: 如果 thop 计算失败怎么办？**  
A: 函数会捕获异常并尝试至少统计模型参数量，不会影响训练流程。

**Q: FLOPs 和 MACs 有什么区别？**  
A: MACs (Multiply-Accumulate Operations) 通常约等于 FLOPs/2，但本项目统一使用 FLOPs。

---

**添加时间**: 2025年10月3日  
**测试状态**: ✅ 已通过所有测试  
**环境**: fusion_model (conda)
