# Stage 2 修改总结

## 修改背景

由于更换了数据集（从 `hot_zone` 改为 `hot_zone_fine`），新数据集具有以下特点：
- 时长约 1.4 秒，比 Stage 1 使用的 `cold_zone`（≤0.64秒）稍长
- 都是 cough 和 non_cough 两类
- 不需要复杂的滑动窗口处理
- **关键特征**：约 20% 的咳嗽声在 0.64 秒之后才达到能量峰值

因此，Stage 2 的数据处理方式改为与 Stage 1 相同的简单处理方式，但增加了智能窗口选择机制。

## 主要修改内容

### 1. 数据加载器修改 (`datasets/dataloader.py`)

#### 修改前（初版）：
- 训练模式：固定长度 64 帧
- 测试模式：
  - 源域：固定长度 64 帧
  - **目标域：保持原始长度（用于滑动窗口）**

#### 修改后（初版，有问题）：
- **训练和测试模式统一：都使用固定长度 64 帧**
- 简单截取前 64 帧
- ❌ **问题**：导致晚发咳嗽被截断，F1 score 从 0.81 降到 0.22

#### 最终版本（智能窗口选择）：
- **训练和测试模式统一：都使用固定长度 64 帧**
- ✅ **智能窗口选择**：
  - 当音频长度 > 64 帧时，不是简单截取前 64 帧
  - 计算所有可能的 64 帧窗口的 RMS 能量
  - 选择能量最高的窗口（捕捉咳嗽的峰值部分）
  - 频谱图和波形同步选择相同的时间窗口
- 适用于不同时长的数据（短时数据填充，长时数据智能截取）

### 2. 训练器修改 (`audio_cross_domain_trainer.py`)

#### 2.1 添加新的验证方法
- 新增 `_validate_stage2()` 方法
- 与 `_validate_stage1()` 逻辑相同，不使用滑动窗口
- 直接对固定长度的数据进行前向传播
- 支持目标域数据（包含 waveform）

#### 2.2 修改 Stage 2 训练流程
- 移除滑动窗口预处理逻辑（`_preprocess_target_data_with_sliding_window`）
- 验证时调用新的 `_validate_stage2()` 方法
- 保留原有的教师-学生蒸馏机制
- 保留课程学习功能

#### 2.3 修改数据加载器配置
- 目标域测试加载器的 batch_size 从 1 改为正常的 `args.batch_size`
- 提高测试效率

#### 2.4 更新日志输出
- 移除关于滑动窗口筛选的日志信息

### 3. 主程序修改 (`main.py`)

#### 3.1 更新默认参数
- `--target_domain` 默认值从 `hot_zone` 改为 `hot_zone_fine`

#### 3.2 移除不需要的参数
- 移除 `--window_size`（滑动窗口大小）
- 移除 `--hop_size`（滑动窗口步长）
- 移除 `--test_batch_size`（不再需要固定为 1）
- 移除 `--use_sliding_window_filter` 及相关参数：
  - `--filter_confidence_threshold`
  - `--filter_window_size`
  - `--filter_hop_size`
  - `--filter_use_silence_removal`
  - `--filter_silence_top_db`

#### 3.3 更新输出信息
- 移除关于滑动窗口筛选的打印信息

## 保留的功能

以下功能仍然保留，未受影响：

1. **两阶段训练框架**
   - Stage 1: 特征表示学习
   - Stage 2: 分类器重训练 + 动态蒸馏

2. **动态蒸馏机制**
   - 教师-学生模型
   - EMA 更新
   - 弱增强/强增强数据

3. **课程学习**
   - 基于置信度的样本筛选
   - 动态阈值调整

4. **长尾学习技术**
   - Focal Loss
   - Logit Adjustment
   - Weighted Random Sampler

5. **可选功能**
   - Stage 2 使用源域数据
   - 早停机制

## 向后兼容性

**注意：** 此修改主要针对短时数据（约 1 秒）。如果需要处理长时数据（如原来的 `hot_zone`，30 多秒），需要：
1. 保留 `_validate_stage2_sliding_window()` 方法
2. 在 `dataloader.py` 中恢复对长时数据的特殊处理
3. 根据数据集特点选择合适的验证方法

## 使用方法

### 基本训练命令（使用新数据集）
```bash
python main.py --target_domain hot_zone_fine
```

### 跳过 Stage 1，直接训练 Stage 2
```bash
python main.py --skip_stage1 --stage1_model_path path/to/stage1_model.pth --target_domain hot_zone_fine
```

### 使用其他数据集
如果数据集名称不是 `hot_zone_fine`，可以通过参数指定：
```bash
python main.py --target_domain your_dataset_name
```

## 测试建议

1. **数据集验证**：确保 `data/hot_zone_fine/` 目录存在且包含 `cough/` 和 `non-cough/` 子目录
2. **数据时长检查**：验证音频文件时长是否约为 1 秒
3. **训练测试**：运行完整训练流程，确认没有错误
4. **性能对比**：比较修改前后的模型性能

## 已移除的代码

以下方法保留在代码中但不再使用（如需完全移除，可以手动删除）：

- `_preprocess_target_data_with_sliding_window()`：Stage 2 滑动窗口预处理
- `_validate_stage2_sliding_window()`：Stage 2 滑动窗口验证

这些方法可以保留以供参考或未来需要处理长时数据时使用。

