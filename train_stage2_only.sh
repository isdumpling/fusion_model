#!/bin/bash
# 专注于 Stage 2 调优的训练脚本
# 使用 --skip_stage1 参数跳过 Stage 1，直接加载已训练好的模型进行 Stage 2 训练

# ===========================
# 方案 1: 使用最新训练的 Stage 1 模型（默认）
# ===========================
# 自动从 output/cold_zone_to_hot_zone_*/ 目录中查找最新的 best_model_stage1_audio.pth
python main.py \
    --data_dir data/ \
    --source_domain cold_zone \
    --target_domain hot_zone \
    --skip_stage1 \
    --use_sliding_window_filter \
    --filter_confidence_threshold 0.65 \
    --no_filter_silence_removal \
    --use_source_in_stage2 \
    --source_target_ratio 1.0

# ===========================
# 方案 2: 手动指定 Stage 1 模型路径
# ===========================
# 使用 models/ 目录下的预训练模型
python main.py \
    --data_dir data/ \
    --source_domain cold_zone \
    --target_domain hot_zone \
    --skip_stage1 \
    --stage1_model_path models/best_model_stage1_audio.pth \
    --use_sliding_window_filter \
    --filter_confidence_threshold 0.65 \
    --no_filter_silence_removal \
    --filter_window_size 0.5 \
    --filter_hop_size 0.25 \
    --use_source_in_stage2 \
    --source_target_ratio 1.0

# ===========================
# 方案 3: 指定特定训练输出的模型
# ===========================
# 使用特定训练运行的模型进行 Stage 2 调优
python main.py \
    --data_dir data/ \
    --source_domain cold_zone \
    --target_domain hot_zone \
    --skip_stage1 \
    --stage1_model_path output/cold_zone_to_hot_zone_10-8_12-30/best_model_stage1_audio.pth \
    --use_sliding_window_filter \
    --filter_confidence_threshold 0.70 \
    --no_filter_silence_removal \
    --filter_window_size 0.5 \
    --filter_hop_size 0.25

# ===========================
# 方案 4: 不使用滑动窗口筛选
# ===========================
# 纯自我学习，无滑动窗口预筛选
python main.py \
    --data_dir data/ \
    --source_domain cold_zone \
    --target_domain hot_zone \
    --skip_stage1 \
    --stage1_model_path models/best_model_stage1_audio.pth \
    --use_source_in_stage2 \
    --source_target_ratio 1.0

