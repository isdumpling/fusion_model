#!/bin/bash
# 修复Stage 2训练问题的实验脚本
# 问题：模型在Stage 2训练时，每个epoch的macro F1都维持在0.3374（很低且不变）
#      模型将所有样本都预测为NonCough类（Cough F1=0, NonCough Recall=1.0）
#      但最后通过温度缩放和阈值优化能得到F1=0.7714

echo "======================================"
echo "修复方案说明"
echo "======================================"
echo "问题诊断："
echo "1. Stage 2完全依赖伪标签，但Stage 1模型质量不足（Macro F1=0.6072）"
echo "2. 伪标签阈值过高（正类0.3，负类0.85），导致有效训练样本极少"
echo "3. 模型学会了'偷懒策略'——把所有样本都分为多数类NonCough"
echo ""
echo "解决方案："
echo "方案1: 大幅降低伪标签阈值"
echo "方案2: 提高学习率+降低KD权重"
echo "方案3: 使用BCE pos_weight增强正类训练"
echo "======================================"
echo ""

# ============================================
# 方案1: 降低伪标签阈值（最关键的修复）
# 0.7118
# ============================================
echo "[实验1] 降低伪标签阈值：正类 0.3→0.1, 负类 0.85→0.6"
python main.py \
    --skip_stage1 \
    --stage1_model_path models/best_model_stage1_audio.pth \
    --finetune_lr 3e-5 \
    --stage2_ce_conf_thresh_pos 0.1 \
    --stage2_ce_conf_thresh_neg 0.6 \
    --distill_weight_high 0.3 \
    --distill_weight_low 0.05

echo ""
echo "======================================"
echo ""

# ============================================
# 方案2: 进一步降低阈值 + 提高学习率
# 0.7118
# ============================================
echo "[实验2] 更激进的阈值+更高学习率"
python main.py \
    --skip_stage1 \
    --stage1_model_path models/best_model_stage1_audio.pth \
    --finetune_lr 5e-5 \
    --stage2_ce_conf_thresh_pos 0.05 \
    --stage2_ce_conf_thresh_neg 0.5 \
    --distill_weight_high 0.2 \
    --distill_weight_low 0.0

echo ""
echo "======================================"
echo ""

# ============================================
# 方案3: 阈值降低 + BCE pos_weight
# 0.3374
# ============================================
echo "[实验3] 降低阈值 + 使用BCE pos_weight强化正类"
python main.py \
    --skip_stage1 \
    --stage1_model_path models/best_model_stage1_audio.pth \
    --finetune_lr 4e-5 \
    --stage2_ce_conf_thresh_pos 0.1 \
    --stage2_ce_conf_thresh_neg 0.6 \
    --distill_weight_high 0.2 \
    --distill_weight_low 0.0 \
    --use_bce_pos_weight \
    --bce_pos_weight 3.0

echo ""
echo "======================================"
echo ""

# ============================================
# 方案4: 极端宽松阈值（诊断性实验）
# 0.7022
# ============================================
echo "[实验4-诊断] 极端宽松阈值：查看是否是阈值问题"
python main.py \
    --skip_stage1 \
    --stage1_model_path models/best_model_stage1_audio.pth \
    --finetune_lr 3e-5 \
    --stage2_ce_conf_thresh_pos 0.0 \
    --stage2_ce_conf_thresh_neg 0.0 \
    --distill_weight_high 0.1 \
    --distill_weight_low 0.0

echo ""
echo "======================================"
echo "实验完成！"
echo "======================================"

