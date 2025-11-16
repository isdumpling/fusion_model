#!/bin/bash

# ============================================================================
# 消融实验脚本
# 用于运行三种消融实验配置：
# (a) 无采样和长尾损失
# (b) 仅无长尾损失  
# (c) 仅无采样
# ============================================================================

# 使用方法:
# ./run_ablation_experiments.sh <experiment_type> [其他参数]
# 
# experiment_type 可选值:
#   a 或 no_sampling_no_loss  : 无采样和长尾损失
#   b 或 no_loss_only        : 仅无长尾损失（保留采样器）
#   c 或 no_sampling_only    : 仅无采样（保留长尾损失）
#
# 示例:
#   ./run_ablation_experiments.sh a --epochs 50
#   ./run_ablation_experiments.sh b --lr 0.0001
#   ./run_ablation_experiments.sh c --use_focal_loss

set -e  # 遇到错误时退出

# 检查参数
if [ $# -lt 1 ]; then
    echo "错误: 需要指定实验类型"
    echo ""
    echo "使用方法: $0 <experiment_type> [其他参数]"
    echo ""
    echo "实验类型:"
    echo "  a | no_sampling_no_loss  : (a) 无采样和长尾损失"
    echo "                              - 使用均匀随机采样（不用WeightedRandomSampler）"
    echo "                              - 使用标准交叉熵损失（不用Focal Loss/Logit Adjustment）"
    echo ""
    echo "  b | no_loss_only         : (b) 仅无长尾损失"
    echo "                              - 保留WeightedRandomSampler"
    echo "                              - 使用普通交叉熵（不用Focal Loss/Logit Adjustment）"
    echo ""
    echo "  c | no_sampling_only     : (c) 仅无采样"
    echo "                              - 移除采样器（不用WeightedRandomSampler）"
    echo "                              - 保留长尾损失（使用Focal Loss和Logit Adjustment）"
    echo ""
    echo "示例:"
    echo "  $0 a --epochs 50"
    echo "  $0 b --lr 0.0001 --wd 1e-3"
    echo "  $0 c --use_focal_loss --focal_gamma 2.0"
    exit 1
fi

EXPERIMENT_TYPE=$1
shift  # 移除第一个参数，剩余参数传递给main.py

# 设置基础参数（所有实验通用）
BASE_ARGS=""

# 根据实验类型设置特定参数
case $EXPERIMENT_TYPE in
    a|no_sampling_no_loss)
        echo "========================================================================"
        echo "运行消融实验 (a): 无采样和长尾损失"
        echo "========================================================================"
        echo "配置说明:"
        echo "  - 采样策略: 均匀随机采样 (禁用 WeightedRandomSampler)"
        echo "  - 损失函数: 标准交叉熵损失 (禁用 Focal Loss 和 Logit Adjustment)"
        echo "  - 说明: 移除了过采样和特定于长尾的损失，作为baseline实验"
        echo "========================================================================"
        echo ""
        
        # 禁用采样器和长尾损失
        SPECIFIC_ARGS="--no_weighted_sampler"
        # 不添加 --use_focal_loss 和 --use_logit_adjustment (默认就是不启用)
        
        OUTPUT_SUFFIX="ablation_a_no_sampling_no_loss"
        ;;
        
    b|no_loss_only)
        echo "========================================================================"
        echo "运行消融实验 (b): 仅无长尾损失"
        echo "========================================================================"
        echo "配置说明:"
        echo "  - 采样策略: WeightedRandomSampler (启用)"
        echo "  - 损失函数: 标准交叉熵损失 (禁用 Focal Loss 和 Logit Adjustment)"
        echo "  - 说明: 保留采样器来处理不平衡，但使用普通交叉熵损失"
        echo "========================================================================"
        echo ""
        
        # 启用采样器，禁用长尾损失
        SPECIFIC_ARGS=""
        # --use_weighted_sampler 是默认启用的，无需显式指定
        # 不添加 --use_focal_loss 和 --use_logit_adjustment (默认就是不启用)
        
        OUTPUT_SUFFIX="ablation_b_no_loss_only"
        ;;
        
    c|no_sampling_only)
        echo "========================================================================"
        echo "运行消融实验 (c): 仅无采样"
        echo "========================================================================"
        echo "配置说明:"
        echo "  - 采样策略: 均匀随机采样 (禁用 WeightedRandomSampler)"
        echo "  - 损失函数: Focal Loss + Logit Adjustment (启用)"
        echo "  - 说明: 移除采样器，仅通过损失函数来处理不平衡问题"
        echo "========================================================================"
        echo ""
        
        # 禁用采样器，启用长尾损失
        SPECIFIC_ARGS="--no_weighted_sampler --use_focal_loss --use_logit_adjustment"
        
        OUTPUT_SUFFIX="ablation_c_no_sampling_only"
        ;;
        
    *)
        echo "错误: 未知的实验类型 '$EXPERIMENT_TYPE'"
        echo "有效选项: a, b, c, no_sampling_no_loss, no_loss_only, no_sampling_only"
        exit 1
        ;;
esac

# 自动设置输出目录（包含时间戳和实验类型）
TIMESTAMP=$(date +"%m-%d_%H-%M")
OUTPUT_DIR="output/${OUTPUT_SUFFIX}_${TIMESTAMP}"

# 合并所有参数
ALL_ARGS="$BASE_ARGS $SPECIFIC_ARGS --out $OUTPUT_DIR $@"

# 显示完整命令
echo "执行命令:"
echo "python main.py $ALL_ARGS"
echo ""

# 运行训练
python main.py $ALL_ARGS

echo ""
echo "========================================================================"
echo "实验完成！"
echo "结果保存在: $OUTPUT_DIR"
echo "========================================================================"
