# 组合一：Focal Loss + Logit Adjustment (无采样器)
python main.py --ablation_config combo1

# 组合二：仅 Logit Adjustment
python main.py --ablation_config combo2  

# 组合三：采样器 + Focal Loss
python main.py --ablation_config combo3

# 组合四：仅S采样器
python main.py --ablation_config combo4

# 组合五：全部启用
python main.py --ablation_config combo5