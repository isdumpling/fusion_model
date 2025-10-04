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

# 前五个组合进行stage 1的对比

# 之后，对于stage 2是否使用源域，以及源域与目标域的占比进行对比

# 不使用源域
python main.py

# 源域/目标域 1.0
python main.py --use_source_in_stage2 --source_target_ratio 1.0 --gpu 0

# 源域/目标域 0.75
python main.py --use_source_in_stage2 --source_target_ratio 0.75 --gpu 0

# 源域/目标域 0.50
python main.py --use_source_in_stage2 --source_target_ratio 0.50 --gpu 0

# 源域/目标域 0.25
python main.py --use_source_in_stage2 --source_target_ratio 0.25 --gpu 0