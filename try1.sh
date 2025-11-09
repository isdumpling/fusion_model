# # 实验 1: 学习率 6e-5
# python main.py \
#     --skip_stage1 \
#     --stage1_model_path models/best_model_stage1_audio.pth \
#     --finetune_lr 6e-5 \
#     --distill_weight_high 0.0 \
#     --distill_weight_low 0.0

# # 实验 2: 学习率 8e-5
# python main.py \
#     --skip_stage1 \
#     --stage1_model_path models/best_model_stage1_audio.pth \
#     --finetune_lr 8e-5 \
#     --distill_weight_high 0.0 \
#     --distill_weight_low 0.0

# python main.py \
#     --skip_stage1 \
#     --stage1_model_path models/best_model_stage1_audio.pth \
#     --finetune_lr 1e-4 \
#     --distill_weight_high 0.0 \
#     --distill_weight_low 0.0

# 实验 A: 更低的微调学习率
python main.py \
    --skip_stage1 \
    --stage1_model_path models/best_model_stage1_audio.pth \
    --finetune_lr 3e-5 \
    --distill_weight_high 0.0 \
    --distill_weight_low 0.0

实验 B: 使用 Logit Adjustment
python main.py \
    --skip_stage1 \
    --stage1_model_path models/best_model_stage1_audio.pth \
    --finetune_lr 1e-4 \
    --distill_weight_high 0.0 \
    --distill_weight_low 0.0 \
    --use_logit_adjustment