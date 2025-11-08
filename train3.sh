python main.py \
  --source_domain cold_zone \
  --target_domain hot_zone_fine \
  --use_curriculum_learning \
  --initial_confidence_threshold 0.90 --final_confidence_threshold 0.70 \
  --teacher_ema_warmup 1 \
  --distill_weight_high 0.4 --distill_weight_low 0.0 \
  --finetune_lr 5e-5
