python main.py \
  --source_domain cold_zone \
  --target_domain hot_zone_fine \
  --use_source_in_stage2 --source_target_ratio 1.0 \
  --use_curriculum_learning \
  --initial_confidence_threshold 0.90 --final_confidence_threshold 0.75 \
  --teacher_ema_warmup 3 \
  --distill_weight_high 0.6 --distill_weight_low 0.1 \
  --finetune_lr 1e-4
