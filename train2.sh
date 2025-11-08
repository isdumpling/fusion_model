python main.py \
  --source_domain cold_zone \
  --target_domain hot_zone_fine \
  --use_source_in_stage2 --source_target_ratio 0.5 \
  --use_curriculum_learning \
  --initial_confidence_threshold 0.95 --final_confidence_threshold 0.75 \
  --filter_confidence_threshold 0.70 \
  --teacher_ema_warmup 1 \
  --distill_weight_high 0.5 --distill_weight_low 0.0 \
  --finetune_lr 1e-4
