python main.py \
  --source_domain cold_zone \
  --target_domain hot_zone_fine \
  --teacher_ema_warmup 5 \
  --distill_weight_high 0.7 \
  --distill_weight_low 0.3 \
  --finetune_lr 0.00001