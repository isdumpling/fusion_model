python main.py \
  --source_domain cold_zone --target_domain hot_zone \
  --stage2_ce_conf_thresh 0.40 \
  --teacher_ema_warmup 3 \
  --distill_weight_high 0.0 \
  --distill_weight_low 0.2 \
  --distill_temperature 3.0 \
  --finetune_lr 3e-5 \
  --use_focal_loss --focal_gamma 2 --focal_alpha 0.75

python main.py \
  --source_domain cold_zone --target_domain hot_zone \
  --use_source_in_stage2 --source_target_ratio 0.25 \
  --stage2_ce_conf_thresh 0.45 \
  --teacher_ema_warmup 3 \
  --distill_weight_high 0.0 \
  --distill_weight_low 0.2 \
  --distill_temperature 3.0 \
  --use_focal_loss --focal_gamma 2 --focal_alpha 0.75 \
  --finetune_lr 3e-5
