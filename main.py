# main.py

import argparse
from datetime import datetime
import os
import torch
import numpy as np
import random

from audio_cross_domain_trainer import CrossDomainAudioTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Cross-Domain Audio Long-Tail Training')

    # --- 数据集和路径 ---
    parser.add_argument('--data_dir', default='data/', help='Root directory of datasets')
    parser.add_argument('--source_domain', default='cold_zone', help='Folder name of the source domain dataset')
    parser.add_argument('--target_domain', default='hot_zone_fine', help='Folder name of the target domain dataset')
    parser.add_argument('--out', default='output', help='Directory to output the results')

    # --- 模型和任务 ---
    parser.add_argument('--num_class', type=int, default=2, help='Number of classes (cough/non-cough)')
    parser.add_argument('--pretrained_model', 
                        default='models/vggish-10086976.pth',
                        type=str,
                        help='Path to the pretrained VGGish model')
    parser.add_argument('--cur_stage', default='stage1', type=str, help='Internal tracking of training stage')
    
    # --- Stage 1 模型控制 ---
    parser.add_argument('--skip_stage1', action='store_true', default=False,
                        help='Skip stage 1 training and load a pretrained stage 1 model')
    parser.add_argument('--stage1_model_path', type=str, default=None,
                        help='Path to stage 1 model. If not specified and skip_stage1 is True, '
                             'will use the latest model from output/cold_zone_to_hot_zone_*/best_model_stage1_audio.pth')

    # --- Stage 1: 特征学习参数 (源域) ---
    parser.add_argument('--epochs', default=100, type=int, help='Number of total epochs for stage 1')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, help='Initial learning rate for stage 1')
    parser.add_argument('--wd', default=1e-3, type=float, help='Weight decay for stage 1 optimizer')
    parser.add_argument('--early_stopping_patience', default=20, type=int, 
                        help='Patience for early stopping in stage 1. Set to 0 to disable.')

    # --- Stage 2: 分类器微调和蒸馏参数 (源域+目标域) ---
    parser.add_argument('--finetune_epoch', default=30, type=int, help='Number of total epochs for stage 2')
    parser.add_argument('--finetune_lr', default=0.001, type=float, help='Learning rate for stage 2')
    parser.add_argument('--finetune_wd', default=1e-4, type=float, help='Weight decay for stage 2 optimizer')

    # --- LOS 参数 ---
    parser.add_argument('--label_smooth', default=0.98, type=float, help='Label over-smoothing value for LOS')

    # --- dynamic-cdfsl 参数 ---
    parser.add_argument('--distill_weight', default=1.0, type=float, help='Weight for the distillation loss')
    parser.add_argument('--momentum_update', default=0.999, type=float, help='EMA momentum for teacher model update')
    parser.add_argument('--center_momentum', default=0.9, type=float, help='EMA momentum for the center vector')
    parser.add_argument('--apply_center', action='store_true', default=False,
                        help='Apply center vector to teacher logits')
    parser.add_argument('--stage2_ce_conf_thresh', type=float, default=0.6,
                        help='Min confidence to use hard pseudo labels for CE in Stage2')
    parser.add_argument('--distill_temperature', type=float, default=2.0,
                        help='Temperature for KL distillation in Stage2')
    
    # --- Teacher EMA Warm-up 参数 ---
    parser.add_argument('--teacher_ema_warmup', type=int, default=5,
                        help='Number of epochs to wait before starting teacher EMA updates (default: 5)')
    parser.add_argument('--distill_weight_high', type=float, default=0.7,
                        help='High distillation weight during warm-up period (default: 1.0)')
    parser.add_argument('--distill_weight_low', type=float, default=0.3,
                        help='Low distillation weight after warm-up period (default: 0.3)')

    # --- 滑动窗口测试参数 ---
    parser.add_argument('--window_size', type=float, default=0.64, help='Sliding window size in seconds for testing')
    parser.add_argument('--hop_size', type=float, default=0.32, help='Sliding window hop size in seconds for testing')

    # --- 训练设置 ---
    parser.add_argument('--batch_size', default=16, type=int, help='Train batchsize')
    parser.add_argument('--test_batch_size', default=1, type=int, help='Test batchsize, must be 1 for sliding window')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--seed', type=str, default='42', help='Manual seed')
    parser.add_argument('--gpu', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')

    # === 消融实验参数 ===
    
    # --- WeightedRandomSampler 控制 ---
    parser.add_argument('--use_weighted_sampler', action='store_true', default=True,
                        help='Use weighted random sampler to handle class imbalance')
    parser.add_argument('--no_weighted_sampler', dest='use_weighted_sampler', action='store_false',
                        help='Disable weighted random sampler')
    
    # --- Focal Loss 参数 ---
    parser.add_argument('--use_focal_loss', action='store_true', default=False,
                        help='Use focal loss instead of cross entropy')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal loss gamma parameter. Set to 0 to disable focal loss effect')
    parser.add_argument('--focal_alpha', type=float, default=None,
                        help='Focal loss alpha parameter for class weighting. None for no weighting')
    
    # --- Logit Adjustment 参数 ---
    parser.add_argument('--use_logit_adjustment', action='store_true', default=False,
                        help='Use logit adjustment for long-tail distribution')
    parser.add_argument('--logit_adj_tau', type=float, default=1.0,
                        help='Temperature parameter for logit adjustment. Set to 0 to disable')
    
    # --- 快捷配置参数 ---
    parser.add_argument('--ablation_config', type=str, default=None,
                        choices=['combo1', 'combo2', 'combo3', 'combo4', 'combo5'],
                        help='Predefined ablation experiment configurations: '
                             'combo1: Focal Loss + Logit Adj (no sampler), '
                             'combo2: Only Logit Adj (no focal, no sampler), '
                             'combo3: Sampler + Focal (no logit adj), '
                             'combo4: Only Sampler, '
                             'combo5: ALl Enabled')
    
    # --- 课程学习参数 (Curriculum Learning for Stage 2) ---
    parser.add_argument('--use_curriculum_learning', action='store_true', default=True,
                        help='Enable curriculum learning based on teacher model confidence')
    parser.add_argument('--initial_confidence_threshold', type=float, default=0.85,
                        help='Initial confidence threshold for curriculum learning')
    parser.add_argument('--final_confidence_threshold', type=float, default=0.85,
                        help='Final confidence threshold for curriculum learning')
    
    # --- Stage 2 源域数据使用参数 ---
    parser.add_argument('--use_source_in_stage2', action='store_true', default=True,
                        help='Use source domain data in stage 2 (in addition to target domain data)')
    parser.add_argument('--source_target_ratio', type=float, default=2.0,
                        help='Ratio of source to target batches in stage 2 when use_source_in_stage2 is enabled')
    
    # --- Stage 2 滑动窗口筛选参数 ---
    parser.add_argument('--use_sliding_window_filter', action='store_true', default=True,
                        help='Filter target domain training data using sliding window before stage 2')
    parser.add_argument('--filter_confidence_threshold', type=float, default=0.65,
                        help='Confidence threshold for filtering cough segments (default: 0.75, lenient for cross-domain)')
    parser.add_argument('--filter_window_size', type=float, default=0.8,
                        help='Window size in seconds for filtering (default: 0.8s, must be > 0.64s for sliding)')
    parser.add_argument('--filter_hop_size', type=float, default=0.4,
                        help='Hop size in seconds for filtering (default: 0.4s, 50% overlap)')
    parser.add_argument('--filter_use_silence_removal', action='store_true', default=True,
                        help='Use silence removal in filtering')
    parser.add_argument('--no_filter_silence_removal', dest='filter_use_silence_removal', action='store_false',
                        help='Disable silence removal in filtering')
    parser.add_argument('--filter_silence_top_db', type=float, default=30,
                        help='Silence threshold in dB for filtering (higher = more lenient, default: 30)')
    parser.add_argument('--filter_keep_negatives', action='store_true', default=True,
                        help='Keep high-confidence negative samples in filtering')
    parser.add_argument('--filter_neg_conf_threshold', type=float, default=0.70,
                        help='Confidence threshold for filtering negative samples (default: filter_confidence_threshold + 0.05)')

    args = parser.parse_args()
    
    # 应用快捷配置
    if args.ablation_config:
        if args.ablation_config == 'combo1':
            # 组合一：关闭 WeightedRandomSampler，只使用 Focal Loss + Logit Adjustment
            args.use_weighted_sampler = False
            args.use_focal_loss = True
            args.use_logit_adjustment = True
            print("Ablation Config 1: Focal Loss + Logit Adjustment (No WeightedSampler)")
            
        elif args.ablation_config == 'combo2':
            # 组合二：只使用 Logit Adjustment
            args.use_weighted_sampler = False
            args.use_focal_loss = True
            args.focal_gamma = 0  # 设置gamma=0禁用focal effect
            args.use_logit_adjustment = True
            print("Ablation Config 2: Only Logit Adjustment")
            
        elif args.ablation_config == 'combo3':
            # 组合三：使用 WeightedRandomSampler + Focal Loss
            args.use_weighted_sampler = True
            args.use_focal_loss = True
            args.use_logit_adjustment = True
            args.logit_adj_tau = 0  # 设置tau=0禁用logit adjustment
            print("Ablation Config 3: WeightedSampler + Focal Loss (No Logit Adjustment)")
            
        elif args.ablation_config == 'combo4':
            print("Ablation config 4: Only WeightedSampler")

        elif args.ablation_config == 'combo5':
            # 组合四：全部启用
            args.use_weighted_sampler = True
            args.use_focal_loss = True
            args.use_logit_adjustment = True
            print("Ablation Config 5: All Methods Enabled")

    # --- 自动设置输出路径 ---
    now = datetime.now()
    time_str = f'{now.month}-{now.day}_{now.hour}-{now.minute}'
    
    # 在输出路径中包含消融配置信息
    ablation_suffix = ""
    if args.ablation_config:
        ablation_suffix = f"_{args.ablation_config}"
    elif not args.use_weighted_sampler or args.use_focal_loss or args.use_logit_adjustment:
        components = []
        if args.use_weighted_sampler:
            components.append("WS")
        if args.use_focal_loss:
            components.append("FL")
        if args.use_logit_adjustment:
            components.append("LA")
        if components:
            ablation_suffix = "_" + "-".join(components)
    
    # 添加 Stage 2 源域数据使用标识
    if args.use_source_in_stage2:
        ablation_suffix += f"_S2Src{args.source_target_ratio}"
    
    args.out = os.path.join(args.out, f'{args.source_domain}_to_{args.target_domain}{ablation_suffix}_{time_str}')

    if args.gpu:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    return args


def set_reproducibility(seed):
    if seed != 'None':
        seed = int(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)
        random.seed(seed)


def main():
    args = parse_args()
    set_reproducibility(args.seed)
    
    # 打印消融实验配置
    print("\n" + "="*60)
    print("ABLATION EXPERIMENT CONFIGURATION")
    print("="*60)
    print(f"Skip Stage 1: {'YES' if args.skip_stage1 else 'NO'}")
    if args.skip_stage1:
        print(f"  - Stage 1 Model Path: {args.stage1_model_path if args.stage1_model_path else 'Auto (latest from output/)'}")
    print(f"WeightedRandomSampler: {'ENABLED' if args.use_weighted_sampler else 'DISABLED'}")
    print(f"Focal Loss: {'ENABLED' if args.use_focal_loss else 'DISABLED'}")
    if args.use_focal_loss:
        print(f"  - Gamma: {args.focal_gamma}")
        print(f"  - Alpha: {args.focal_alpha}")
    print(f"Logit Adjustment: {'ENABLED' if args.use_logit_adjustment else 'DISABLED'}")
    if args.use_logit_adjustment:
        print(f"  - Tau: {args.logit_adj_tau}")
    print(f"Stage 2 Use Source Data: {'ENABLED' if args.use_source_in_stage2 else 'DISABLED'}")
    if args.use_source_in_stage2:
        print(f"  - Source/Target Ratio: {args.source_target_ratio}")
    print(f"Stage 2 Sliding Window Filter: {'ENABLED' if args.use_sliding_window_filter else 'DISABLED'}")
    if args.use_sliding_window_filter:
        print(f"  - Confidence Threshold: {args.filter_confidence_threshold}")
    print("="*60 + "\n")

    # 实例化训练器
    trainer = CrossDomainAudioTrainer(args)

    # 启动完整的两阶段训练
    trainer.run_full_training()


if __name__ == '__main__':
    main()