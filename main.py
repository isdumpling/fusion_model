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
    parser.add_argument('--target_domain', default='hot_zone', help='Folder name of the target domain dataset')
    parser.add_argument('--out', default='output', help='Directory to output the results')

    # --- 模型和任务 ---
    parser.add_argument('--num_class', type=int, default=2, help='Number of classes (cough/non-cough)')
    parser.add_argument('--pretrained_model', 
                        default='models/vggish-10086976.pth',
                        type=str,
                        help='Path to the pretrained VGGish model')
    parser.add_argument('--cur_stage', default='stage1', type=str, help='Internal tracking of training stage')

    # --- Stage 1: 特征学习参数 (源域) ---
    parser.add_argument('--epochs', default=100, type=int, help='Number of total epochs for stage 1')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, help='Initial learning rate for stage 1')
    parser.add_argument('--wd', default=1e-3, type=float, help='Weight decay for stage 1 optimizer')
    parser.add_argument('--early_stopping_patience', default=20, type=int, 
                        help='Patience for early stopping in stage 1. Set to 0 to disable.')

    # --- Stage 2: 分类器微调和蒸馏参数 (源域+目标域) ---
    parser.add_argument('--finetune_epoch', default=50, type=int, help='Number of total epochs for stage 2')
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

    # --- 滑动窗口测试参数 ---
    parser.add_argument('--window_size', type=float, default=0.8, help='Sliding window size in seconds for testing')
    parser.add_argument('--hop_size', type=float, default=0.5, help='Sliding window hop size in seconds for testing')

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
    print(f"WeightedRandomSampler: {'ENABLED' if args.use_weighted_sampler else 'DISABLED'}")
    print(f"Focal Loss: {'ENABLED' if args.use_focal_loss else 'DISABLED'}")
    if args.use_focal_loss:
        print(f"  - Gamma: {args.focal_gamma}")
        print(f"  - Alpha: {args.focal_alpha}")
    print(f"Logit Adjustment: {'ENABLED' if args.use_logit_adjustment else 'DISABLED'}")
    if args.use_logit_adjustment:
        print(f"  - Tau: {args.logit_adj_tau}")
    print("="*60 + "\n")

    # 实例化训练器
    trainer = CrossDomainAudioTrainer(args)

    # 启动完整的两阶段训练
    trainer.run_full_training()


if __name__ == '__main__':
    main()