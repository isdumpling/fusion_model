# audio_cross_domain_trainer.py

import os
import time
import copy
import numpy as np
import random
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.utils.data as data
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import f1_score
import torchaudio
from torch_audiomentations import Compose, PitchShift, Gain
import torch.nn.functional as F

from datasets.dataloader import get_cross_domain_audio_dataset
from audio_distill_los_system import AudioDistillLOSSystem
from utils.common import hms_string, calculate_flops
from utils.logger import logger
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def worker_init_fn(worker_id):
    """
    为DataLoader的每个worker设置不同但可复现的随机种子
    这确保了多worker情况下的可复现性
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class CrossDomainAudioTrainer:
    """
    跨域音频长尾分类训练器
    整合LOS两阶段训练和dynamic-cdfsl动态蒸馏
    支持消融实验的各种组合
    """
    
    def __init__(self, args):
        self.args = args
        self.logger = logger(args)
        
        # 最佳性能记录
        self.best_macro_f1 = 0
        self.best_model = None
        self.many_best = 0
        self.med_best = 0
        self.few_best = 0
        
        # 早停相关变量
        self.early_stopping_patience = getattr(args, 'early_stopping_patience', 0)
        if self.early_stopping_patience <= 0:
            self.early_stopping_patience = float('inf')
        self.patience_counter = 0
        self.best_val_loss = float('inf')

        # F1 score tracking for plots
        self.stage1_f1_scores = []
        self.stage2_f1_scores = []

        # 数据加载
        self.setup_data()
        
        # --- 设备初始化 (修复AttributeError) ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger(f"Using device: {self.device}")

        # 初始化模型并移动到设备
        self.model = AudioDistillLOSSystem(args, class_counts=self.N_SAMPLES_PER_CLASS).to(self.device)
        
        # 计算并记录模型FLOPs和参数量（在独立的随机状态下）
        self._compute_and_log_flops()
        
        self.strong_augment = Compose([
            PitchShift(min_transpose_semitones=-4, max_transpose_semitones=4, sample_rate=16000, p=0.5),  # 随机变调
            Gain(min_gain_in_db=-12, max_gain_in_db=12, p=0.5),              # 随机调整增益（音量）
        ])
        
    def setup_data(self):
        """设置数据加载器"""
        print("==> Preparing cross-domain audio dataset")
        
        # 获取源域和目标域数据集
        self.source_trainset, self.source_testset, \
        self.target_trainset, self.target_testset = get_cross_domain_audio_dataset(
            self.args.data_dir, self.args
        )
        
        # 计算类别样本数 (用于处理长尾分布)
        self.N_SAMPLES_PER_CLASS = self.source_trainset.img_num_list
        
        # 根据配置决定是否使用加权采样器
        if self.args.use_weighted_sampler:
            print("[*] Using WeightedRandomSampler for class imbalance")
            self.setup_weighted_sampler()
        else:
            print("[*] WeightedRandomSampler DISABLED")
            # 使用普通的随机采样
            self.source_trainloader = data.DataLoader(
                self.source_trainset,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.workers,
                pin_memory=True,
                drop_last=True,
                worker_init_fn=worker_init_fn
            )
        
        # 目标域数据加载器 (无标签数据用于蒸馏)
        self.target_unlabeled_loader = data.DataLoader(
            self.target_trainset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=worker_init_fn
        )
        
        # 测试数据加载器
        self.source_testloader = data.DataLoader(
            self.source_testset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn
        )
        
        self.target_testloader = data.DataLoader(
            self.target_testset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn
        )
        
    def setup_weighted_sampler(self):
        """设置加权采样器处理长尾分布"""
        class_counts = np.array(self.N_SAMPLES_PER_CLASS)
        class_weights = 1.0 / class_counts
        
        sample_weights = np.array([class_weights[t] for t in self.source_trainset.targets])
        sample_weights = torch.from_numpy(sample_weights).double()
        
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        self.source_trainloader = data.DataLoader(
            self.source_trainset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
            drop_last=True,
            pin_memory=True,
            sampler=sampler,
            worker_init_fn=worker_init_fn
        )
    
    def _compute_and_log_flops(self):
        """计算并记录模型的FLOPs和参数量"""
        import random
        
        print("\n" + "="*60)
        print("计算模型复杂度...")
        print("="*60)
        
        # 保存当前所有的随机状态
        torch_rng_state = torch.get_rng_state()
        numpy_rng_state = np.random.get_state()
        python_rng_state = random.getstate()
        if torch.cuda.is_available():
            cuda_rng_state = torch.cuda.get_rng_state()
        
        try:
            # 创建一个全新的模型实例用于计算FLOPs
            from audio_distill_los_system import AudioDistillLOSSystem
            temp_model = AudioDistillLOSSystem(self.args, class_counts=self.N_SAMPLES_PER_CLASS).cuda()
            temp_model.eval()
            
            # 计算FLOPs（使用标准输入尺寸: batch=1, channels=1, n_mels=96, time_frames=64）
            flops, params, flops_formatted, params_formatted = calculate_flops(
                temp_model, 
                input_shape=(1, 1, 96, 64),
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            # 删除临时模型
            del temp_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # 记录到日志
            self.logger("="*60, level=1)
            self.logger("MODEL COMPLEXITY", level=1)
            self.logger("="*60, level=1)
            self.logger(f"FLOPs: {flops_formatted}", level=1)
            self.logger(f"Parameters: {params_formatted}", level=1)
            if flops is not None:
                self.logger(f"FLOPs (exact): {flops:.0f}", level=2)
                self.logger(f"Parameters (exact): {params:.0f}", level=2)
            self.logger("="*60, level=1)
            
            print(f"FLOPs: {flops_formatted}")
            print(f"Parameters: {params_formatted}")
            print("="*60 + "\n")
            
        finally:
            # 恢复所有随机状态，确保FLOPs计算不影响训练
            torch.set_rng_state(torch_rng_state)
            np.random.set_state(numpy_rng_state)
            random.setstate(python_rng_state)
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(cuda_rng_state)
        
    def log_ablation_config(self):
        """记录消融实验配置"""
        self.logger("="*60, level=1)
        self.logger("ABLATION EXPERIMENT CONFIGURATION", level=1)
        self.logger("="*60, level=1)
        self.logger(f"WeightedRandomSampler: {'ENABLED' if self.args.use_weighted_sampler else 'DISABLED'}", level=1)
        self.logger(f"Focal Loss: {'ENABLED' if self.args.use_focal_loss else 'DISABLED'}", level=1)
        if self.args.use_focal_loss:
            self.logger(f"  - Gamma: {self.args.focal_gamma}", level=2)
            self.logger(f"  - Alpha: {self.args.focal_alpha}", level=2)
        self.logger(f"Logit Adjustment: {'ENABLED' if self.args.use_logit_adjustment else 'DISABLED'}", level=1)
        if self.args.use_logit_adjustment:
            self.logger(f"  - Tau: {self.args.logit_adj_tau}", level=2)
        self.logger(f"Label Smoothing: {self.args.label_smooth}", level=1)
        self.logger(f"Stage 2 Use Source Data: {'ENABLED' if self.args.use_source_in_stage2 else 'DISABLED'}", level=1)
        if self.args.use_source_in_stage2:
            self.logger(f"  - Source/Target Ratio: {self.args.source_target_ratio}", level=2)
        self.logger("="*60, level=1)
    
    def train_stage1(self):
        """
        Stage 1: 特征表示学习
        在源域数据上训练整个网络
        """
        print("=" * 50)
        print("Starting Stage 1: Feature Representation Learning")
        print("=" * 50)
        
        self.model.unfreeze_all()
        
        optimizer = self.model.get_stage1_optimizer(lr=self.args.lr, weight_decay=self.args.wd)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, self.args.epochs, eta_min=0.0)
        
        start_time = time.time()
        
        for epoch in range(self.args.epochs):
            train_loss, train_acc = self._train_stage1_epoch(optimizer)
            
            # Stage 1: 使用源域测试集进行验证，不使用滑动窗口
            test_loss, test_acc, test_cls, micro_f1, macro_f1, class_metrics = self._validate_stage1(self.source_testloader)

            self.stage1_f1_scores.append(macro_f1)

            lr = scheduler.get_last_lr()[0]
            scheduler.step()
            
            if macro_f1 > self.best_macro_f1:
                self.best_macro_f1 = macro_f1
                self.many_best = test_cls[0] if len(test_cls) > 0 else 0
                self.med_best = test_cls[1] if len(test_cls) > 1 else 0
                self.few_best = test_cls[2] if len(test_cls) > 2 else 0
                self.best_model = copy.deepcopy(self.model.state_dict())
            
            self._log_epoch_results(epoch + 1, self.args.epochs, 
                                  train_loss, train_acc, test_loss, test_acc, 
                                  test_cls, lr, "Stage1", micro_f1, macro_f1, class_metrics)
            
            if test_loss < self.best_val_loss:
                self.best_val_loss = test_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.early_stopping_patience:
                self.logger(f'Early stopping triggered after {epoch + 1} epochs.', level=1)
                break
        
        end_time = time.time()
        
        self._save_model("best_model_stage1_audio.pth")
        
        self._plot_f1_scores(self.stage1_f1_scores, "stage1")

        self.logger(f'Stage 1 Training Time: {hms_string(end_time - start_time)}', level=1)
        self.logger(f'Stage 1 Best Macro F1: {self.best_macro_f1:.4f}', level=1)
    
    def train_stage2(self):
        """
        Stage 2: 分类器重训练 + 动态蒸馏
        支持基于置信度的课程学习策略
        
        师兄建议：
        - 目标域数据接近均衡(399 cough / 412 non-cough)，关闭WeightedRandomSampler
        - 在均衡数据上使用加权采样会破坏概率校准
        """
        print("=" * 50)
        print("Starting Stage 2: Classifier Retraining with Cross-Domain Distillation")
        if self.args.use_source_in_stage2:
            print(f"[INFO] Stage 2 将同时使用源域数据（监督学习）和目标域数据（自我学习）")
            print(f"[INFO] 源域/目标域批次比率: {self.args.source_target_ratio}")
        else:
            print(f"[INFO] Stage 2 仅使用目标域数据（纯自我学习）")
        
        # 师兄建议：Stage 2 关闭加权采样（目标域接近均衡）
        use_weighted_sampler_stage2 = getattr(self.args, 'use_weighted_sampler_stage2', False)
        if use_weighted_sampler_stage2:
            print(f"[INFO] Stage 2 使用WeightedRandomSampler")
        else:
            print(f"[INFO] Stage 2 关闭WeightedRandomSampler（师兄建议：目标域接近均衡，无需加权）")
            # 重新创建目标域数据加载器（不使用加权采样）
            self.target_unlabeled_loader = data.DataLoader(
                self.target_trainset,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.workers,
                pin_memory=True,
                drop_last=True,
                worker_init_fn=worker_init_fn
            )
        print("=" * 50)
        
        if self.best_model is not None:
            self.model.load_state_dict(self.best_model)
        
        # ===== 在 Stage 2 开始前，使用 Stage 1 最佳模型测试目标域性能 =====
        print("\n" + "=" * 60)
        print("Pre-Stage 2 Evaluation: Testing Stage 1 Best Model on Target Domain")
        print("=" * 60)
        self.logger("="*60, level=1)
        self.logger("PRE-STAGE 2 EVALUATION: Stage 1 Best Model on Target Domain", level=1)
        self.logger("="*60, level=1)
        
        test_loss, test_acc, test_cls, micro_f1, macro_f1, class_metrics = self._validate_stage1(self.target_testloader)
        
        self.logger(f'[Target Test - Stage 1 Model] Loss: {test_loss:.4f}  Acc: {test_acc:.4f}', level=1)
        self.logger(f'[Target Test - Stage 1 Model] Micro F1: {micro_f1:.4f}  Macro F1: {macro_f1:.4f}', level=1)
        self.logger(f'[Target Test - Stage 1 Model] Cough F1: {class_metrics["cough_f1"]:.4f}  Precision: {class_metrics["cough_precision"]:.4f}  Recall: {class_metrics["cough_recall"]:.4f}', level=1)
        self.logger(f'[Target Test - Stage 1 Model] NonCough F1: {class_metrics["non_cough_f1"]:.4f}  Precision: {class_metrics["non_cough_precision"]:.4f}  Recall: {class_metrics["non_cough_recall"]:.4f}', level=1)
        
        print(f"\n[Pre-Stage 2 Baseline on Target Domain]")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_acc:.4f}%")
        print(f"  Macro F1: {macro_f1:.4f}")
        print(f"  Cough - F1: {class_metrics['cough_f1']:.4f}, Precision: {class_metrics['cough_precision']:.4f}, Recall: {class_metrics['cough_recall']:.4f}")
        print(f"  NonCough - F1: {class_metrics['non_cough_f1']:.4f}, Precision: {class_metrics['non_cough_precision']:.4f}, Recall: {class_metrics['non_cough_recall']:.4f}")
        print("=" * 60 + "\n")
        
        self.logger("="*60, level=1)
        
        self.model.switch_to_stage2()
        
        self.best_macro_f1 = 0
        
        optimizer = self.model.get_stage2_optimizer(lr=self.args.finetune_lr, weight_decay=self.args.finetune_wd)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, self.args.finetune_epoch, eta_min=0.0)
        
        # ===== 课程学习：预计算所有目标域样本的伪标签和置信度 =====
        pseudo_label_info = None
        if self.args.use_curriculum_learning:
            print("\n" + "=" * 50)
            print("Curriculum Learning Enabled - Precomputing Pseudo Labels")
            print("=" * 50)
            self.logger("Starting pseudo-label precomputation for curriculum learning...", level=1)
            pseudo_label_info = self._precompute_pseudo_labels(self.target_trainset)
            self.logger(f"Precomputed {len(pseudo_label_info)} samples with confidence scores", level=1)
            print("=" * 50 + "\n")
        
        start_time = time.time()
        
        # ===== Teacher EMA Warm-up 参数 =====
        warmup = getattr(self.args, 'teacher_ema_warmup', 5)
        distill_weight_high = getattr(self.args, 'distill_weight_high', 1.0)
        distill_weight_low = getattr(self.args, 'distill_weight_low', 0.3)
        
        print("\n" + "=" * 50)
        print("Teacher EMA Warm-up Configuration:")
        print(f"  - Warm-up epochs: {warmup}")
        print(f"  - Distillation weight (high): {distill_weight_high}")
        print(f"  - Distillation weight (low): {distill_weight_low}")
        print("=" * 50 + "\n")
        
        for epoch in range(self.args.finetune_epoch):
            # ===== 蒸馏权重动态调整（支持余弦衰减） =====
            use_kd_cosine_decay = getattr(self.args, 'use_kd_cosine_decay', False)
            
            if use_kd_cosine_decay:
                # 余弦衰减：从 distill_weight_high 衰减到 0
                import math
                progress = epoch / max(self.args.finetune_epoch - 1, 1)
                current_distill_weight = distill_weight_high * 0.5 * (1 + math.cos(math.pi * progress))
                self.model.hparams.distill_weight = current_distill_weight
                print(f"Epoch {epoch + 1}/{self.args.finetune_epoch}: KD权重余弦衰减 ({current_distill_weight:.4f})")
            elif epoch + 1 <= warmup:
                self.model.hparams.distill_weight = distill_weight_high
                print(f"Epoch {epoch + 1}/{self.args.finetune_epoch}: Using HIGH distillation weight ({distill_weight_high})")
            else:
                self.model.hparams.distill_weight = distill_weight_low
                print(f"Epoch {epoch + 1}/{self.args.finetune_epoch}: Using LOW distillation weight ({distill_weight_low})")
            
            # ===== 课程学习：动态筛选数据 =====
            current_target_loader = self.target_unlabeled_loader  # 默认使用所有数据
            num_selected_samples = len(self.target_trainset)
            
            if self.args.use_curriculum_learning and pseudo_label_info is not None:
                # 计算当前epoch的置信度阈值（线性衰减）
                current_threshold = self._compute_current_threshold(
                    epoch, 
                    self.args.finetune_epoch,
                    self.args.initial_confidence_threshold,
                    self.args.final_confidence_threshold
                )
                
                # 根据阈值筛选样本
                selected_indices = [idx for idx, conf in pseudo_label_info if conf >= current_threshold]
                num_selected_samples = len(selected_indices)
                
                # 创建包含筛选样本的数据加载器（使用与训练相同的数据集）
                if num_selected_samples > 0:
                    current_target_loader = self._create_filtered_loader(
                        self.target_trainset,
                        selected_indices,
                        self.args.batch_size
                    )
                    self.logger(f"Epoch {epoch + 1}/{self.args.finetune_epoch}: Using {num_selected_samples}/{len(self.target_trainset)} target samples (threshold={current_threshold:.4f})", level=1)
                else:
                    # 如果没有样本满足阈值，使用全部数据
                    self.logger(f"Epoch {epoch + 1}/{self.args.finetune_epoch}: No samples meet threshold {current_threshold:.4f}, using all samples", level=1)
            
            # 使用动态数据加载器进行训练
            train_results = self._train_stage2_epoch(optimizer, current_target_loader, num_selected_samples)
            
            # Stage 2: 使用目标域测试集进行验证
            test_loss, test_acc, test_cls, micro_f1, macro_f1, class_metrics = self._validate_stage1(self.target_testloader)

            self.stage2_f1_scores.append(macro_f1)

            lr = scheduler.get_last_lr()[0]
            scheduler.step()
            
            # ===== Teacher EMA Warm-up 控制 =====
            if epoch + 1 > warmup:
                self.model.update_teacher_ema()
                if epoch + 1 == warmup + 1:
                    print(f"\n[Epoch {epoch + 1}] Teacher EMA warm-up完成，开始更新Teacher模型\n")
            else:
                if epoch == 0:
                    print(f"\n[Epoch {epoch + 1}] Teacher EMA warm-up期间，暂停Teacher模型更新\n")
            
            if macro_f1 > self.best_macro_f1:
                self.best_macro_f1 = macro_f1
                self.many_best = test_cls[0] if len(test_cls) > 0 else 0
                self.med_best = test_cls[1] if len(test_cls) > 1 else 0
                self.few_best = test_cls[2] if len(test_cls) > 2 else 0
                self.best_model = copy.deepcopy(self.model.state_dict())
            
            self._log_stage2_epoch(epoch + 1, train_results, test_loss, test_acc, test_cls, lr, micro_f1, macro_f1, class_metrics)

        end_time = time.time()
        
        self._save_model("best_model_stage2_audio.pth")
        
        self._plot_f1_scores(self.stage2_f1_scores, "stage2")
        
        self.logger(f'Stage 2 Training Time: {hms_string(end_time - start_time)}', level=1)
        self.logger(f'Stage 2 Best Target Macro F1: {self.best_macro_f1:.4f}', level=1)
        
        # ===== Stage 2 完成后：在目标域验证集上扫描最优阈值 =====
        if getattr(self.args, 'scan_threshold_after_stage2', True):
            print("\n" + "=" * 60)
            print("Stage 2 训练完成，开始扫描最优决策阈值...")
            print("=" * 60)
            
            # 加载最佳模型
            if self.best_model is not None:
                self.model.load_state_dict(self.best_model)
            
            # 在目标域验证集上扫描阈值
            target_recall = getattr(self.args, 'target_recall_threshold', 0.85)
            optimal_threshold_info = self.scan_optimal_threshold(self.target_testloader, target_recall=target_recall)
            
            # 保存阈值信息到文件
            import json
            threshold_file = os.path.join(self.args.out, 'optimal_threshold.json')
            with open(threshold_file, 'w') as f:
                json.dump(optimal_threshold_info, f, indent=2)
            
            print(f"最优阈值信息已保存到: {threshold_file}")
            self.logger(f"最优阈值信息已保存到: {threshold_file}", level=1)

    def _train_stage1_epoch(self, optimizer):
        self.model.train()
        total_loss, total_acc, num_batches = 0, 0, 0
        progress_bar = tqdm(self.source_trainloader, desc="Stage 1 Training")
        for batch in progress_bar:
            batch = [item.cuda() for item in batch[:2]]
            optimizer.zero_grad()
            results = self.model.compute_stage1_loss(batch)
            loss = results['loss']
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_acc += results['train_acc'].item()
            num_batches += 1
            progress_bar.set_postfix({'loss': f'{total_loss / num_batches:.4f}', 'acc': f'{total_acc / num_batches:.4f}'})
        return total_loss / num_batches, total_acc / num_batches

    def _train_stage2_epoch(self, optimizer, target_loader=None, num_target_samples=None):
        """
        Stage 2 单个 epoch 的训练
        - 默认：仅使用目标域数据（无标签自我学习）
        - 可选：同时使用源域数据（有标签监督学习）+ 目标域数据
        """
        self.model.train()
        
        if target_loader is None:
            target_loader = self.target_unlabeled_loader
        
        total_losses = []
        ce_losses = []
        distill_losses = []
        correct = 0
        total = 0
        
        pbar = tqdm(target_loader, desc=f'Stage 2 Training (Target Only)', leave=False)

        # 主要训练循环
        for batch_idx, target_unlabeled in enumerate(pbar):
            # 正确解包目标域数据：(mel_spectrogram, label, idx, waveform)
            x_target_weak, labels, indices, waveforms = target_unlabeled
            x_target_weak = x_target_weak.to(self.device)
            waveforms = waveforms.to(self.device)
            labels = labels.to(self.device)
            
            # 对原始波形应用强增强生成 x_target_strong
            x_target_strong = []
            for waveform in waveforms:
                # 应用强增强到波形
                waveform_augmented = self.strong_augment(waveform.unsqueeze(0), sample_rate=16000).squeeze(0)
                
                # 将增强后的波形转换为 Mel 频谱图
                mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
                    sample_rate=16000,
                    n_fft=400,
                    win_length=400,
                    hop_length=160,
                    n_mels=96,
                    f_min=125,
                    f_max=7500
                ).to(self.device)
                
                mel_spec = mel_spectrogram_transform(waveform_augmented)
                log_mel_spec = torch.log(mel_spec + 1e-9)
                x_target_strong.append(log_mel_spec)
            
            x_target_strong = torch.stack(x_target_strong).to(self.device)

            optimizer.zero_grad()

            # --- Teacher Forward (EMA model on weak augmentation) ---
            with torch.no_grad():
                self.model.teacher.eval()
                teacher_scores = self.model.teacher(x_target_weak)
                teacher_probs = F.softmax(teacher_scores, dim=1)
                max_conf, pseudo_labels = torch.max(teacher_probs, dim=1)

            # --- Student Forward (Online model on strong augmentation) ---
            # 主模型 self.model' (即学生) 在训练模式下运行
            self.model.train()
            student_scores = self.model(x_target_strong)

            # --- Compute Stage 2 Loss ---
            total_loss, loss_ce, distill_loss = self.model.compute_stage2_loss(
                student_scores, teacher_scores, pseudo_labels, max_conf, is_weak=False
            )

            loss = total_loss
            loss.backward()
            optimizer.step()

            # 注意：batch级EMA更新已移除(师兄建议)
            # EMA Teacher 只在epoch级按warm-up规则更新 (见train_stage2方法)

            # 计算准确率（使用伪标签）
            _, predicted = torch.max(student_scores, 1)
            total += pseudo_labels.size(0)
            correct += (predicted == pseudo_labels).sum().item()

            # 记录损失
            total_losses.append(total_loss.item())
            ce_losses.append(loss_ce.item())
            distill_losses.append(distill_loss.item())
            
            pbar.set_postfix({
                'total': np.mean(total_losses),
                'ce': np.mean(ce_losses),
                'distill': np.mean(distill_losses),
                'acc': correct / total if total > 0 else 0
            })
        
        # 注意：batch级EMA更新已移除(师兄建议)
        # 现在只在epoch级按warm-up规则更新EMA (见train_stage2方法)
        
        train_acc = correct / total if total > 0 else 0
            
        return {
            "total_loss": np.mean(total_losses),
            "ce_loss": np.mean(ce_losses),
            "distill_loss": np.mean(distill_losses),
            "train_acc": train_acc
        }

    def _precompute_pseudo_labels(self, dataset):
        """
        预计算所有目标域训练数据的伪标签和置信度
        使用当前的教师模型（即Stage 1训练的最佳模型）进行预测
        
        Args:
            dataset: 用于预计算的数据集
        
        Returns:
            List[Tuple[int, float]]: 列表包含 (样本索引, 置信度) 元组
        """
        self.model.eval()
        pseudo_label_info = []
        
        # 创建一个不打乱的数据加载器，用于预计算
        precompute_loader = data.DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,  # 不打乱，保持索引对应
            num_workers=self.args.workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn
        )
        
        with torch.no_grad():
            current_idx = 0
            for batch in tqdm(precompute_loader, desc="Precomputing Pseudo Labels"):
                spectrograms = batch[0].cuda()
                
                # 使用教师模型进行预测
                # 在Stage 2开始时，教师模型已经被创建，它是Stage 1最佳模型的副本
                if self.model.teacher is not None:
                    outputs = self.model.teacher(spectrograms)
                else:
                    # 如果教师模型还未创建，使用当前模型
                    outputs = self.model(spectrograms)
                
                # 计算softmax概率
                probs = torch.softmax(outputs, dim=1)
                
                # 获取最高概率（置信度）
                max_probs, _ = torch.max(probs, dim=1)
                
                # 保存每个样本的索引和置信度
                for i in range(max_probs.size(0)):
                    confidence = max_probs[i].item()
                    pseudo_label_info.append((current_idx, confidence))
                    current_idx += 1
        
        # 按置信度降序排序（可选，但有助于调试）
        pseudo_label_info.sort(key=lambda x: x[1], reverse=True)
        
        # 打印置信度统计信息
        confidences = [conf for _, conf in pseudo_label_info]
        self.logger(f"Confidence statistics - Min: {min(confidences):.4f}, Max: {max(confidences):.4f}, Mean: {np.mean(confidences):.4f}, Median: {np.median(confidences):.4f}", level=1)
        
        return pseudo_label_info
    
    def _compute_current_threshold(self, current_epoch, total_epochs, initial_threshold, final_threshold):
        """
        计算当前epoch的置信度阈值（线性衰减）
        
        Args:
            current_epoch: 当前epoch（从0开始）
            total_epochs: 总epoch数
            initial_threshold: 初始置信度阈值
            final_threshold: 最终置信度阈值
            
        Returns:
            float: 当前epoch的置信度阈值
        """
        # 线性插值
        progress = current_epoch / max(total_epochs - 1, 1)  # 避免除以0
        current_threshold = initial_threshold - progress * (initial_threshold - final_threshold)
        return current_threshold
    
    def _create_filtered_loader(self, dataset, selected_indices, batch_size):
        subset = data.Subset(dataset, selected_indices)
        use_drop_last = len(selected_indices) >= batch_size
        eff_bs = min(batch_size, len(selected_indices)) if len(selected_indices) > 0 else 1
        return data.DataLoader(
            subset,
            batch_size=eff_bs,
            shuffle=True,
            num_workers=self.args.workers,
            pin_memory=True,
            drop_last=use_drop_last,
            worker_init_fn=worker_init_fn
        )

    
    def _validate_stage1(self, testloader):
        """
        Stage 1 验证：不使用滑动窗口，直接处理1秒数据
        """
        self.model.eval()
        
        all_preds = []
        all_targets = []
        total_loss = 0
        num_batches = 0
        
        class_correct = [0] * self.args.num_class
        class_total = [0] * self.args.num_class
        
        with torch.no_grad():
            for batch in tqdm(testloader, desc="Stage 1 Validation"):
                # 兼容3个值和4个值的batch
                spectrogram = batch[0].cuda()
                target = batch[1].cuda()
                
                # Stage 1: 直接前向传播，不使用滑动窗口
                outputs = self.model(spectrogram)
                loss = self.model.compute_validation_loss(outputs, target)
                total_loss += loss.item()
                num_batches += 1
                
                _, predicted = outputs.max(1)
                
                # 批量处理预测结果
                for i in range(target.size(0)):
                    label = target[i].item()
                    pred = predicted[i].item()
                    
                    all_preds.append(pred)
                    all_targets.append(label)
                    
                    class_total[label] += 1
                    if pred == label:
                        class_correct[label] += 1

        # 计算各项指标
        correct = sum(class_correct)
        total = sum(class_total)
        test_acc = 100.0 * correct / total if total > 0 else 0
        test_loss = total_loss / num_batches if num_batches > 0 else 0
        
        test_cls = self._calculate_class_accuracies(class_correct, class_total)
        
        # 计算总体F1
        micro_f1 = f1_score(all_targets, all_preds, average='micro')
        macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        
        # 计算每个类别的F1分数
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, support = precision_recall_fscore_support(
            all_targets, all_preds, labels=[0, 1], zero_division=0
        )
        
        # 包装类别F1分数 (cough=0, non-cough=1)
        class_metrics = {
            'cough_f1': f1[0],
            'non_cough_f1': f1[1],
            'cough_precision': precision[0],
            'non_cough_precision': precision[1],
            'cough_recall': recall[0],
            'non_cough_recall': recall[1]
        }
        
        return test_loss, test_acc, test_cls, micro_f1, macro_f1, class_metrics
    
    def _calculate_class_accuracies(self, class_correct, class_total):
        if len(class_correct) >= 3:
            many_acc = class_correct[0] / max(1, class_total[0]) * 100
            med_acc = class_correct[1] / max(1, class_total[1]) * 100  
            few_acc = class_correct[2] / max(1, class_total[2]) * 100
            return [many_acc, med_acc, few_acc]
        else:
            # 对于二分类，只返回前两个
            accs = [(class_correct[i] / max(1, class_total[i])) * 100 for i in range(len(class_correct))]
            while len(accs) < 3:
                accs.append(0)
            return accs
    
    def _log_epoch_results(self, epoch, total_epochs, train_loss, train_acc, 
                          test_loss, test_acc, test_cls, lr, stage, micro_f1, macro_f1, class_metrics):
        """记录epoch结果"""
        self.logger(f'{stage} - Epoch: [{epoch} | {total_epochs}]', level=1)
        self.logger(f'[Train]\tLoss:\t{train_loss:.4f}\tAcc:\t{train_acc:.4f}', level=2)
        self.logger(f'[Test ]\tLoss:\t{test_loss:.4f}\tAcc:\t{test_acc:.4f}', level=2)
        self.logger(f'[Test ]\tMicro F1:\t{micro_f1:.4f}\tMacro F1:\t{macro_f1:.4f}', level=2)
        # 添加类别特定的F1分数
        self.logger(f'[Cough   ]\tF1:\t{class_metrics["cough_f1"]:.4f}\tPrecision:\t{class_metrics["cough_precision"]:.4f}\tRecall:\t{class_metrics["cough_recall"]:.4f}', level=2)
        self.logger(f'[NonCough]\tF1:\t{class_metrics["non_cough_f1"]:.4f}\tPrecision:\t{class_metrics["non_cough_precision"]:.4f}\tRecall:\t{class_metrics["non_cough_recall"]:.4f}', level=2)
        if len(test_cls) >= 3:
            self.logger(f'[Stats]\tMany:\t{test_cls[0]:.4f}\tMedium:\t{test_cls[1]:.4f}\tFew:\t{test_cls[2]:.4f}', level=2)
        self.logger(f'[Param]\tLR:\t{lr:.8f}', level=2)
        
    def _log_stage2_epoch(self, epoch, train_results, test_loss, test_acc, test_cls, lr, micro_f1, macro_f1, class_metrics):
        """记录Stage 2 epoch结果"""
        self.logger(f'Stage2 - Epoch: [{epoch} | {self.args.finetune_epoch}]', level=1)
        
        # 根据是否使用源域数据显示不同的损失信息
        if self.args.use_source_in_stage2:
            self.logger(f'[Train]\tTotal Loss:\t{train_results["total_loss"]:.4f}\tCE Loss:\t{train_results["ce_loss"]:.4f}\tDistill Loss:\t{train_results["distill_loss"]:.4f}\tSource Loss:\t{train_results["source_loss"]:.4f}', level=2)
        else:
            self.logger(f'[Train]\tTotal Loss:\t{train_results["total_loss"]:.4f}\tCE Loss:\t{train_results["ce_loss"]:.4f}\tDistill Loss:\t{train_results["distill_loss"]:.4f}', level=2)
        
        self.logger(f'[Train]\tAcc:\t{train_results["train_acc"]:.4f}', level=2)
        self.logger(f'[Test ]\tLoss:\t{test_loss:.4f}\tAcc:\t{test_acc:.4f}', level=2)
        self.logger(f'[Test ]\tMicro F1:\t{micro_f1:.4f}\tMacro F1:\t{macro_f1:.4f}', level=2)
        # 添加类别特定的F1分数
        self.logger(f'[Cough   ]\tF1:\t{class_metrics["cough_f1"]:.4f}\tPrecision:\t{class_metrics["cough_precision"]:.4f}\tRecall:\t{class_metrics["cough_recall"]:.4f}', level=2)
        self.logger(f'[NonCough]\tF1:\t{class_metrics["non_cough_f1"]:.4f}\tPrecision:\t{class_metrics["non_cough_precision"]:.4f}\tRecall:\t{class_metrics["non_cough_recall"]:.4f}', level=2)
        if len(test_cls) >= 3:
            self.logger(f'[Stats]\tMany:\t{test_cls[0]:.4f}\tMedium:\t{test_cls[1]:.4f}\tFew:\t{test_cls[2]:.4f}', level=2)
        self.logger(f'[Param]\tLR:\t{lr:.8f}', level=2)
    
    def _save_model(self, filename):
        if self.best_model is not None:
            file_path = os.path.join(self.args.out, filename)
            torch.save(self.best_model, file_path)
            self.logger(f'Model saved to {file_path}', level=1)
    
    def _plot_f1_scores(self, f1_scores, stage_name):
        """
        绘制并保存 F1 分数曲线图
        
        Args:
            f1_scores (list): 每个 epoch 的 Macro F1 分数列表
            stage_name (str): 阶段名称 ('stage1' or 'stage2')
        """
        if not f1_scores:
            self.logger(f"No F1 scores to plot for {stage_name}.", level=1)
            return
            
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(f1_scores) + 1)
        plt.plot(epochs, f1_scores, marker='o', linestyle='-', label=f'{stage_name.capitalize()} Macro F1')
        
        plt.title(f'{stage_name.capitalize()} Macro F1 Score Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Macro F1 Score')
        plt.grid(True)
        plt.legend()
        
        # 标注最佳F1分数
        best_f1 = max(f1_scores)
        best_epoch = f1_scores.index(best_f1) + 1
        plt.annotate(f'Best F1: {best_f1:.4f} at Epoch {best_epoch}',
                     xy=(best_epoch, best_f1),
                     xytext=(best_epoch, best_f1 - 0.05 if best_f1 > 0.1 else best_f1 + 0.05),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                     horizontalalignment='center',
                     verticalalignment='top' if best_f1 > 0.1 else 'bottom')
        
        plt.tight_layout()
        
        file_path = os.path.join(self.args.out, f'{stage_name}_macro_f1.png')
        try:
            plt.savefig(file_path)
            self.logger(f'F1 score plot for {stage_name} saved to {file_path}', level=1)
        except Exception as e:
            self.logger(f"Error saving plot to {file_path}: {e}", level=1)
        finally:
            plt.close()
    
    def apply_temperature_scaling(self, logits, labels):
        """
        温度缩放（Temperature Scaling）用于概率校准
        使用验证集优化温度参数T，使预测概率更准确
        
        师兄建议：优先级A-1，立竿见影的校准方法
        
        Args:
            logits: 模型原始logits (N, num_classes)
            labels: 真实标签 (N,)
        
        Returns:
            optimal_temperature: 最优温度参数
        """
        from torch.optim import LBFGS
        
        print("\n" + "=" * 60)
        print("Temperature Scaling - 概率校准 (师兄建议: 优先级A-1)")
        print("=" * 60)
        
        # 创建温度参数
        temperature = nn.Parameter(torch.ones(1).cuda() * 1.5)
        
        # NLL损失
        criterion = nn.CrossEntropyLoss()
        
        # 优化温度
        optimizer = LBFGS([temperature], lr=0.01, max_iter=50)
        
        def eval():
            optimizer.zero_grad()
            loss = criterion(logits / temperature, labels.long())
            loss.backward()
            return loss
        
        optimizer.step(eval)
        
        optimal_temp = temperature.item()
        print(f"最优温度 T = {optimal_temp:.4f}")
        print(f"校准前后对比：温度缩放可以将过度自信的概率拉回合理区间")
        print("=" * 60 + "\n")
        
        self.logger(f"Temperature Scaling: 最优温度 T = {optimal_temp:.4f}", level=1)
        
        return optimal_temp
    
    def apply_prior_shift(self, logits, source_prior=0.07, target_prior=0.49):
        """
        先验logit平移：补偿源域和目标域的类别先验不匹配
        
        师兄建议：优先级A-2
        源域先验 7% → 目标域 ~49%
        在推理时给 cough logit 加上 Δ = logit(π_target) - logit(π_source)
        
        Args:
            logits: 模型原始logits (N, 2)，[:, 0]是cough, [:, 1]是non-cough
            source_prior: 源域正类先验
            target_prior: 目标域正类先验
        
        Returns:
            adjusted_logits: 调整后的logits
        """
        import math
        
        # 计算logit偏移量
        delta = math.log(target_prior / (1 - target_prior)) - math.log(source_prior / (1 - source_prior))
        
        print(f"\n应用先验logit平移（师兄建议: 优先级A-2）")
        print(f"  源域先验: {source_prior:.4f}")
        print(f"  目标域先验: {target_prior:.4f}")
        print(f"  Logit偏移量 Δ: {delta:.4f}")
        print(f"  效果：系统性降低cough的判决阈值\n")
        
        # 对cough类的logit加上偏移量
        adjusted_logits = logits.clone()
        adjusted_logits[:, 0] += delta
        
        return adjusted_logits
    
    def plot_pr_curve(self, y_true, y_scores, save_path):
        """
        绘制PR曲线（Precision-Recall Curve）
        
        师兄建议：诊断工具，查看模型在不同阈值下的性能
        """
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, marker='.', label=f'PR Curve (AP={ap:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve (师兄建议: 诊断工具)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        print(f"PR曲线已保存到: {save_path}")
        self.logger(f"PR曲线已保存到: {save_path}", level=1)
        self.logger(f"Average Precision: {ap:.4f}", level=1)
    
    def plot_score_histogram(self, y_true, y_scores, save_path):
        """
        绘制分数直方图（正负样本分开）
        
        师兄建议：诊断工具
        如果两条直方图几乎完全重叠在低分区，就是"整体打分偏低"的铁证
        """
        pos_scores = y_scores[y_true == 1]
        neg_scores = y_scores[y_true == 0]
        
        plt.figure(figsize=(12, 6))
        plt.hist(neg_scores, bins=50, alpha=0.5, label=f'Non-Cough (n={len(neg_scores)})', color='blue')
        plt.hist(pos_scores, bins=50, alpha=0.5, label=f'Cough (n={len(pos_scores)})', color='red')
        plt.xlabel('Prediction Score (Probability)')
        plt.ylabel('Frequency')
        plt.title('Score Histogram: Pos vs Neg (师兄建议: 校准诊断)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 添加统计信息
        info_text = f"Cough: mean={pos_scores.mean():.4f}, std={pos_scores.std():.4f}\n"
        info_text += f"Non-Cough: mean={neg_scores.mean():.4f}, std={neg_scores.std():.4f}"
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        print(f"分数直方图已保存到: {save_path}")
        self.logger(f"分数直方图已保存到: {save_path}", level=1)
        self.logger(f"Cough scores - mean: {pos_scores.mean():.4f}, std: {pos_scores.std():.4f}", level=1)
        self.logger(f"Non-Cough scores - mean: {neg_scores.mean():.4f}, std: {neg_scores.std():.4f}", level=1)
        
        # 检查是否严重重叠
        if abs(pos_scores.mean() - neg_scores.mean()) < 0.2:
            print("⚠️  警告：正负样本分数严重重叠！这是'整体打分偏低/欠校准'的证据")
            self.logger("⚠️  警告：正负样本分数严重重叠，需要校准", level=1)

    def scan_optimal_threshold(self, testloader, target_recall=0.85):
        """
        在验证集上扫描最优决策阈值（整合师兄建议的所有校准技术）
        
        改进：
        1. 温度缩放（Temperature Scaling）- 优先级A-1
        2. 先验logit平移 - 优先级A-2
        3. PR曲线可视化
        4. 分数直方图诊断
        5. 阈值选择（目标Recall约束下的F2最优）
        
        Args:
            testloader: 验证集数据加载器
            target_recall: 目标召回率阈值（默认0.85）
        
        Returns:
            dict: 包含最优阈值和对应的指标
        """
        self.model.eval()
        
        logits_list = []
        y_true_list = []
        
        print("\n" + "=" * 60)
        print(f"扫描最优阈值（目标Recall≥{target_recall}）- 师兄建议的完整流程")
        print("=" * 60)
        
        # 收集所有验证集的logits和真实标签
        with torch.no_grad():
            for batch in tqdm(testloader, desc="收集验证集预测"):
                spectrogram = batch[0].cuda()
                target = batch[1].cuda()
                
                outputs = self.model(spectrogram)
                logits_list.append(outputs)
                y_true_list.append(target)
        
        # 合并所有batch的结果
        logits = torch.cat(logits_list)  # (N, 2)
        y_true = torch.cat(y_true_list).float()
        
        # ===== 步骤1: 温度缩放（师兄建议: 优先级A-1）=====
        temperature = 1.0
        if getattr(self.args, 'apply_temperature_scaling', True):
            temperature = self.apply_temperature_scaling(logits, y_true)
            logits_calibrated = logits / temperature
        else:
            logits_calibrated = logits
            print("跳过温度缩放")
        
        # ===== 步骤2: 先验logit平移（师兄建议: 优先级A-2）=====
        if getattr(self.args, 'apply_prior_shift', True):
            source_prior = getattr(self.args, 'source_prior', 0.07)
            target_prior = getattr(self.args, 'target_prior', 0.49)
            logits_final = self.apply_prior_shift(logits_calibrated, source_prior, target_prior)
        else:
            logits_final = logits_calibrated
            print("跳过先验logit平移")
        
        # 计算最终的概率（使用softmax）
        probs = F.softmax(logits_final, dim=1)[:, 0]  # cough类的概率
        
        # ===== 步骤3: 可视化诊断（师兄建议: 排查清单）=====
        # 转换为numpy用于绘图
        y_true_np = y_true.cpu().numpy()
        probs_np = probs.cpu().numpy()
        
        if getattr(self.args, 'plot_pr_curve', True):
            pr_curve_path = os.path.join(self.args.out, 'pr_curve_stage2.png')
            self.plot_pr_curve(y_true_np, probs_np, pr_curve_path)
        
        if getattr(self.args, 'plot_score_histogram', True):
            histogram_path = os.path.join(self.args.out, 'score_histogram_stage2.png')
            self.plot_score_histogram(y_true_np, probs_np, histogram_path)
        
        # ===== 步骤4: 扫描最优阈值（目标Recall约束下的F2最优）=====
        taus = torch.linspace(0, 1, 1001)
        best = None
        
        for t in taus:
            y_pred = (probs >= t).float()
            
            # 计算混淆矩阵元素
            tp = ((y_pred == 1) & (y_true == 1)).sum().item()
            fp = ((y_pred == 1) & (y_true == 0)).sum().item()
            fn = ((y_pred == 0) & (y_true == 1)).sum().item()
            tn = ((y_pred == 0) & (y_true == 0)).sum().item()
            
            # 计算指标
            recall = tp / (tp + fn + 1e-9)
            precision = tp / (tp + fp + 1e-9)
            f1 = 2 * precision * recall / (precision + recall + 1e-9)
            f2 = (5 * precision * recall) / (4 * precision + recall + 1e-9)  # Fβ, β=2
            
            # 只考虑满足召回率要求的阈值
            if recall >= target_recall:
                if best is None or f2 > best['f2']:
                    best = {
                        'tau': float(t),
                        'recall': recall,
                        'precision': precision,
                        'f1': f1,
                        'f2': f2,
                        'tp': tp,
                        'fp': fp,
                        'fn': fn,
                        'tn': tn,
                        'temperature': temperature
                    }
        
        if best is None:
            # 如果没有找到满足条件的阈值，返回召回率最高的阈值
            print(f"⚠️  警告：未找到满足Recall≥{target_recall}的阈值，返回最高召回率的阈值")
            print(f"   这说明模型对cough的打分整体偏低，严重欠校准！")
            max_recall = 0
            for t in taus:
                y_pred = (probs >= t).float()
                tp = ((y_pred == 1) & (y_true == 1)).sum().item()
                fn = ((y_pred == 0) & (y_true == 1)).sum().item()
                recall = tp / (tp + fn + 1e-9)
                if recall > max_recall:
                    max_recall = recall
                    fp = ((y_pred == 1) & (y_true == 0)).sum().item()
                    tn = ((y_pred == 0) & (y_true == 0)).sum().item()
                    precision = tp / (tp + fp + 1e-9)
                    f1 = 2 * precision * recall / (precision + recall + 1e-9)
                    f2 = (5 * precision * recall) / (4 * precision + recall + 1e-9)
                    best = {
                        'tau': float(t),
                        'recall': recall,
                        'precision': precision,
                        'f1': f1,
                        'f2': f2,
                        'tp': tp,
                        'fp': fp,
                        'fn': fn,
                        'tn': tn,
                        'temperature': temperature
                    }
        
        # 打印结果
        print("\n" + "=" * 60)
        print("最优阈值扫描结果（已应用师兄建议的校准）:")
        print("=" * 60)
        print(f"温度参数 T: {temperature:.4f}")
        print(f"最优阈值 τ: {best['tau']:.4f}")
        print(f"召回率 (Recall): {best['recall']:.4f}")
        print(f"精确率 (Precision): {best['precision']:.4f}")
        print(f"F1 分数: {best['f1']:.4f}")
        print(f"F2 分数: {best['f2']:.4f}")
        print(f"混淆矩阵: TP={best['tp']}, FP={best['fp']}, FN={best['fn']}, TN={best['tn']}")
        print("=" * 60 + "\n")
        
        # 记录到日志
        self.logger("=" * 60, level=1)
        self.logger("最优阈值扫描结果（已应用校准）", level=1)
        self.logger("=" * 60, level=1)
        self.logger(f"目标召回率: {target_recall}", level=1)
        self.logger(f"温度参数 T: {temperature:.4f}", level=1)
        self.logger(f"最优阈值 τ: {best['tau']:.4f}", level=1)
        self.logger(f"召回率 (Recall): {best['recall']:.4f}", level=1)
        self.logger(f"精确率 (Precision): {best['precision']:.4f}", level=1)
        self.logger(f"F1 分数: {best['f1']:.4f}", level=1)
        self.logger(f"F2 分数: {best['f2']:.4f}", level=1)
        self.logger(f"混淆矩阵: TP={best['tp']}, FP={best['fp']}, FN={best['fn']}, TN={best['tn']}", level=1)
        self.logger("=" * 60, level=1)
        
        return best

    def run_full_training(self):
        print("Starting Cross-Domain Audio Long-Tail Training")
        print(f"Source domain: {self.args.source_domain}")
        print(f"Target domain: {self.args.target_domain}")
        print(f"Number of classes: {self.args.num_class}")
        
        # 记录消融实验配置
        self.log_ablation_config()
        
        # 检查是否跳过 Stage 1
        if getattr(self.args, 'skip_stage1', False):
            print("\n" + "=" * 50)
            print("SKIPPING STAGE 1 TRAINING")
            print("=" * 50)
            
            # 确定要加载的 Stage 1 模型路径
            stage1_model_path = self._get_stage1_model_path()
            
            if stage1_model_path and os.path.exists(stage1_model_path):
                print(f"Loading Stage 1 model from: {stage1_model_path}")
                checkpoint = torch.load(stage1_model_path)
                
                # 【关键修复】：对于 VGGish 模型，embeddings 是延迟初始化的
                # 需要先进行一次前向传播来初始化 embeddings，然后再加载权重
                if self.model.feature_extractor.embeddings is None:
                    print("Initializing embeddings layer before loading checkpoint...")
                    # 创建一个 dummy 输入来触发 embeddings 的初始化
                    # VGGish 输入形状：(batch, 1, n_mels, time_frames)
                    dummy_input = torch.randn(1, 1, 96, 64).to(self.device)
                    with torch.no_grad():
                        _ = self.model(dummy_input)
                    print("Embeddings layer initialized.")
                
                # 尝试加载模型，使用 strict=False 以忽略不匹配的键
                missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint, strict=False)
                
                if missing_keys:
                    print(f"Warning: Missing keys in checkpoint: {len(missing_keys)} keys")
                    self.logger(f"Warning: Missing keys in checkpoint: {len(missing_keys)} keys", level=1)
                    for key in missing_keys[:5]:  # 只显示前5个
                        print(f"  - {key}")
                if unexpected_keys:
                    print(f"Warning: Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
                    self.logger(f"Warning: Unexpected keys in checkpoint: {len(unexpected_keys)} keys", level=1)
                    for key in unexpected_keys[:5]:
                        print(f"  - {key}")
                
                self.best_model = copy.deepcopy(self.model.state_dict())
                print("Stage 1 model loaded successfully!\n")
                self.logger("Stage 1 model loaded successfully", level=1)
            else:
                print(f"ERROR: Stage 1 model not found at {stage1_model_path}")
                print("Please specify a valid --stage1_model_path or train Stage 1 first.")
                return
        else:
            self.train_stage1()
        
        self.train_stage2()
        
        print("=" * 50)
        print("Training Complete!")
        print(f"Best Target Domain Macro F1: {self.best_macro_f1:.4f}")
        print("=" * 50)
    
    def _get_stage1_model_path(self):
        """
        获取 Stage 1 模型路径
        优先级：
        1. 用户指定的 --stage1_model_path
        2. 最新的 output/cold_zone_to_hot_zone_*/best_model_stage1_audio.pth
        """
        # 如果用户指定了路径，直接使用
        if hasattr(self.args, 'stage1_model_path') and self.args.stage1_model_path:
            return self.args.stage1_model_path
        
        # 否则查找最新的训练输出目录
        import glob
        pattern = f"output/{self.args.source_domain}_to_{self.args.target_domain}_*/best_model_stage1_audio.pth"
        matching_paths = glob.glob(pattern)
        
        if matching_paths:
            # 按修改时间排序，选择最新的
            latest_model = max(matching_paths, key=os.path.getmtime)
            return latest_model
        
        return None