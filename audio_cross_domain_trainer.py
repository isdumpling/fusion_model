# audio_cross_domain_trainer.py

import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.utils.data as data
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import f1_score
import torchaudio
from torch_audiomentations import Compose, PitchShift, Gain

from datasets.dataloader import get_cross_domain_audio_dataset
from audio_distill_los_system import AudioDistillLOSSystem
from utils.common import hms_string, calculate_flops
from utils.logger import logger
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
        
        # 初始化模型 - 传入类别样本数以支持Logit Adjustment
        self.model = AudioDistillLOSSystem(args, class_counts=self.N_SAMPLES_PER_CLASS).cuda()
        
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
                drop_last=True
            )
        
        # 目标域数据加载器 (无标签数据用于蒸馏)
        self.target_unlabeled_loader = data.DataLoader(
            self.target_trainset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.workers,
            pin_memory=True,
            drop_last=True
        )
        
        # 测试数据加载器
        self.source_testloader = data.DataLoader(
            self.source_testset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.workers,
            pin_memory=True
        )
        
        self.target_testloader = data.DataLoader(
            self.target_testset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.workers,
            pin_memory=True
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
            sampler=sampler
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
        """
        print("=" * 50)
        print("Starting Stage 2: Classifier Retraining with Cross-Domain Distillation")
        if self.args.use_source_in_stage2:
            print(f"[INFO] Stage 2 将同时使用源域数据（监督学习）和目标域数据（自我学习）")
            print(f"[INFO] 源域/目标域批次比率: {self.args.source_target_ratio}")
        else:
            print(f"[INFO] Stage 2 仅使用目标域数据（纯自我学习）")
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
            # ===== 蒸馏权重动态调整 =====
            if epoch + 1 <= warmup:
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
            
            # Stage 2: 使用目标域测试集进行验证，使用滑动窗口
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
        
        total_loss, total_ce_loss, total_distill_loss, total_acc, num_batches = 0, 0, 0, 0, 0
        total_source_loss = 0  # 源域监督损失
        
        # 如果启用了源域数据使用
        if self.args.use_source_in_stage2:
            # 创建源域和目标域数据的迭代器
            source_iter = iter(self.source_trainloader)
            target_iter = iter(target_loader)
            
            # 计算总批次数（基于目标域）
            num_target_batches = len(target_loader)
            num_source_batches_per_target = self.args.source_target_ratio
            
            progress_bar = tqdm(range(num_target_batches), desc="Stage 2 Training (Source+Target)")
            
            for _ in progress_bar:
                # ========== 处理目标域数据（蒸馏损失） ==========
                try:
                    target_batch = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_loader)
                    target_batch = next(target_iter)
                
                spectrogram_target = target_batch[0].cuda()
                waveform_target = target_batch[3].cuda()
                
                # 弱增强版本
                x_u_weak_spec = spectrogram_target
                
                # 强增强版本
                x_u_strong_waveform = self.strong_augment(samples=waveform_target, sample_rate=16000)
                mel_transform = torchaudio.transforms.MelSpectrogram(
                    sample_rate=16000, n_fft=400, win_length=400, hop_length=160, 
                    n_mels=self.args.num_mels if hasattr(self.args, 'num_mels') else 96,
                    f_min=125, f_max=7500
                ).cuda()
                mel_strong = mel_transform(x_u_strong_waveform)
                x_u_strong_spec = torch.log(mel_strong + 1e-9)
                
                target_unlabeled = ((x_u_weak_spec, x_u_strong_spec),)
                
                # 计算目标域损失
                optimizer.zero_grad()
                target_results = self.model.compute_stage2_loss(target_unlabeled)
                target_loss = target_results['loss']
                
                # ========== 处理源域数据（监督损失） ==========
                source_loss = 0
                if num_source_batches_per_target > 0:
                    for _ in range(int(num_source_batches_per_target)):
                        try:
                            source_batch = next(source_iter)
                        except StopIteration:
                            source_iter = iter(self.source_trainloader)
                            source_batch = next(source_iter)
                        
                        source_spec = source_batch[0].cuda()
                        source_label = source_batch[1].cuda()
                        
                        # 对源域数据使用监督学习
                        source_output = self.model(source_spec)
                        source_batch_loss = self.model.compute_validation_loss(source_output, source_label)
                        source_loss += source_batch_loss
                    
                    # 平均源域损失
                    source_loss = source_loss / max(1, int(num_source_batches_per_target))
                
                # ========== 组合损失 ==========
                # 组合目标域损失和源域损失
                combined_loss = target_loss + source_loss
                
                combined_loss.backward()
                optimizer.step()
                
                # 更新统计
                total_loss += combined_loss.item()
                total_ce_loss += target_results['loss_ce'].item()
                total_distill_loss += target_results['loss_distill'].item()
                total_source_loss += source_loss.item() if isinstance(source_loss, torch.Tensor) else source_loss
                total_acc += target_results['train_acc'].item()
                num_batches += 1
                
                progress_bar.set_postfix({
                    'total': f'{total_loss / num_batches:.4f}',
                    'target': f'{target_loss.item():.4f}',
                    'source': f'{source_loss.item() if isinstance(source_loss, torch.Tensor) else source_loss:.4f}',
                    'acc': f'{total_acc / num_batches:.4f}'
                })
            
            # 防止除以零
            if num_batches == 0:
                print("[WARNING] No batches were processed. Returning zero losses.")
                return {
                    'total_loss': 0.0, 
                    'ce_loss': 0.0, 
                    'distill_loss': 0.0, 
                    'source_loss': 0.0,
                    'train_acc': 0.0
                }
            
            return {
                'total_loss': total_loss / num_batches, 
                'ce_loss': total_ce_loss / num_batches, 
                'distill_loss': total_distill_loss / num_batches, 
                'source_loss': total_source_loss / num_batches,
                'train_acc': total_acc / num_batches
            }
        
        else:
            # ========== 原始逻辑：仅使用目标域数据 ==========
            progress_bar = tqdm(target_loader, desc="Stage 2 Training (Target Only)")
            
            for target_batch in progress_bar:
                # target_batch[0] 是频谱图, target_batch[3] 是波形
                spectrogram_target = target_batch[0].cuda()
                waveform_target = target_batch[3].cuda()
                
                # 弱增强版本直接使用 dataloader 的输出（智能裁剪后的频谱图）
                x_u_weak_spec = spectrogram_target

                # 在波形上应用强数据增强
                x_u_strong_waveform = self.strong_augment(samples=waveform_target, sample_rate=16000)

                # 将强增强后的波形转换为频谱图
                mel_transform = torchaudio.transforms.MelSpectrogram(
                    sample_rate=16000, n_fft=400, win_length=400, hop_length=160, 
                    n_mels=self.args.num_mels if hasattr(self.args, 'num_mels') else 96,
                    f_min=125, f_max=7500
                ).cuda()
                mel_strong = mel_transform(x_u_strong_waveform)
                x_u_strong_spec = torch.log(mel_strong + 1e-9)
                
                target_unlabeled = ((x_u_weak_spec, x_u_strong_spec),)
                
                optimizer.zero_grad()
                
                # 调用新的损失函数，只传入目标域数据
                results = self.model.compute_stage2_loss(target_unlabeled)
                loss = results['loss']
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_ce_loss += results['loss_ce'].item()
                total_distill_loss += results['loss_distill'].item()
                total_acc += results['train_acc'].item()
                num_batches += 1
                
                progress_bar.set_postfix({
                    'total_loss': f'{total_loss / num_batches:.4f}', 
                    'acc': f'{total_acc / num_batches:.4f}'
                })
            
            # 防止除以零
            if num_batches == 0:
                print("[WARNING] No batches were processed. Returning zero losses.")
                return {
                    'total_loss': 0.0, 
                    'ce_loss': 0.0, 
                    'distill_loss': 0.0,
                    'source_loss': 0.0,
                    'train_acc': 0.0
                }
            
            return {
                'total_loss': total_loss / num_batches, 
                'ce_loss': total_ce_loss / num_batches, 
                'distill_loss': total_distill_loss / num_batches,
                'source_loss': 0.0,  # 未使用源域数据
                'train_acc': total_acc / num_batches
            }
    
    def _precompute_pseudo_labels(self, dataset):
        """
        预计算所有目标域训练数据的伪标签和置信度
        使用当前的教师模型（即Stage 1训练的最佳模型）进行预测
        
        Args:
            dataset: 用于预计算的数据集（可以是原始数据集或滑窗过滤后的数据集）
        
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
            pin_memory=True
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
            drop_last=use_drop_last
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
                
                # 尝试加载模型，使用 strict=False 以忽略不匹配的键
                missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint, strict=False)
                
                if missing_keys:
                    print(f"Warning: Missing keys in checkpoint: {len(missing_keys)} keys")
                if unexpected_keys:
                    print(f"Warning: Unexpected keys in checkpoint: {len(unexpected_keys)} keys (ignored)")
                
                self.best_model = copy.deepcopy(self.model.state_dict())
                print("Stage 1 model loaded successfully!\n")
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