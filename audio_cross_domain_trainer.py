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

from datasets.dataloader import get_cross_domain_audio_dataset
from audio_distill_los_system import AudioDistillLOSSystem
from utils.common import hms_string
from utils.logger import logger
from tqdm import tqdm

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

        # 数据加载
        self.setup_data()
        
        # 初始化模型 - 传入类别样本数以支持Logit Adjustment
        self.model = AudioDistillLOSSystem(args, class_counts=self.N_SAMPLES_PER_CLASS).cuda()
        
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
            batch_size=1,  # Stage 2目标域必须使用batch_size=1
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
        
        self.logger(f'Stage 1 Training Time: {hms_string(end_time - start_time)}', level=1)
        self.logger(f'Stage 1 Best Macro F1: {self.best_macro_f1:.4f}', level=1)
        
    def train_stage2(self):
        """
        Stage 2: 分类器重训练 + 动态蒸馏
        """
        print("=" * 50)
        print("Starting Stage 2: Classifier Retraining with Cross-Domain Distillation")
        print("=" * 50)
        
        if self.best_model is not None:
            self.model.load_state_dict(self.best_model)
        
        self.model.switch_to_stage2()
        
        self.best_macro_f1 = 0
        
        optimizer = self.model.get_stage2_optimizer(lr=self.args.finetune_lr, weight_decay=self.args.finetune_wd)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, self.args.finetune_epoch, eta_min=0.0)
        
        start_time = time.time()
        
        for epoch in range(self.args.finetune_epoch):
            train_results = self._train_stage2_epoch(optimizer)
            
            # Stage 2: 使用目标域测试集进行验证，使用滑动窗口
            test_loss, test_acc, test_cls, micro_f1, macro_f1, class_metrics = self._validate_stage2_sliding_window(self.target_testloader)

            lr = scheduler.get_last_lr()[0]
            scheduler.step()
            
            self.model.update_teacher_ema()
            
            if macro_f1 > self.best_macro_f1:
                self.best_macro_f1 = macro_f1
                self.many_best = test_cls[0] if len(test_cls) > 0 else 0
                self.med_best = test_cls[1] if len(test_cls) > 1 else 0
                self.few_best = test_cls[2] if len(test_cls) > 2 else 0
                self.best_model = copy.deepcopy(self.model.state_dict())
            
            self._log_stage2_epoch(epoch + 1, train_results, test_loss, test_acc, test_cls, lr, micro_f1, macro_f1, class_metrics)

        end_time = time.time()
        
        self._save_model("best_model_stage2_audio.pth")
        
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

    def _train_stage2_epoch(self, optimizer):
        self.model.train()
        total_loss, total_ce_loss, total_distill_loss, total_acc, num_batches = 0, 0, 0, 0, 0
        target_iter = iter(self.target_unlabeled_loader)
        progress_bar = tqdm(self.source_trainloader, desc="Stage 2 Training")
        for source_batch in progress_bar:
            source_batch = [item.cuda() for item in source_batch[:2]]
            try:
                target_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(self.target_unlabeled_loader)
                target_batch = next(target_iter)
            x_target = target_batch[0].cuda()
            x_u_weak = x_target
            x_u_strong = x_target + 0.01 * torch.randn_like(x_target)
            target_unlabeled = ((x_u_weak, x_u_strong),)
            optimizer.zero_grad()
            results = self.model.compute_stage2_loss(source_batch, target_unlabeled)
            loss = results['loss']
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_ce_loss += results['loss_ce'].item()
            total_distill_loss += results['loss_distill'].item()
            total_acc += results['train_acc'].item()
            num_batches += 1
            progress_bar.set_postfix({'total_loss': f'{total_loss / num_batches:.4f}', 'acc': f'{total_acc / num_batches:.4f}'})
        return {'total_loss': total_loss / num_batches, 'ce_loss': total_ce_loss / num_batches, 
                'distill_loss': total_distill_loss / num_batches, 'train_acc': total_acc / num_batches}
    
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
            for spectrogram, target, _ in tqdm(testloader, desc="Stage 1 Validation"):
                spectrogram = spectrogram.cuda()
                target = target.cuda()
                
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
    
    def _validate_stage2_sliding_window(self, testloader):
        """
        Stage 2 验证：使用滑动窗口处理目标域数据
        """
        self.model.eval()
        
        # 将秒转换为频谱图的帧数
        # sample_rate=16000, hop_length=160 -> 1 frame = 160/16000 = 0.01s
        window_frames = int(self.args.window_size / 0.01)  # 0.8秒 = 80帧
        hop_frames = int(self.args.hop_size / 0.01)  # 0.5秒 = 50帧
        
        # 从dataloader获取训练时使用的固定长度
        # Stage 2的滑动窗口需要调整到与Stage 1训练时相同的大小
        from datasets.dataloader import TARGET_LENGTH
        model_input_frames = TARGET_LENGTH  # 64帧，与Stage 1训练时一致
        
        all_preds = []
        all_targets = []
        
        class_correct = [0] * self.args.num_class
        class_total = [0] * self.args.num_class
        
        # testloader的batch_size必须为1
        with torch.no_grad():
            for spectrogram, target, _ in tqdm(testloader, desc="Stage 2 Sliding Window Validation"):
                spectrogram = spectrogram.cuda()
                target = target.cuda()
                
                # 如果音频太短，无法使用滑动窗口，直接调整到模型输入大小
                if spectrogram.shape[3] <= window_frames:
                    # 调整到模型期望的输入大小（64帧）
                    if spectrogram.shape[3] < model_input_frames:
                        padding = model_input_frames - spectrogram.shape[3]
                        spectrogram = torch.nn.functional.pad(spectrogram, (0, padding))
                    elif spectrogram.shape[3] > model_input_frames:
                        spectrogram = spectrogram[:, :, :, :model_input_frames]
                    
                    outputs = self.model(spectrogram)
                    _, predicted = outputs.max(1)
                    final_prediction = predicted.item()
                else:
                    # 使用滑动窗口：默认预测为 non-cough (label 1)
                    final_prediction = 1 
                    
                    start = 0
                    while start + window_frames <= spectrogram.shape[3]:
                        # 提取滑动窗口（80帧）
                        window = spectrogram[:, :, :, start:start + window_frames]
                        
                        # 将窗口调整到模型期望的输入大小（64帧）
                        # 窗口是80帧，需要裁剪到64帧
                        if window.shape[3] > model_input_frames:
                            # 取中心部分64帧
                            excess = window.shape[3] - model_input_frames
                            start_trim = excess // 2
                            window = window[:, :, :, start_trim:start_trim + model_input_frames]
                        elif window.shape[3] < model_input_frames:
                            # 理论上不应该发生，因为window_frames=80 > model_input_frames=64
                            padding = model_input_frames - window.shape[3]
                            window = torch.nn.functional.pad(window, (0, padding))
                        
                        outputs = self.model(window)
                        _, predicted = outputs.max(1)
                        
                        # 假设 cough 的标签为 0
                        if predicted.item() == 0:
                            final_prediction = 0
                            break # 检测到cough，立即停止对此文件的处理
                            
                        start += hop_frames
                
                all_preds.append(final_prediction)
                all_targets.append(target.item())
                
                label = target.item()
                class_total[label] += 1
                if final_prediction == label:
                    class_correct[label] += 1

        # 计算各项指标
        correct = sum(class_correct)
        total = sum(class_total)
        test_acc = 100.0 * correct / total if total > 0 else 0
        
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
        
        # 由于是逐样本计算，loss的意义不大，这里返回0
        return 0.0, test_acc, test_cls, micro_f1, macro_f1, class_metrics
    
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
    
    def run_full_training(self):
        print("Starting Cross-Domain Audio Long-Tail Training")
        print(f"Source domain: {self.args.source_domain}")
        print(f"Target domain: {self.args.target_domain}")
        print(f"Number of classes: {self.args.num_class}")
        
        # 记录消融实验配置
        self.log_ablation_config()
        
        self.train_stage1()
        self.train_stage2()
        
        print("=" * 50)
        print("Training Complete!")
        print(f"Best Target Domain Macro F1: {self.best_macro_f1:.4f}")
        print("=" * 50)