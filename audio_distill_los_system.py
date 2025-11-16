# audio_distill_los_system.py

import torch
from torchmetrics.functional import accuracy
from torch import nn
from torch.nn import functional as F
import copy
from typing import Any
import numpy as np
import torch.distributed as dist
from torch.utils.data.sampler import WeightedRandomSampler

# Import the new loss functions
from losses.long_tail_losses import create_loss_function

class AudioDistillLOSSystem(nn.Module):
    """
    融合LOS和dynamic-cdfsl的音频跨域长尾识别系统
    支持多种长尾学习技术的消融实验
    
    核心创新:
    1. 支持Focal Loss, Logit Adjustment, WeightedSampling的灵活组合
    2. 集成dynamic-cdfsl的动态蒸馏框架处理跨域问题
    3. 专门优化音频数据的特征表示学习
    """
    
    def __init__(self, hparams, datamodule=None, class_counts=None):
        super().__init__()
        self.hparams = hparams
        self.datamodule = datamodule
        self.num_classes = hparams.num_class
        self.class_counts = class_counts if class_counts is not None else [1] * self.num_classes
        
        # 初始化音频特征提取器 (VGGish模型)
        self.feature_extractor = self._create_audio_backbone()

        # 学生模型就是特征提取器本身，它已经包含了分类头
        self.student = self.feature_extractor

        # 分类器仅仅是学生模型中分类头的一个引用
        self.classifier = self.student.head
        
        # 教师模型初始化 (稍后创建)
        self.teacher = None
        
        # 训练阶段标志
        self.current_stage = 1
        self.stage = hparams.cur_stage
        
        # dynamic-cdfsl相关参数
        self.momentum_update = hparams.momentum_update
        self.center_momentum = hparams.center_momentum
        self.apply_center = hparams.apply_center
        
        # 创建损失函数 (支持Focal Loss, Logit Adjustment等)
        self.loss_function = create_loss_function(hparams, self.class_counts)
        
        # Label smoothing (如果同时使用其他技术，可能需要调整)
        self.label_smooth = hparams.label_smooth
        
        # 注册中心向量缓冲区 (用于教师模型输出的中心化)
        if self.apply_center:
            self.register_buffer("center", torch.zeros(1, self.num_classes))
            
    def _create_audio_backbone(self):
        """Create VGGish feature extractor"""
        from models.vggish_model import create_vggish_model
        model = create_vggish_model(
            num_classes=self.num_classes,
            pretrained_path=self.hparams.pretrained_model
        )
        return model
    
    def create_teacher(self):
        """创建教师模型 - dynamic-cdfsl的核心组件"""
        # 教师模型应该是学生模型的一个完整深拷贝
        self.teacher = copy.deepcopy(self.student)
        
        # 冻结教师模型的所有参数
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        print("Teacher model created and frozen.")
    
    def forward(self, x):
        """前向传播"""
        return self.student(x)
    
    def set_forward(self, x_support, x_unlabeled=None):
        """
        设置前向传播，同时处理有标签和无标签数据
        
        Args:
            x_support: 有标签支持集数据
            x_unlabeled: 无标签数据 (用于动态蒸馏)
        """
        scores_support = self.forward(x_support)
        
        if x_unlabeled is not None:
            scores_unlabeled = self.forward(x_unlabeled)
            return scores_support, scores_unlabeled
        
        return scores_support
    
    def compute_validation_loss(self, outputs, targets):
        """计算验证损失"""
        return self.loss_function(outputs, targets)
    
    def compute_stage1_loss(self, batch):
        """
        Stage 1: 特征表示学习阶段
        使用配置的损失函数训练整个网络
        """
        x, y = batch
        scores = self.forward(x)
        
        # 使用配置的损失函数 (可能是CE, Focal, Logit Adjustment等)
        loss = self.loss_function(scores, y)
        
        acc = accuracy(scores.argmax(dim=-1), y, task='multiclass', num_classes=self.num_classes)
        
        return {
            'loss': loss,
            'train_acc': acc,
            'stage': 'stage1'
        }
    
    def get_stage2_optimizer(self, lr=1e-4, weight_decay=1e-4):
        """
        Stage 2: AdamW optimizer with parameter groups for fine-tuning.
        - Classifier (head) gets the full learning rate.
        - Feature extractor's embeddings get a smaller learning rate.
        - Feature extractor's feature layers remain frozen.
        """
        groups = []
        # Embeddings with a lower learning rate
        if hasattr(self.feature_extractor, 'embeddings') and self.feature_extractor.embeddings is not None:
            groups.append({'params': self.feature_extractor.embeddings.parameters(), 'lr': lr * 0.33})
        
        # Classifier (head) with the main learning rate
        groups.append({'params': self.classifier.parameters(), 'lr': lr})
        
        return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)

    def get_stage1_optimizer(self, lr=0.1, weight_decay=5e-4):
        return torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    
    def switch_to_stage2(self):
        """切换到Stage 2模式"""
        self.current_stage = 2
        self.freeze_backbone_for_stage2()
        if self.teacher is None:
            self.create_teacher()
        
        # Stage 2：关闭Label Smoothing以提高召回率
        disable_label_smoothing_stage2 = getattr(self.hparams, 'disable_label_smoothing_stage2', True)
        if disable_label_smoothing_stage2:
            original_smooth = self.label_smooth
            self.label_smooth = 0.0
            
            # 重新创建损失函数（不使用label smoothing）
            temp_smooth = self.hparams.label_smooth
            self.hparams.label_smooth = 0.0
            self.loss_function = create_loss_function(self.hparams, self.class_counts)
            # 将损失函数移动到模型所在的设备（修复 Logit Adjustment 的设备不匹配问题）
            if hasattr(self.loss_function, 'to'):
                device = next(self.parameters()).device
                self.loss_function = self.loss_function.to(device)
            self.hparams.label_smooth = temp_smooth  # 恢复原始值，仅供记录
            
            print(f"Stage 2: Label Smoothing 已关闭 (从 {original_smooth} 改为 0.0)")
        
        print("Switched to Stage 2: classifier retraining with distillation.")
        print(f"Loss function configuration: {type(self.loss_function.loss_fn).__name__}")
    
    def get_feature_extractor(self):
        """获取特征提取器"""
        if hasattr(self, 'teacher') and self.teacher is not None:
            return self.teacher  # 返回教师模型作为特征提取器
        else:
            return self.feature_extractor  # 返回学生模型的特征提取器

    def compute_stage2_loss(self, student_scores, teacher_scores, pseudo_labels, max_conf, is_weak,
                           true_labels=None, labeled_mask=None):
        """
        Computes the loss for Stage 2 using category-aware thresholds for pseudo-labels
        and distillation on high-confidence negative samples with margin constraint.
        
        师兄建议的改进：
        1. 负类阈值提高到0.85（从0.7）
        2. KD仅在"高置信度+高margin"的负类上计算
        
        半监督学习扩展：
        3. 支持 true_labels 和 labeled_mask，允许部分样本使用 ground-truth labels
        4. labeled samples 的 CE 使用真实标签，unlabeled samples 的 CE 使用伪标签
        5. KD 仅应用于 unlabeled 高置信度负样本
        
        Args:
            student_scores: 学生模型输出 (N, num_classes)
            teacher_scores: 教师模型输出 (N, num_classes)
            pseudo_labels: 教师模型生成的伪标签 (N,)
            max_conf: 教师模型的最大置信度 (N,)
            is_weak: 是否是弱增强
            true_labels: 真实标签 (N,), 可选
            labeled_mask: 指示哪些样本使用真实标签的布尔掩码 (N,), 可选
        """
        # --- 1. 构建 effective_labels 和 effective_conf ---
        # 默认从伪标签和置信度开始
        effective_labels = pseudo_labels
        effective_conf = max_conf
        
        if (true_labels is not None) and (labeled_mask is not None):
            # 克隆以避免原地修改问题
            effective_labels = pseudo_labels.clone()
            effective_conf = max_conf.clone()
            
            # 对于 labeled_mask 选中的样本，使用 ground-truth labels
            effective_labels[labeled_mask] = true_labels[labeled_mask]
            
            # labeled 样本总是通过置信度阈值检查（设为1.0）
            effective_conf[labeled_mask] = 1.0
        
        # --- 2. Category-Aware Pseudo-Label Masking ---
        tau_pos = getattr(self.hparams, 'stage2_ce_conf_thresh_pos', 0.3)
        tau_neg = getattr(self.hparams, 'stage2_ce_conf_thresh_neg', 0.85)  # 从0.7提高到0.85 (师兄建议)

        # Assuming 0 is the positive class (cough), 1 is negative class (non-cough)
        pos_class_mask = (effective_labels == 0)
        neg_class_mask = ~pos_class_mask

        mask_pos = pos_class_mask & (effective_conf >= tau_pos)
        mask_neg = neg_class_mask & (effective_conf >= tau_neg)
        
        # Combined mask for Cross-Entropy loss
        ce_mask = mask_pos | mask_neg

        # --- 3. Supervised Cross-Entropy on Effective Labels ---
        if ce_mask.any():
            loss_ce = self.loss_function(student_scores[ce_mask], effective_labels[ce_mask])
        else:
            loss_ce = torch.zeros((), device=student_scores.device)

        # --- 4. Distillation Loss on Unlabeled High-Confidence + High-Margin NEGATIVE Samples ---
        distill_loss = torch.zeros((), device=student_scores.device)
        dw = getattr(self.hparams, 'distill_weight', 0.0)

        # 构建 KD base mask: 仅在 unlabeled 高置信度负样本上应用 KD
        kd_base_mask = mask_neg
        
        if labeled_mask is not None:
            # 只对 unlabeled 负样本应用 KD
            unlabeled_mask = ~labeled_mask
            kd_base_mask = mask_neg & unlabeled_mask
        
        # Only compute distillation if weight is positive and there are eligible samples
        if dw > 0 and kd_base_mask.any():
            T = getattr(self.hparams, 'distill_temperature', 4.0)
            kd_neg_margin = getattr(self.hparams, 'kd_neg_margin', 0.20)  # 新增：KD负类margin约束 (师兄建议)
            
            with torch.no_grad():
                # 计算teacher在符合条件的负类样本上的概率分布
                t_probs = F.softmax(teacher_scores[kd_base_mask] / T, dim=-1)
                # 计算margin: P(non-cough) - P(cough)
                # 假设类别1是non-cough, 类别0是cough
                margin = t_probs[:, 1] - t_probs[:, 0]
                # 只保留margin >= kd_neg_margin的样本
                kd_mask = margin >= kd_neg_margin
            
            # 如果有满足margin条件的样本，才计算KD loss
            if kd_mask.any():
                student_log_probs = F.log_softmax(student_scores[kd_base_mask][kd_mask] / T, dim=-1)
                
                with torch.no_grad():
                    teacher_soft = F.softmax(teacher_scores[kd_base_mask][kd_mask] / T, dim=-1)
                
                distill_loss = F.kl_div(student_log_probs, teacher_soft.detach(), reduction='batchmean') * (T * T)

        # --- 4. Total Loss ---
        total_loss = loss_ce + dw * distill_loss
        
        return total_loss, loss_ce, distill_loss

    def _compute_distillation_loss(self, x_u_weak, scores_u_strong):
        """
        dynamic-cdfsl动态蒸馏损失计算
        """
        if self.teacher is None:
            return torch.tensor(0.0).to(scores_u_strong.device)
        
        # 使用教师模型生成伪标签
        with torch.no_grad():
            teacher_scores = self.teacher(x_u_weak)
            
            # 应用中心化
            if self.apply_center:
                teacher_scores = teacher_scores - self.center
                self.update_center(teacher_scores.clone())
            
            teacher_scores = teacher_scores.detach()
        
        # KL散度损失
        student_log_probs = F.log_softmax(scores_u_strong, dim=-1)
        teacher_probs = F.softmax(teacher_scores, dim=-1)
        
        distill_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        
        return distill_loss
    
    def update_teacher_ema(self):
        """
        动态蒸馏：使用EMA更新教师模型
        """
        if self.teacher is None:
            return
            
        with torch.no_grad():
            m = self.momentum_update
            for param_student, param_teacher in zip(self.student.parameters(),
                                                  self.teacher.parameters()):
                param_teacher.data.mul_(m).add_((1 - m) * param_student.detach().data)
    
    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        更新教师输出的中心向量
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / len(teacher_output)
        
        # EMA更新
        self.center = self.center * self.center_momentum + \
                     batch_center * (1 - self.center_momentum)
    
    def freeze_backbone_for_stage2(self):
        """
        Stage 2阶段：冻结早期卷积层，解冻后端嵌入层和分类器
        改进：只冻结features，解冻embeddings和head，以适应跨域场景
        """
        # 冻住早期卷积
        for p in self.feature_extractor.features.parameters():
            p.requires_grad = False
        
        # 解冻后端非线性嵌入层 + 分类头
        if hasattr(self.feature_extractor, 'embeddings') and self.feature_extractor.embeddings is not None:
            for p in self.feature_extractor.embeddings.parameters():
                p.requires_grad = True
        
        for p in self.classifier.parameters():
            p.requires_grad = True
            
        print("Stage2: features frozen; embeddings + head unfrozen.")
    
    def unfreeze_all(self):
        """解冻所有参数"""
        for param in self.parameters():
            param.requires_grad = True
        print("All parameters unfrozen.")