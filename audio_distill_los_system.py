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
    
    def compute_stage2_loss(self, batch_labeled, batch_unlabeled):
        """
        Stage 2: 融合各种技术的分类器微调阶段
        """
        x_labeled, y_labeled = batch_labeled
        (x_u_weak, x_u_strong), *_ = batch_unlabeled
        
        # 前向传播
        scores_labeled, scores_u_strong = self.set_forward(x_labeled, x_u_strong)
        
        # 使用配置的损失函数计算监督损失
        loss_ce = self.loss_function(scores_labeled, y_labeled)
        
        # 计算训练准确率
        train_acc = accuracy(scores_labeled.argmax(dim=-1), y_labeled, 
                           task='multiclass', num_classes=self.num_classes)
        
        # dynamic-cdfsl伪标签蒸馏损失 (处理跨域问题)
        if self.teacher is not None:
            loss_distill = self._compute_distillation_loss(x_u_weak, scores_u_strong)
        else:
            loss_distill = torch.tensor(0.0).to(scores_labeled.device)
        
        # 总损失
        distill_weight = getattr(self.hparams, 'distill_weight', 1.0)
        total_loss = loss_ce + distill_weight * loss_distill
        
        return {
            'loss': total_loss,
            'loss_ce': loss_ce,
            'loss_distill': loss_distill,
            'train_acc': train_acc,
            'stage': 'stage2'
        }
    
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
        Stage 2阶段：冻结特征提取器，只训练分类器
        这是LOS方法的核心策略
        """
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        for param in self.classifier.parameters():
            param.requires_grad = True
            
        print("Backbone frozen, only classifier will be trained in Stage 2.")
    
    def unfreeze_all(self):
        """解冻所有参数"""
        for param in self.parameters():
            param.requires_grad = True
        print("All parameters unfrozen.")
    
    def get_stage2_optimizer(self, lr=0.01, weight_decay=1e-4):
        """
        获取Stage 2优化器 - 只优化分类器参数
        """
        return torch.optim.SGD(
            self.classifier.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay
        )
    
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
        
        # Stage 2可能需要重新初始化损失函数（如果需要不同的配置）
        # 但通常我们保持相同的损失函数配置
        
        print("Switched to Stage 2: classifier retraining with distillation.")
        print(f"Loss function configuration remains: {type(self.loss_function.loss_fn).__name__}")
    
    def get_feature_extractor(self):
        """获取特征提取器"""
        if hasattr(self, 'teacher') and self.teacher is not None:
            return self.teacher  # 返回教师模型作为特征提取器
        else:
            return self.feature_extractor  # 返回学生模型的特征提取器