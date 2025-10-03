# losses/long_tail_losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Reference: "Focal Loss for Dense Object Detection"
    
    Loss = -alpha * (1-p_t)^gamma * log(p_t)
    where p_t is the model's estimated probability for the correct class
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: predictions from model (before softmax)
            targets: ground truth labels
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Get softmax probabilities
        p = F.softmax(inputs, dim=-1)
        
        # Get class probabilities
        class_mask = F.one_hot(targets, inputs.size(-1))
        probs = (p * class_mask).sum(dim=-1)
        
        # Calculate focal term: (1 - p_t)^gamma
        focal_weight = (1 - probs) ** self.gamma
        
        # Apply alpha if specified (class-level weighting)
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                # alpha is a tensor of class weights
                alpha_t = self.alpha[targets]
            focal_weight = alpha_t * focal_weight
        
        # Apply focal weight to cross entropy loss
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LogitAdjustment(nn.Module):
    """
    Logit Adjustment for long-tail learning
    Reference: "Long-tail Learning via Logit Adjustment"
    
    Adjusts logits based on class frequencies before computing loss
    """
    def __init__(self, class_counts, tau=1.0, base_loss='ce'):
        """
        Args:
            class_counts: list or array of sample counts per class
            tau: temperature parameter (tau=0 disables adjustment)
            base_loss: 'ce' for cross entropy, 'focal' for focal loss
        """
        super(LogitAdjustment, self).__init__()
        self.tau = tau
        self.base_loss_type = base_loss
        
        # Calculate class prior probabilities
        class_counts = np.array(class_counts)
        class_priors = class_counts / class_counts.sum()
        
        # Calculate adjustment term: tau * log(pi_y)
        # Store as buffer (not a parameter, but moves with model)
        adjustment = tau * torch.log(torch.tensor(class_priors + 1e-10).float())
        self.register_buffer('adjustment', adjustment)
        
        # Initialize base loss
        if base_loss == 'focal':
            self.base_loss = FocalLoss(gamma=2.0)
        else:
            self.base_loss = nn.CrossEntropyLoss()
    
    def forward(self, logits, targets):
        """
        Args:
            logits: model predictions (before softmax)
            targets: ground truth labels
        """
        if self.tau == 0:
            # No adjustment when tau=0
            return self.base_loss(logits, targets)
        
        # Apply logit adjustment: logits + tau * log(pi_y)
        adjusted_logits = logits + self.adjustment.unsqueeze(0)
        
        return self.base_loss(adjusted_logits, targets)


class CombinedLongTailLoss(nn.Module):
    """
    Combined loss function for long-tail learning with multiple techniques
    """
    def __init__(self, args, class_counts):
        super(CombinedLongTailLoss, self).__init__()
        
        self.use_focal_loss = args.use_focal_loss
        self.use_logit_adjustment = args.use_logit_adjustment
        self.label_smooth = args.label_smooth
        
        # Build the appropriate loss function based on configuration
        if self.use_logit_adjustment:
            # Logit adjustment as the main framework
            base_loss_type = 'focal' if self.use_focal_loss else 'ce'
            tau = args.logit_adj_tau if hasattr(args, 'logit_adj_tau') else 1.0
            self.loss_fn = LogitAdjustment(class_counts, tau=tau, base_loss=base_loss_type)
            
            # Update focal loss parameters if using focal
            if self.use_focal_loss and hasattr(self.loss_fn.base_loss, 'gamma'):
                self.loss_fn.base_loss.gamma = args.focal_gamma
                if args.focal_alpha is not None:
                    # Convert class counts to weights for focal alpha
                    weights = 1.0 / (np.array(class_counts) + 1)
                    weights = weights / weights.sum() * len(weights)
                    self.loss_fn.base_loss.alpha = torch.tensor(weights).float()
        
        elif self.use_focal_loss:
            # Only focal loss
            self.loss_fn = FocalLoss(gamma=args.focal_gamma)
            if args.focal_alpha is not None:
                weights = 1.0 / (np.array(class_counts) + 1)
                weights = weights / weights.sum() * len(weights)
                self.loss_fn.alpha = torch.tensor(weights).float()
        
        else:
            # Standard cross entropy with optional label smoothing
            if self.label_smooth < 1.0:
                self.loss_fn = nn.CrossEntropyLoss(label_smoothing=1.0 - self.label_smooth)
            else:
                self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, logits, targets):
        return self.loss_fn(logits, targets)


# Utility function to create loss based on args
def create_loss_function(args, class_counts):
    """
    Factory function to create appropriate loss function based on arguments
    
    Args:
        args: argument namespace containing loss configuration
        class_counts: list of sample counts per class
    """
    return CombinedLongTailLoss(args, class_counts)