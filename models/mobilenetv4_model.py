import torch
import torch.nn as nn
from safetensors.torch import load_file


class MobileNetV4(nn.Module):
    """
    MobileNetV4 model for audio feature extraction
    Adapted from image classification to audio spectrograms
    """
    def __init__(self, num_classes=2):
        super(MobileNetV4, self).__init__()
        
        # 使用 timm 库中的 MobileNetV4 作为 backbone
        # 如果没有 timm，我们需要自己实现或使用 torchvision
        self.use_timm = False
        self.feature_dim = None  # 将在第一次前向传播时确定
        
        try:
            import timm
            # 创建 MobileNetV4 模型（不带预训练权重，我们会自己加载）
            self.backbone = timm.create_model(
                'mobilenetv4_conv_small.e2400_r224_in1k',
                pretrained=False,
                num_classes=0,  # 移除分类头，只保留特征提取器
                in_chans=1      # 修改为单通道输入
            )
            
            self.use_timm = True
            print(f"Using timm MobileNetV4")
            
            # 通过测试前向传播来获取真实的特征维度
            self.backbone.eval()  # 设置为评估模式以避免 batch norm 错误
            with torch.no_grad():
                dummy_input = torch.randn(2, 1, 96, 64)  # 使用 batch_size=2
                dummy_output = self.backbone(dummy_input)
                if dummy_output.dim() == 4:
                    # 需要全局池化
                    dummy_output = nn.AdaptiveAvgPool2d((1, 1))(dummy_output)
                    dummy_output = dummy_output.flatten(1)
                self.feature_dim = dummy_output.shape[1]
            self.backbone.train()  # 恢复训练模式
            
            print(f"Detected feature dimension: {self.feature_dim}")
            
        except (ImportError, RuntimeError) as e:
            print(f"Warning: timm not available or model not found ({e}). Using custom MobileNetV4 implementation.")
            # 如果没有 timm，使用简化版本
            self.backbone = self._create_simple_mobilenetv4()
            self.feature_dim = 1280
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类头
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.feature_dim, num_classes)
        )
        
    def _create_simple_mobilenetv4(self):
        """
        创建简化版的 MobileNetV4-like 架构
        使用深度可分离卷积和倒残差块
        """
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True)
            )
        
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                # 深度卷积
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
                # 逐点卷积
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
            )
        
        model = nn.Sequential(
            conv_bn(1, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1280, 1),
        )
        return model
    
    def forward(self, x):
        # x shape: (batch, 1, n_mels, time_frames)
        
        # 特征提取
        if self.use_timm:
            # timm 模型已经返回展平的特征
            features = self.backbone(x)
        else:
            # 自定义实现需要手动池化
            features = self.backbone(x)
            if features.dim() == 4:
                features = self.avgpool(features)
                features = features.flatten(1)
        
        # 分类
        x = self.head(features)
        return x
    
    def forward_features(self, x):
        """Extract features without classification"""
        if self.use_timm:
            features = self.backbone(x)
        else:
            features = self.backbone(x)
            if features.dim() == 4:
                features = self.avgpool(features)
                features = features.flatten(1)
        return features


def create_mobilenetv4_model(num_classes=2, pretrained_path=None):
    """
    Create MobileNetV4 model with optional pretrained weights
    
    Args:
        num_classes: Number of output classes
        pretrained_path: Path to pretrained weights (.safetensors or .pth file)
    """
    model = MobileNetV4(num_classes=num_classes)
    
    if pretrained_path:
        print(f"Loading pretrained MobileNetV4 weights from {pretrained_path}")
        try:
            # 根据文件扩展名选择加载方法
            if pretrained_path.endswith('.safetensors'):
                # 使用 safetensors 加载
                state_dict = load_file(pretrained_path)
            else:
                # 使用 PyTorch 标准加载
                checkpoint = torch.load(pretrained_path, map_location='cpu')
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
            
            # 加载权重
            model_dict = model.state_dict()
            pretrained_dict = {}
            
            # 处理键名不匹配的情况
            for k, v in state_dict.items():
                # 移除可能的前缀
                new_k = k
                if k.startswith('module.'):
                    new_k = k[7:]
                if k.startswith('model.'):
                    new_k = k[6:]
                
                # 尝试匹配到 backbone
                target_key = None
                if new_k in model_dict:
                    target_key = new_k
                elif f'backbone.{new_k}' in model_dict:
                    target_key = f'backbone.{new_k}'
                
                if target_key:
                    # 检查形状是否匹配
                    if v.shape == model_dict[target_key].shape:
                        pretrained_dict[target_key] = v
                    elif 'conv_stem.weight' in target_key or 'conv1.weight' in target_key:
                        # 处理第一层卷积的通道不匹配（3通道 -> 1通道）
                        if len(v.shape) == 4 and v.shape[1] == 3 and model_dict[target_key].shape[1] == 1:
                            # 对 RGB 三通道取平均，转换为单通道
                            adapted_weight = v.mean(dim=1, keepdim=True)
                            pretrained_dict[target_key] = adapted_weight
                            print(f"Adapted first conv layer: {v.shape} -> {adapted_weight.shape}")
                        else:
                            print(f"Skipping {target_key} due to shape mismatch: {v.shape} vs {model_dict[target_key].shape}")
                    else:
                        print(f"Skipping {target_key} due to shape mismatch: {v.shape} vs {model_dict[target_key].shape}")
            
            if pretrained_dict:
                # 部分加载（忽略分类头和形状不匹配的层）
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict, strict=False)
                print(f"Successfully loaded {len(pretrained_dict)} pretrained parameters")
            else:
                print("Warning: No matching pretrained parameters found")
                print("Model will use random initialization")
            
        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
            print("Proceeding with random initialization")
    
    # 初始化分类头
    for m in model.head.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    print(f"MobileNetV4 model created with {num_classes} output classes")
    return model


if __name__ == '__main__':
    # 测试模型
    model = create_mobilenetv4_model(num_classes=2)
    
    # 测试前向传播
    dummy_input = torch.randn(2, 1, 96, 64)  # (batch, channel, freq, time)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # 测试特征提取
    features = model.forward_features(dummy_input)
    print(f"Features shape: {features.shape}")

