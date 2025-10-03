import torch
import torch.nn as nn

class VGGish(nn.Module):
    """
    VGGish model for audio feature extraction
    Based on the AudioSet VGGish architecture
    """
    def __init__(self, num_classes=2):
        super(VGGish, self).__init__()
        
        # Convolutional layers
        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv block 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv block 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv block 4
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # 动态计算特征维度
        self._feature_dim = None
        
        # Fully connected layers for embedding
        self.embeddings = None  # 将在第一次前向传播时初始化
        
        # Classification head
        self.head = nn.Linear(128, num_classes)
        
    def _initialize_embeddings(self, input_dim):
        """动态初始化嵌入层"""
        self.embeddings = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 128),
            nn.ReLU(inplace=True),
        )
        self._feature_dim = input_dim
        
    def forward(self, x):
        # x shape: (batch, 1, n_mels, time_frames)
        x = self.features(x)
        
        # 动态初始化嵌入层（仅在第一次调用时）
        if self.embeddings is None:
            feature_dim = x.size(1) * x.size(2) * x.size(3)
            self._initialize_embeddings(feature_dim)
            # 将新初始化的层移到正确的设备上
            self.embeddings = self.embeddings.to(x.device)
        
        x = x.view(x.size(0), -1)
        x = self.embeddings(x)
        x = self.head(x)
        return x
    
    def forward_features(self, x):
        """Extract 128-dim embeddings without classification"""
        x = self.features(x)
        
        if self.embeddings is None:
            feature_dim = x.size(1) * x.size(2) * x.size(3)
            self._initialize_embeddings(feature_dim)
            self.embeddings = self.embeddings.to(x.device)
            
        x = x.view(x.size(0), -1)
        x = self.embeddings(x)
        return x


def create_vggish_model(num_classes=2, pretrained_path=None):
    """
    Create VGGish model with optional pretrained weights
    """
    model = VGGish(num_classes=num_classes)
    
    if pretrained_path:
        print(f"Loading pretrained VGGish weights from {pretrained_path}")
        try:
            state_dict = torch.load(pretrained_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            # 只加载卷积层的权重
            model_dict = model.state_dict()
            pretrained_dict = {}
            
            for k, v in state_dict.items():
                # 只加载 features 层的权重
                if k.startswith('features.'):
                    if k in model_dict:
                        pretrained_dict[k] = v
            
            if pretrained_dict:
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict, strict=False)
                print(f"Successfully loaded {len(pretrained_dict)} pretrained parameters from convolutional layers")
            else:
                print("No matching pretrained parameters found, using random initialization")
            
        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
            print("Proceeding with random initialization")
    
    # Initialize the classification head
    nn.init.xavier_uniform_(model.head.weight)
    if model.head.bias is not None:
        nn.init.constant_(model.head.bias, 0)
    
    print(f"VGGish model created with {num_classes} output classes")
    return model