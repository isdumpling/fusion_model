import torch


def hms_string(sec):
    """
    Convert seconds to a formatted string (hours:minutes:seconds)
    
    Args:
        sec: Time in seconds
    
    Returns:
        Formatted time string
    """
    hours = int(sec // 3600)
    minutes = int((sec % 3600) // 60)
    seconds = int(sec % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def calculate_flops(model, input_shape=(1, 1, 96, 64), device='cuda'):
    """
    计算模型的FLOPs (浮点运算次数)
    
    注意：此函数会给模型添加hooks用于计算FLOPs，因此建议传入临时模型而非训练模型
    
    Args:
        model: PyTorch模型（建议是临时模型）
        input_shape: 输入张量的形状 (batch, channels, height, width)
        device: 计算设备 ('cuda' 或 'cpu')
    
    Returns:
        flops: FLOPs数量
        params: 参数数量
    """
    try:
        from thop import profile, clever_format
        
        # 确保模型处于评估模式
        model.eval()
        
        # 创建输入张量
        input_tensor = torch.randn(*input_shape).to(device)
        
        # 先做一次前向传播以确保模型完全初始化（特别是对于有动态层的模型）
        with torch.no_grad():
            _ = model(input_tensor)
        
        # 重新创建输入张量用于profile
        input_tensor = torch.randn(*input_shape).to(device)
        
        # 计算FLOPs和参数量
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)
        
        # 格式化输出 (转换为更易读的格式，如 M, G)
        flops_formatted, params_formatted = clever_format([flops, params], "%.3f")
        
        return flops, params, flops_formatted, params_formatted
    
    except ImportError:
        print("警告: 未安装 thop 库，无法计算FLOPs。")
        print("请运行: pip install thop")
        return None, None, "N/A", "N/A"
    except Exception as e:
        print(f"计算FLOPs时出错: {e}")
        # 如果thop失败，尝试手动计算参数量
        try:
            params = sum(p.numel() for p in model.parameters())
            params_formatted = f"{params/1e6:.3f}M"
            print(f"无法计算FLOPs，但成功统计参数量: {params_formatted}")
            return None, params, "N/A", params_formatted
        except:
            return None, None, "N/A", "N/A"
