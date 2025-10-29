# datasets/dataloader.py

import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_LENGTH = 64  
N_MELS = 96


class FilteredCoughSegmentDataset(Dataset):
    """
    筛选后的 Cough 片段数据集
    包含通过 Stage 1 模型滑动窗口筛选出的高置信度 cough 片段
    """
    def __init__(self, segments, data_dir, transform=None, target_length=TARGET_LENGTH, n_mels=N_MELS):
        """
        Args:
            segments: List[Dict] - 每个元素包含 {'file_path', 'label', 'start_frame', 'end_frame', 'confidence'}
            data_dir: 数据根目录
            transform: 数据变换
            target_length: 目标长度（帧数）
            n_mels: mel频谱的频率维度
        """
        self.segments = segments
        self.data_dir = data_dir
        self.transform = transform
        self.target_length = target_length
        self.n_mels = n_mels
        
        # 为兼容性，统计标签
        self.targets = [seg['label'] for seg in segments]
        self.img_num_list = [0] * 2  # 二分类
        for label in self.targets:
            self.img_num_list[label] += 1
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        segment = self.segments[idx]
        audio_path = os.path.join(self.data_dir, segment['file_path'])
        label = segment['label']
        start_frame = segment['start_frame']
        end_frame = segment['end_frame']
        
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
        except Exception as e:
            print(f"Error loading file {audio_path}: {e}")
            return torch.zeros((1, self.n_mels, self.target_length)), label, idx, torch.zeros((1, self.target_length * 160))
        
        # 重采样到 16kHz
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
        
        # 转换为单声道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # 生成完整的 Mel 频谱图
        mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=self.n_mels,
            f_min=125,
            f_max=7500
        )
        mel_spectrogram = mel_spectrogram_transform(waveform)
        log_mel_spectrogram = torch.log(mel_spectrogram + 1e-9)
        
        # 提取指定的片段
        segment_spec = log_mel_spectrogram[:, :, start_frame:end_frame]
        
        # 调整到目标长度
        if segment_spec.shape[2] > self.target_length:
            segment_spec = segment_spec[:, :, :self.target_length]
        else:
            padding = self.target_length - segment_spec.shape[2]
            segment_spec = torch.nn.functional.pad(segment_spec, (0, padding))
        
        # 提取对应的波形片段
        start_sample = start_frame * 160  # hop_length = 160
        end_sample = end_frame * 160
        waveform_segment = waveform[:, start_sample:end_sample]
        
        # 调整波形到目标长度
        target_waveform_length = self.target_length * 160
        if waveform_segment.shape[1] > target_waveform_length:
            waveform_segment = waveform_segment[:, :target_waveform_length]
        else:
            padding = target_waveform_length - waveform_segment.shape[1]
            waveform_segment = torch.nn.functional.pad(waveform_segment, (0, padding))
        
        if self.transform:
            segment_spec = self.transform(segment_spec)
        
        return segment_spec, label, idx, waveform_segment  

class AudioLongTailDataset(Dataset):
    def __init__(self, data_frame, data_dir, transform=None, target_length=TARGET_LENGTH, 
                 n_mels=N_MELS, mode='train', domain='source'):
        """
        Args:
            data_frame: 数据框架
            data_dir: 数据根目录
            transform: 数据变换
            target_length: 目标长度（帧数）
            n_mels: mel频谱的频率维度
            mode: 'train' 或 'test'
            domain: 'source' 或 'target' - 用于确定验证策略
        """
        self.data_frame = data_frame
        self.data_dir = data_dir
        self.transform = transform
        self.target_length = target_length
        self.n_mels = n_mels
        self.mode = mode
        self.domain = domain
        self.targets = data_frame['label'].tolist()
        # 为长尾分析计算每个类别的样本数
        self.img_num_list = [0] * len(data_frame['label'].unique())
        for label in self.targets:
            self.img_num_list[label] += 1

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # 注意：路径拼接时，data_dir 应该是根目录
        audio_path = os.path.join(self.data_dir, self.data_frame.iloc[idx, 0])
        label = self.data_frame.iloc[idx, 1]

        try:
            waveform, sample_rate = torchaudio.load(audio_path)
        except Exception as e:
            print(f"Error loading file {audio_path}: {e}")
            # 返回固定长度的0张量
            return torch.zeros((1, self.n_mels, self.target_length)), -1, idx

        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,  # 25ms window
            win_length=400,
            hop_length=160,  # 10ms hop
            n_mels=self.n_mels,
            f_min=125,
            f_max=7500
        )
        mel_spectrogram = mel_spectrogram_transform(waveform)
        log_mel_spectrogram = torch.log(mel_spectrogram + 1e-9)

        # 数据长度处理逻辑：统一调整到固定长度
        if log_mel_spectrogram.shape[2] > self.target_length:
            log_mel_spectrogram = log_mel_spectrogram[:, :, :self.target_length]
        else:
            padding = self.target_length - log_mel_spectrogram.shape[2]
            log_mel_spectrogram = torch.nn.functional.pad(log_mel_spectrogram, (0, padding))

        if self.transform:
            log_mel_spectrogram = self.transform(log_mel_spectrogram)

        # 目标域：总是返回 waveform（用于 Stage 2 的强增强）
        if self.domain == 'target':
            # 统一调整 waveform 长度以便 batch 化
            target_waveform_length = self.target_length * 160  # 160 是 hop_length
            if waveform.shape[1] > target_waveform_length:
                waveform = waveform[:, :target_waveform_length]
            else:
                padding = target_waveform_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            
            return log_mel_spectrogram, label, idx, waveform
        
        return log_mel_spectrogram, label, idx

def _load_domain_data(domain_dir, class_to_idx):
    print("-" * 50)
    print(f"[*] 正在扫描这个目录: {domain_dir}")
    if not os.path.isdir(domain_dir):
        print(f"[!] 警告: 目录不存在！")
    file_paths, labels = [], []
    for class_name in class_to_idx.keys():
        class_dir = os.path.join(domain_dir, class_name)
        if os.path.isdir(class_dir):
            for file_name in os.listdir(class_dir):
                if file_name.endswith(('.wav', '.flac', '.mp3')):
                    # 存储相对路径，方便拼接
                    file_paths.append(os.path.join(os.path.basename(domain_dir), class_name, file_name))
                    labels.append(class_to_idx[class_name])
    return pd.DataFrame({'file_path': file_paths, 'label': labels})

def get_cross_domain_audio_dataset(root, args):
    source_dir = os.path.join(root, args.source_domain)
    target_dir = os.path.join(root, args.target_domain)

    class_to_idx = {'cough': 0, 'non-cough': 1} # 二分类

    source_df = _load_domain_data(source_dir, class_to_idx)
    target_df = _load_domain_data(target_dir, class_to_idx)

    seed = int(args.seed) if args.seed != 'None' else None

    # 划分源域
    source_train_df, source_test_df = train_test_split(source_df, test_size=0.2, random_state=seed, stratify=source_df['label'])

    # 划分目标域
    target_train_df, target_test_df = train_test_split(target_df, test_size=0.2, random_state=seed, stratify=target_df['label'])

    # 创建Dataset实例，注意 mode 和 domain 参数
    source_train_dataset = AudioLongTailDataset(source_train_df, root, mode='train', domain='source')
    source_test_dataset = AudioLongTailDataset(source_test_df, root, mode='test', domain='source')
    target_train_dataset = AudioLongTailDataset(target_train_df, root, mode='train', domain='target') 
    target_test_dataset = AudioLongTailDataset(target_test_df, root, mode='test', domain='target')

    print(f"Source Domain: Train={len(source_train_dataset)}, Test={len(source_test_dataset)}")
    print(f"Target Domain: Train(unlabeled)={len(target_train_dataset)}, Test={len(target_test_dataset)}")
    print(f"Source Train Class Counts: {source_train_dataset.img_num_list}")

    return source_train_dataset, source_test_dataset, target_train_dataset, target_test_dataset