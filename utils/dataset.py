import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from .transforms import get_train_transforms, get_val_transforms

class RegressionDataset(Dataset):
    def __init__(self, img_dir, label_file, is_train=True, stats=None):
        """
        img_dir: 图片目录 (train/val/test)
        label_file: CSV文件路径 (含image_path, value列)
        is_train: 是否为训练模式
        """
        self.img_dir = img_dir
        self.labels = pd.read_csv(label_file)
        self.transform = get_train_transforms() if is_train else get_val_transforms()
        
        # 仅训练集计算归一化参数
        if is_train:
            self.value_mean = self.labels['value'].mean()
            self.value_std = self.labels['value'].std()
            self.norm_labels = (self.labels - self.value_mean) / self.value_std
        else:
            # 测试集/验证集：必须使用训练集的统计量
            assert stats is not None, "测试集必须传入训练集的统计量"
            self.value_mean = stats['mean']
            self.value_std = stats['std']
            self.norm_labels = (self.labels - self.value_mean) / self.value_std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.basename(self.labels.iloc[idx]['image_path'])
        img_path = os.path.join(self.img_dir, img_name)
        
        # 图像加载与转换
        img = Image.open(img_path).convert('RGB')  # 强制转为RGB三通道
        value = self.labels.iloc[idx]['value']
        
        # 标签标准化
        normalized_value = (value - self.value_mean) / self.value_std
        if self.transform:
            img = self.transform(img)
            
        return img, torch.tensor(normalized_value, dtype=torch.float32)
    
    def denormalize(self, value):
        """将标准化值还原为原始量纲"""
        return value * self.value_std + self.value_mean
    