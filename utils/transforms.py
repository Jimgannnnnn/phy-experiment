import torchvision.transforms as transforms
import random
import torch

class HighResAdapter:
    """高分辨率图像适配器"""
    def __init__(self, crop_mode="random"):
        self.crop_mode = crop_mode  # random/center
        
    def __call__(self, img):
        # 第一阶段：保持宽高比缩小 (1920x1080 → 256x144)
        resize = transforms.Resize(256, max_size=144)
        img = resize(img)
        
        # 第二阶段：裁剪策略
        if self.crop_mode == "random":
            i = random.randint(0, img.height - 224)
            j = random.randint(0, img.width - 224)
            crop = transforms.functional.crop(img, i, j, 224, 224)
        else:  # center模式
            crop = transforms.CenterCrop(224)(img)
        return crop

def get_train_transforms():
    """训练集预处理、数据增强"""
    return transforms.Compose([
        HighResAdapter(crop_mode="random"),  # 随机裁剪
        transforms.RandomHorizontalFlip(p=0.5),  
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 亮度对比度微调
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])  # ImageNet标准参数
    ])

def get_val_transforms():
    """验证/测试集预处理（无增强）"""
    return transforms.Compose([
        HighResAdapter(crop_mode="center"),  # 中心裁剪
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

def transform_invert(img_tensor):
    """张量→PIL图像
    参数：img_tensor - 归一化后的图像张量 (C, H, W)
    """
    reverse_norm = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    img = reverse_norm(img_tensor).clamp(0, 1)
    return transforms.ToPILImage()(img)
