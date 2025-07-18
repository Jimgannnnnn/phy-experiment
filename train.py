import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.vgg19_bn_reg import VGG19Reg
from utils.dataset import RegressionDataset
from utils.transforms import get_train_transforms, get_val_transforms
import argparse

# 参数配置
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/', help='数据集根目录')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批量大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='初始学习率')
    parser.add_argument('--patience', type=int, default=7, help='早停等待轮数')
    parser.add_argument('--save_dir', type=str, default='checkpoints/', help='模型保存路径')
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 确保目录存在
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 数据预处理
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()
    
    # 数据集加载
    train_dataset = RegressionDataset(
        root_dir=args.data_dir,
        transform=train_transform,
        is_train=True
    )
    val_dataset = RegressionDataset(
        root_dir=args.data_dir,
        transform=val_transform,
        is_train=False
    )
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # 模型初始化
    model = VGG19Reg()
    

    # 加载预训练权重
    pretrained_dict = torch.load('vgg19_bn-c79401a0.pth')
    model_dict = model.state_dict()
    
    # 过滤全连接层,自定义全连接层的key有regressor标记
    pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                      if k in model_dict and "regressor" not in k}
    
    # 冻结卷积层参数
    for name, param in model.named_parameters():
        if "features" in name:
            param.requires_grad = False  # 冻结卷积层
            print(f"冻结层: {name}")
    
    # 加载预训练权重（跳过全连接层）
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.to(device)
    
    # 优化器（仅优化未冻结参数）
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_params, lr=args.lr, weight_decay=1e-4)
    
    # 损失函数（回归任务采用均方差）
    criterion = nn.MSELoss()
    
    # 学习率调度器、早停
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=args.patience//2
    )
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    # 训练循环
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            
            # 梯度裁剪防震荡
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item() * images.size(0)
        
        # 计算epoch平均损失
        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f}")
        
        # 学习率调整、早停
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # 保存最佳模型

            label_stats = {
                'mean': train_dataset.value_mean, 
                'std': train_dataset.value_std
            }

            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'label_stats': label_stats  # 保存标准化参数
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"保存最佳模型 (Val Loss: {val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"早停触发，连续 {args.patience} 轮未提升")
                break

if __name__ == '__main__':
    main()
    