import argparse
import torch
import numpy as np
from PIL import Image
from models.vgg19_bn_reg import VGG19Reg  # 导入自定义模型
from utils.transforms import get_val_transforms, transform_invert
from utils.dataset import RegressionDataset

def load_model(model_path, device, checkpoint=None):
    model = VGG19Reg()  # 创建模型实例
    if checkpoint is None:
        checkpoint = torch.load(model_path, map_location=device)
    
    # 加载模型权重（允许部分匹配）
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)  
    
    model.to(device)
    model.eval()  # 切换为评估模式（固定BN/Dropout）
    return model

def preprocess_image(image_path, transform):
    """图像预处理流程"""
    img = Image.open(image_path).convert('RGB')
    original_img = img.copy()  # 保存原始图像用于可视化
    img_tensor = transform(img).unsqueeze(0)  # 添加batch维度
    print(f"图像预处理完成 | 尺寸: {img.size}->{img_tensor.shape}")
    return img_tensor, original_img

def predict(model, input_tensor, value_mean, value_std, device):
    """执行模型预测并反归一化结果"""
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        prediction = model(input_tensor)
    
    # 反归一化到原始量纲
    raw_pred = prediction.item() * value_std + value_mean
    print(f"预测结果: {prediction.item():.4f} (标准化) → {raw_pred:.4f} (原始量纲)")
    return raw_pred

def visualize_results(original_img, prediction, save_path=None):
    """可视化原始图像和预测结果"""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title("原始图像")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.text(0.5, 0.5, f"预测值: {prediction:.4f}", 
             fontsize=15, ha='center', va='center')
    plt.axis('off')
    plt.title("回归预测结果")
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"结果已保存至: {save_path}")
    plt.show()

if __name__ == "__main__":
    # 参数解析
    parser = argparse.ArgumentParser(description='VGG19回归模型测试')
    parser.add_argument('--image', type=str, required=True, help='测试图像路径')
    parser.add_argument('--model', type=str, default='best_model.pth', help='模型权重路径')
    parser.add_argument('--save', type=str, help='结果保存路径(可选)')
    args = parser.parse_args()
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载标签统计量
    checkpoint = torch.load(args.model, map_location=device)  # 加载整个checkpoint
    label_stats = checkpoint['label_stats']  # 提取训练集统计量
    value_mean = label_stats['mean']
    value_std = label_stats['std']


    # 初始化预处理流程
    transform = get_val_transforms()  # 使用验证集预处理
    
    # 主执行流程
    try:
        model = load_model(args.model, device, checkpoint)
        input_tensor, original_img = preprocess_image(args.image, transform)
        prediction = predict(model, input_tensor, value_mean, value_std, device)
        visualize_results(original_img, prediction, args.save)
    
    except Exception as e:
        print(f"测试失败: {str(e)}")
        