import torch.nn as nn
from torchvision.models import vgg19_bn

class VGG19Reg(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = vgg19_bn(pretrained=True).features
        for param in self.features.parameters():
            param.requires_grad = False  # 冻结卷积层
          
        # 自定义回归头
        self.regressor = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),  # 匹配VGG19最后一层输出
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 1)   # 输出层
        )
        
    def forward(self, x):
        x = self.features(x)
        return self.regressor(x)
    