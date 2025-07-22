import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import re

class AttentionBlock(nn.Module):
    def __init__(self, g_channels, x_channels, inter_channels):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(g_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(x_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, dropout_p=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.attention = AttentionBlock(out_channels, skip_channels, out_channels // 2)
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)  # 上采样
        # 如果skip大小和x不一致，则对skip做插值resize
        if skip.shape[2:] != x.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
        skip = self.attention(x, skip)
        x = torch.cat([x, skip], dim=1)
        x = self.dropout(x)
        return self.conv(x)

class AttentionUNetResNet34(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, pretrained=True, freeze_encoder=True):
        super().__init__()
        resnet = models.resnet34(pretrained=pretrained)

        # 修改第一层conv1以支持不同输入通道数
        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 定义分层结构（按解冻顺序排列）
        self.stage_layers = {
            "layer4": resnet.layer4,  # 最深层
            "layer3": resnet.layer3,
            "layer2": resnet.layer2,
            "layer1": resnet.layer1,
            "conv1": [resnet.conv1, resnet.bn1]
        }

        if freeze_encoder:
            self.freeze_all()
        else:
            self.unfreeze_all()
         
        self.input_layer = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool  # 标准ResNet第一层卷积后的下采样
        )
        self.encoder1 = resnet.layer1  # 64 channels
        self.encoder2 = resnet.layer2  # 128 channels
        self.encoder3 = resnet.layer3  # 256 channels
        self.encoder4 = resnet.layer4  # 512 channels

        self.se2 = SEBlock(64)     # after encoder1 (x2)
        self.se3 = SEBlock(128)    # after encoder2 (x3)
        self.se4 = SEBlock(256)    # after encoder3 (x4)
        self.se5 = SEBlock(512)    # after encoder4 (x5)
        
        # Decoder
        self.up1 = UpBlock(512, 256, 256, dropout_p=0.3)
        self.up2 = UpBlock(256, 128, 128, dropout_p=0.3)
        self.up3 = UpBlock(128, 64, 64, dropout_p=0.3)
        self.up4 = UpBlock(64, 64, 32, dropout_p=0.3)

        self.final_up = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.input_layer(x)      # [B, 64, 64, 64] 因为有maxpool，256/4=64
        x2 = self.encoder1(x1)        # [B, 64, 64, 64]
        x2 = self.se2(x2)
        x3 = self.encoder2(x2)        # [B, 128, 32, 32]
        x3 = self.se3(x3)
        x4 = self.encoder3(x3)        # [B, 256, 16, 16]
        x4 = self.se4(x4)
        x5 = self.encoder4(x4)        # [B, 512, 8, 8]
        x5 = self.se5(x5)
        
        d1 = self.up1(x5, x4)         # [B, 256, 16, 16]
        d2 = self.up2(d1, x3)         # [B, 128, 32, 32]
        d3 = self.up3(d2, x2)         # [B, 64, 64, 64]
        d4 = self.up4(d3, x1)         # [B, 32, 64, 64]
        d5 = self.final_up(d4)        # [B, 32, 128, 128]
     
        out = self.out_conv(d5)       # [B, out_channels, 256, 256]
        return out

    def freeze_all(self):
        #冻结整个编码器参数
        for name, param in self.named_parameters():
            if "encoder" in name or any(layer in name for layer in ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]):
                param.requires_grad = False

    def unfreeze_all(self):
        #解冻全部参数
        for param in self.parameters():
            param.requires_grad = True    
    
    def unfreeze_from(self, stage_name="layer4"):
        import re
        #分层冻结实现
        if stage_name == "all":
            for param in self.parameters():
                param.requires_grad = False
            return
        
        # 解冻指定阶段及更深的层
        stages_order = ["layer4", "layer3", "layer2", "layer1", "conv1"]
        target_idx = stages_order.index(stage_name)
        
        pattern = re.compile(r"\.({})\.".format("|".join(stages_order)))
       # 首先冻结所有层
        for param in self.parameters():
            param.requires_grad = False
        # 解冻目标层及更深层
        for name, param in self.named_parameters():
            # 解冻解码器部分（始终需要训练）
            if any(key in name for key in ["up", "final_up", "out_conv"]):
                param.requires_grad = True
                continue
            if "se" in name:
                param.requires_grad = True
                continue
            
            # 解冻目标层及更深层
            for i, layer_name in enumerate(stages_order):
                # 精确匹配层级名称（避免部分匹配）
                if f".{layer_name}." in name or name.endswith(f".{layer_name}"):
                    if i <= target_idx:
                        param.requires_grad = True
                    break  # 找到匹配后跳出循环
                
    def print_trainable(self):
        #打印当前可训练参数
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"[Trainable] {name}")