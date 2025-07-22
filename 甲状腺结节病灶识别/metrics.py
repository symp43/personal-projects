import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms

# 损失函数定义
class StabilizedBCE(nn.Module):
    def __init__(self, pos_weight=10.0):
        super().__init__()
        self.pos_weight = pos_weight
    
    def forward(self, pred, target):
        # 数值稳定性保护
        pred = torch.clamp(pred, -10, 10)
        return F.binary_cross_entropy_with_logits(
            pred, target, 
            pos_weight=torch.tensor([self.pos_weight], device=pred.device)
        )
        
def dice_coeff(pred, target):
    smooth = 1.
    #pred = torch.sigmoid(pred)  # Sigmoid激活
    pred_flat = (pred > 0.5).float().view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

def iou_score(pred, target):
    smooth = 1.
    pred = torch.sigmoid(pred)  # Sigmoid激活
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    return (intersection + smooth) / (union + smooth)
    
#数据可视化
def visualize_data(dataset, title="数据样本", num_samples=4):
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples*5))
    
    for i in range(num_samples):
        image, mask = dataset[i]
        
        # 转换为numpy数组并调整维度
        image_np = image.permute(1, 2, 0).numpy().squeeze()
        
        mask_np = mask.squeeze().numpy()
        
        # 显示图像
        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title("图像")
        axes[i, 0].axis("off")
        
        # 显示掩码
        axes[i, 1].imshow(mask_np, cmap="gray")
        axes[i, 1].set_title("掩码")
        axes[i, 1].axis("off")
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    
#模型预测可视化
def plot_prediction(model, dataset, idx=0, title="预测结果"):
    model.eval()
    image, mask = dataset[idx]
    image_tensor = image.unsqueeze(0).to(model.device if hasattr(model, 'device') else "cuda")
    
    with torch.no_grad():
        pred = model(image_tensor)
        pred = torch.sigmoid(pred)
        pred = (pred > best_thresh).float().cpu().squeeze()
    
    # 转换为numpy数组
    image_np = image.permute(1, 2, 0).numpy()
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
    mask_np = mask.squeeze().numpy()
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image_np)
    axes[0].set_title("输入图像")
    axes[0].axis("off")
    
    axes[1].imshow(mask_np, cmap="gray")
    axes[1].set_title("真实掩码")
    axes[1].axis("off")
    
    axes[2].imshow(pred.numpy(), cmap="gray")
    axes[2].set_title("预测掩码")
    axes[2].axis("off")
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

        
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        self.gamma = gamma 
        
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
         # 计算交集和并集（添加平滑项）
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        # 添加gamma调整难易样本权重
        p_t = pred_flat * target_flat + (1 - pred_flat) * (1 - target_flat)
        modulating_factor = (1.0 - p_t) ** self.gamma
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return (1 - dice) * modulating_factor.mean()

class PrecisionBalancedLoss(nn.Module):
    def __init__(self, pos_weight=4.0, gamma=2.0, smooth=1.0):
        
        super().__init__()
        self.pos_weight = pos_weight
        self.gamma = gamma
        self.smooth = smooth
        
    def forward(self, pred, target):
        # 加权BCE部分（核心正样本补偿）
        bce_loss = F.binary_cross_entropy_with_logits(
            pred, target, 
            pos_weight=torch.tensor([self.pos_weight], device=pred.device)
        )
        
        # Focal调制（关注难样本）
        pred_sig = torch.sigmoid(pred)
        pt = torch.where(target == 1, pred_sig, 1 - pred_sig)
        focal_factor = (1 - pt) ** self.gamma
        focal_loss = focal_factor * bce_loss
        
        # Dice部分（增强区域一致性）
        pred_flat = pred_sig.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        dice_loss = 1 - dice
        
        # 自适应权重：当召回率低时增加Dice权重
        recall = intersection / (target_flat.sum() + 1e-6)
        dice_weight = min(0.8, max(0.4, 1.0 - recall.item() * 0.5))
        
        return 0.6 * focal_loss.mean() + dice_weight * dice_loss
        
class DiceFocalLoss(nn.Module):
    def __init__(self, alpha=0.7, gamma=2.0, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, pred, target):
        # Sigmoid激活 
        pred_sig = torch.sigmoid(pred)
        
        # ===== 计算Dice Loss =====
        pred_flat = pred_sig.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_score

        # ===== 计算Focal Loss =====
        # 使用PyTorch内置的稳定BCE实现
        bce_loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none'
        )
        
        # Focal调制因子
        p_t = torch.exp(-bce_loss)  # 概率值
        focal_factor = (1 - p_t) ** self.gamma
        
        focal_loss = focal_factor * bce_loss
        focal_loss = focal_loss.mean()

        # 组合损失
        return self.alpha * dice_loss + (1 - self.alpha) * focal_loss  