import torch
import os
from torch.utils.data import Dataset
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SimpleTransform:
    def __init__(self):
        self.transform = A.Compose([
            A.Resize(256, 256, always_apply=True),
            A.Normalize(mean=[0.0], std=[1.0]),
            ToTensorV2(),
        ])
    
    def __call__(self, image, mask=None):
        if mask is None:
            result = self.transform(image=image)
            return result['image']
        else:
            result = self.transform(image=image, mask=mask)
            return result['image'], result['mask']
            
class DualTransform:
    def __init__(self, p=0.5):
        self.transform = A.Compose([
            A.Resize(256, 256, always_apply=True),
            A.RandomResizedCrop(height=256, width=256, scale=(0.6, 1.0), p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            
            A.GridDistortion(p=0.2),
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
            A.Normalize(mean=[0.0], std=[1.0]),
            ToTensorV2(),
        ], p=p)
    
    def __call__(self, image, mask=None):
        
        
        if mask is None:
            return self.transform(image=image)['image']
            
        else:
            res = self.transform(image=image, mask=mask)
            return res['image'], res['mask']  # 返回两个 tensor

    
    def high_freq_enhance(self, img):
        # 如果是掩码，直接返回
        if img.max() <= 1.0 and (img == 0).all() or (img == 1).all():
            return img
            
        # 转换为单通道
        if img.shape[-1] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img.squeeze()
        
        # 标准化到[0,1]
        if gray.max() > 1:
            gray = gray / 255.0
        
        # Sobel边缘检测（float32）
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge = np.sqrt(sobelx**2 + sobely**2)
        
        # 增强并限制范围
        enhanced = np.clip(gray + 0.5*edge, 0, 1)
        
        # 恢复通道维度
        if img.shape[-1] == 3:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        else:
            enhanced = enhanced[..., np.newaxis]
        
        return enhanced.astype(np.float32)

class TestTransform:
    def __init__(self):
        self.transform = A.Compose([
            A.Resize(256, 256, always_apply=True),
            A.Normalize(mean=[0.0], std=[1.0]),
            ToTensorV2(),
        ])
    
    def __call__(self, image, mask=None):  
        if mask is None:
            # 只转换图像
            transformed = self.transform(image=image)
            return transformed['image']
        else:
            # 同时转换图像和掩码
            transformed = self.transform(image=image, mask=mask)
            return transformed['image'], transformed['mask']
            
class MedicalDataset(Dataset):
    def __init__(self, image_dir, mask_dir, indices=None, transform=None, mode="train", stage=1):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mode = mode
        self.stage = stage  
        
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
        
        if indices is not None:
            self.images = [self.images[i] for i in indices]
            self.masks = [self.masks[i] for i in indices]
        
        self.test_transform = TestTransform()
        self.simple_transform = SimpleTransform()
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # H x W
        image = np.expand_dims(image, axis=-1)  # H x W x 1

        # 显式转 float32，并归一化，避免 ByteTensor 报错
        image = image.astype(np.float32) / 255.0
        
        # 根据阶段选择预处理
        if self.mode != "test":
            mask_path = os.path.join(self.mask_dir, self.masks[idx])
            mask = cv2.imread(mask_path, 0)
            mask = (mask > 127).astype(np.float32)  # 二值化为 0/1
            
            if self.transform:
            # 阶段1使用简化预处理
                if self.stage == 1:
                    transformed = self.simple_transform(image=image, mask=mask)
                    
            # 阶段2和3使用完整预处理
                else:
                    transformed = self.transform(image=image, mask=mask)
                
                image, mask = transformed

                # 确保 mask 有通道维度
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
                
            return image, mask

        else:  # 测试模式
            mask_path = os.path.join(self.mask_dir, self.masks[idx])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if mask is None:
                raise ValueError(f"Failed to read mask: {mask_path}")
                
            mask = np.expand_dims(mask, axis=-1)  
            mask = (mask > 127).astype(np.float32)  # 二值化为 0/1
            
            image, mask = self.test_transform(image=image, mask=mask)
                    
            # 确保 mask 有通道维度
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
                
            return image, mask

