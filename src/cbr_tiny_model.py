"""
Track 3: CBR-Tiny CNN from Scratch
Ablation baseline reproducing the published literature architecture (Lachance et al. 2021/2022).
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class CBRBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        return self.pool(self.relu(self.bn(self.conv(x))))

class CBRTiny(nn.Module):
    def __init__(self, num_classes=1, task="classification"):
        super().__init__()
        self.task = task
        # 4 sequential Conv-BatchNorm-ReLU blocks
        self.features = nn.Sequential(
            CBRBlock(1, 32),
            CBRBlock(32, 64),
            CBRBlock(64, 128),
            CBRBlock(128, 256)
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        feat = self.features(x)
        pooled = self.global_pool(feat)
        flat = torch.flatten(pooled, 1)
        out = self.classifier(flat)
        return out.squeeze(-1)

class OCTDualScanDataset(Dataset):
    def __init__(self, df, transform=None, is_train=True):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train
        
        # Dual augmentations matching literature (flips, random rotation, contrast jitter)
        if transform is not None:
            self.transform = transform
        elif is_train:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])

    def __len__(self):
        # 2 scans per patient (H and V treated as dual samples)
        return len(self.df) * 2

    def __getitem__(self, idx):
        patient_idx = idx // 2
        is_vertical = (idx % 2 == 1)
        row = self.df.iloc[patient_idx]
        
        img_path = row['oct_baseline_V_path'] if is_vertical else row['oct_baseline_H_path']
        
        with Image.open(img_path) as img:
            img_tensor = self.transform(img.convert('L'))
            
        label_clf = torch.tensor(row['VA_gain_6mo_binary'], dtype=torch.float32)
        label_reg = torch.tensor(row['delta_VA_6months'], dtype=torch.float32)
        
        return {
            'image': img_tensor,
            'label_clf': label_clf,
            'label_reg': label_reg,
            'patient_id': row['id']
        }
