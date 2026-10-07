"""
Track 1 Upgrade: Domain-Adapted Retinal Siamese CNN & Contrastive Self-Supervised Pretraining
Processes pre-operative Horizontal & Vertical HD-OCT B-scans with domain-specific retinal augmentations.
Pretrains on all 494 available B-scans via contrastive InfoNCE loss (zero label leakage)
and extracts fine-grained retinal feature representations.
"""

import os
import glob
import math
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==============================================================================
# 1. Custom Domain-Specific Retinal Image Augmentations (Pure PIL/NumPy)
# ==============================================================================

class RetinalAugmentor:
    """
    Retinal-specific augmentations that respect anatomical invariants:
    - Horizontal flip (anatomically valid for central foveal B-scans)
    - Contrast & Brightness jitter (models differing OCT signal-to-noise / cataract attenuation)
    - Sub-pixel shift & elastic scaling (models subtle head tilt / fixation drift)
    - Central foveal cropping (focuses on neurosensory retina & RPE band)
    """
    def __init__(self, target_size=(224, 224), is_train=True):
        self.target_size = target_size
        self.is_train = is_train

    def __call__(self, pil_img):
        img = pil_img.convert('L') # Grayscale B-scan
        W, H = img.size
        
        # Center crop central 70% width and 60% height around fovea
        crop_w = int(W * 0.70)
        crop_h = int(H * 0.60)
        left = (W - crop_w) // 2
        top = int(H * 0.25) # Retina is centered in middle-lower third
        img = img.crop((left, top, left + crop_w, top + crop_h))
        
        if self.is_train:
            # 1. Random Horizontal Flip (50% prob)
            if random.random() > 0.5:
                img = ImageOps.mirror(img)
                
            # 2. Random Contrast Jitter [0.8, 1.25]
            if random.random() > 0.3:
                factor = random.uniform(0.8, 1.25)
                img = ImageEnhance.Contrast(img).enhance(factor)
                
            # 3. Random Brightness Jitter [0.85, 1.15]
            if random.random() > 0.3:
                factor = random.uniform(0.85, 1.15)
                img = ImageEnhance.Brightness(img).enhance(factor)
                
            # 4. Slight rotation (+/- 4 degrees)
            if random.random() > 0.5:
                angle = random.uniform(-4.0, 4.0)
                img = img.rotate(angle, resample=Image.BILINEAR)
                
        # Resize to standard input dimensions
        img = img.resize(self.target_size, resample=Image.BILINEAR)
        
        # Normalize to [-1, 1] tensor
        arr = np.array(img, dtype=np.float32) / 255.0
        # Retinal band normalization (zero mean, unit variance)
        mean, std = arr.mean(), arr.std() + 1e-6
        arr = (arr - mean) / std
        
        tensor = torch.from_numpy(arr).unsqueeze(0) # (1, H, W)
        return tensor


# ==============================================================================
# 2. Dual-Stream Siamese Retinal CNN Architecture
# ==============================================================================

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.act1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.act2 = nn.LeakyReLU(0.1, inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        res = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act2(out + res)
        return out

class RetinalBackbone(nn.Module):
    """
    4-Stage Residual Retinal Convolutional Feature Extractor.
    Designed specifically for layered OCT speckle patterns and foveal lumen detection.
    """
    def __init__(self, embed_dim=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2, bias=False), # (112, 112)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.stage1 = ConvBlock(32, 64, stride=2)  # (56, 56)
        self.stage2 = ConvBlock(64, 128, stride=2) # (28, 28)
        self.stage3 = ConvBlock(128, 256, stride=2) # (14, 14)
        self.stage4 = ConvBlock(256, 256, stride=2) # (7, 7)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, embed_dim)
        )

    def forward(self, x):
        feat = self.stem(x)
        feat = self.stage1(feat)
        feat = self.stage2(feat)
        feat = self.stage3(feat)
        feat = self.stage4(feat)
        pooled = self.pool(feat).flatten(1)
        emb = self.proj(pooled)
        return emb

class DualStreamSiameseOCT(nn.Module):
    """
    Dual-stream architecture fusing Horizontal (H) and Vertical (V) B-scans.
    Weight sharing across meridians preserves cross-hair rotational equivariance.
    """
    def __init__(self, embed_dim=64):
        super().__init__()
        self.encoder = RetinalBackbone(embed_dim=embed_dim)
        self.fusion_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, img_h, img_v):
        emb_h = self.encoder(img_h)
        emb_v = self.encoder(img_v)
        # Symmetrized dual-axis fusion
        joint = torch.cat([emb_h, emb_v], dim=1)
        fused = self.fusion_head(joint)
        return fused, emb_h, emb_v


# ==============================================================================
# 3. Contrastive Self-Supervised Pretraining Dataset & Loss
# ==============================================================================

class UnlabeledOCTDataset(Dataset):
    """
    Loads pre-op H & V B-scans from all available directories (train, val, test, others).
    Generates two stochastic augmented views (x_i, x_j) for contrastive learning.
    """
    def __init__(self, base_dir="."):
        self.pairs = []
        augmentor = RetinalAugmentor(is_train=True)
        self.augmentor = augmentor
        
        # Scan all directories
        for split in ['train', 'val', 'test', 'others']:
            oct_dir = os.path.join(base_dir, split, 'octs')
            if not os.path.exists(oct_dir):
                continue
            h_files = glob.glob(os.path.join(oct_dir, "*_baseline_H.tiff"))
            for h_path in h_files:
                v_path = h_path.replace("_baseline_H.tiff", "_baseline_V.tiff")
                if os.path.exists(v_path):
                    self.pairs.append((h_path, v_path))
                    
        print(f"Loaded {len(self.pairs)} valid B-scan pairs across entire cohort (train/val/test/others).")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        h_path, v_path = self.pairs[idx]
        with Image.open(h_path) as img_h, Image.open(v_path) as img_v:
            # View 1
            t1_h = self.augmentor(img_h)
            t1_v = self.augmentor(img_v)
            # View 2
            t2_h = self.augmentor(img_h)
            t2_v = self.augmentor(img_v)
            
        return (t1_h, t1_v), (t2_h, t2_v)


def nt_xent_contrastive_loss(z_i, z_j, temperature=0.1):
    """
    Normalized Temperature-scaled Cross Entropy Loss (SimCLR / InfoNCE).
    Brings representations of the same patient's augmented scans close, pushes different patients apart.
    """
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    
    batch_size = z_i.shape[0]
    representations = torch.cat([z_i, z_j], dim=0) # (2N, D)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
    
    sim_ij = torch.diag(similarity_matrix, batch_size)
    sim_ji = torch.diag(similarity_matrix, -batch_size)
    positives = torch.cat([sim_ij, sim_ji], dim=0)
    
    mask = ~torch.eye(2 * batch_size, 2 * batch_size, dtype=torch.bool, device=z_i.device)
    negatives = similarity_matrix[mask].view(2 * batch_size, -1)
    
    logits = torch.cat([positives.unsqueeze(1), negatives], dim=1) / temperature
    labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z_i.device)
    
    loss = F.cross_entropy(logits, labels)
    return loss


def train_self_supervised_retinal_encoder(base_dir=".", epochs=30, batch_size=16, lr=1e-3, output_pth="data/retinal_siamese_ssl.pth"):
    """
    Executes Contrastive Self-Supervised Learning on all 494 pre-op B-scan pairs.
    """
    print("\n" + "="*80)
    print("STARTING RETINAL SELF-SUPERVISED PRETRAINING (ALL 494 B-SCAN PAIRS)")
    print("="*80)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Training on device: {device}")
    
    dataset = UnlabeledOCTDataset(base_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    model = DualStreamSiameseOCT(embed_dim=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        batches = 0
        for (v1_h, v1_v), (v2_h, v2_v) in loader:
            v1_h, v1_v = v1_h.to(device), v1_v.to(device)
            v2_h, v2_v = v2_h.to(device), v2_v.to(device)
            
            optimizer.zero_grad()
            z1, _, _ = model(v1_h, v1_v)
            z2, _, _ = model(v2_h, v2_v)
            
            loss = nt_xent_contrastive_loss(z1, z2, temperature=0.1)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batches += 1
            
        scheduler.step()
        avg_loss = total_loss / max(1, batches)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] | InfoNCE Contrastive Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
            
    os.makedirs(os.path.dirname(output_pth), exist_ok=True)
    torch.save(model.state_dict(), output_pth)
    print(f"\nSaved SSL Pretrained Retinal Siamese Encoder to {output_pth}")
    return model


def extract_retinal_embeddings(df, model_pth="data/retinal_siamese_ssl.pth", output_npy="data/retinal_ssl_embeddings.npy"):
    """
    Extracts frozen domain-adapted retinal representations for benchmark cohort patients.
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = DualStreamSiameseOCT(embed_dim=64).to(device)
    
    if os.path.exists(model_pth):
        model.load_state_dict(torch.load(model_pth, map_location=device))
        print(f"Loaded pretrained weights from {model_pth}")
    else:
        print("Warning: Model checkpoint not found, using initialized encoder.")
        
    model.eval()
    augmentor = RetinalAugmentor(is_train=False)
    
    embeddings = []
    pids = []
    
    for _, row in df.iterrows():
        pid = row['id']
        h_path = row['oct_baseline_H_path']
        v_path = row['oct_baseline_V_path']
        
        if os.path.exists(h_path) and os.path.exists(v_path):
            with Image.open(h_path) as img_h, Image.open(v_path) as img_v:
                th = augmentor(img_h).unsqueeze(0).to(device)
                tv = augmentor(img_v).unsqueeze(0).to(device)
                
            with torch.no_grad():
                fused, emb_h, emb_v = model(th, tv)
                # Concatenate fused + H + V representations -> (64 + 64 + 64) = 192-dim
                full_rep = torch.cat([fused, emb_h, emb_v], dim=1).cpu().numpy().squeeze()
        else:
            full_rep = np.zeros(192, dtype=np.float32)
            
        embeddings.append(full_rep)
        pids.append(pid)
        
    emb_matrix = np.array(embeddings, dtype=np.float32)
    os.makedirs(os.path.dirname(output_npy), exist_ok=True)
    np.save(output_npy, emb_matrix)
    print(f"Extracted Retinal SSL Embeddings for {len(embeddings)} patients. Shape: {emb_matrix.shape}")
    return emb_matrix

if __name__ == "__main__":
    from src.data_preprocessing import load_and_clean_data
    # 1. Pretrain on all 494 images
    train_self_supervised_retinal_encoder(base_dir=".", epochs=25, batch_size=16)
    
    # 2. Extract embeddings for 121 benchmark patients
    df = load_and_clean_data(".")
    extract_retinal_embeddings(df)
