"""
Track 1: Pretrained Foundation / Vision Model Embedding Extractor
Extracts frozen deep visual representations from baseline H and V B-scans using a Siamese backbone.
"""

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

class SiameseVisionEncoder:
    def __init__(self, model_name="resnet50", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading pretrained vision backbone ({model_name}) on device: {self.device}...")
        
        # Load backbone with pretrained ImageNet / bio weights
        if model_name == "resnet50":
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.encoder = nn.Sequential(*list(backbone.children())[:-1]) # Global average pool output (2048-dim)
            self.embed_dim = 2048
        elif model_name == "densenet121":
            backbone = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            self.encoder = backbone.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.embed_dim = 1024
        elif model_name == "convnext_tiny":
            backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
            self.encoder = backbone.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.embed_dim = 768
            
        self.encoder.eval().to(self.device)
        self.model_name = model_name
        
        # Standard OCT visual transform (3-channel expansion, 224x224, standard normalization)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x[:3, :, :]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_image_embedding(self, img_path):
        if not os.path.exists(img_path):
            return np.zeros(self.embed_dim, dtype=np.float32)
        
        with Image.open(img_path) as img:
            img_tensor = self.transform(img.convert('RGB')).unsqueeze(0).to(self.device)
            
        with torch.no_grad():
            feat = self.encoder(img_tensor)
            if hasattr(self, 'pool'):
                feat = self.pool(feat)
            feat = torch.flatten(feat, 1).cpu().numpy().squeeze()
        return feat

    def extract_cohort_embeddings(self, df, output_npy="data/vision_embeddings_raw.npy"):
        embeddings = []
        patient_ids = []
        
        for _, row in df.iterrows():
            pid = row['id']
            h_path = row['oct_baseline_H_path']
            v_path = row['oct_baseline_V_path']
            
            emb_h = self.extract_image_embedding(h_path)
            emb_v = self.extract_image_embedding(v_path)
            
            # Siamese weight sharing: H + V mean pooling
            emb_patient = (emb_h + emb_v) / 2.0
            embeddings.append(emb_patient)
            patient_ids.append(pid)
            
        emb_matrix = np.array(embeddings, dtype=np.float32)
        os.makedirs(os.path.dirname(output_npy), exist_ok=True)
        np.save(output_npy, emb_matrix)
        np.save("data/embedding_patient_ids.npy", np.array(patient_ids))
        print(f"Extracted Siamese embeddings for {len(embeddings)} patients. Shape: {emb_matrix.shape}")
        return emb_matrix, patient_ids

if __name__ == "__main__":
    from src.data_preprocessing import load_and_clean_data
    df = load_and_clean_data(".")
    encoder = SiameseVisionEncoder("resnet50")
    encoder.extract_cohort_embeddings(df)
