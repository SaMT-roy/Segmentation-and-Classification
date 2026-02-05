import torch
import timm
import torchvision.transforms as T
import torch.nn as nn
import cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

class Config:
    model_name: str = 'mobilenetv3_small_050.lamb_in1k'
    hidden_dim: int = 128
    classes: int = 2
    global_img_size: int = 224

cfg = Config()
device = 'cpu'

class CNNAttentionPooling(nn.Module):
    def __init__(self, C, hidden, classes):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(C, hidden, 1),
            nn.ReLU(),
            nn.Conv1d(hidden, 1, 1)
        )
        self.fc = nn.Linear(C, classes)

    def forward(self, x):  
        scores = self.attn(x)        # [B, 1, N]
        weights = F.softmax(scores, dim=-1)
        pooled = (weights * x).sum(dim=-1)  # [B, C]
        return self.fc(pooled)
    
class classification_inference_model(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.img_emb_model = timm.create_model(cfg.model_name, pretrained=False, features_only=True)
        self.data_config = timm.data.resolve_model_data_config(self.img_emb_model)
        self.num_channels = self.img_emb_model.feature_info[-1]["num_chs"]

        self.dense = CNNAttentionPooling(C=self.num_channels, hidden=cfg.hidden_dim, classes=cfg.classes)

    def forward(self, img_tensors):

        features = self.img_emb_model(img_tensors)[-1] 
        B, C, H, W = features.shape
        features = features.reshape(B, C, H * W) 
        logits = self.dense(features)
        return logits
    
model_inf = classification_inference_model(cfg).to(device)
model_inf.load_state_dict(torch.load("model_state.pt", map_location=device))
model_inf.eval()

dummy = torch.randn(1, 3, cfg.global_img_size, cfg.global_img_size).to(device)
torch.onnx.export(
    model_inf,
    dummy,
    "latte_art_classifier.onnx",
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["image"],
    output_names=["logits"],
    dynamic_axes={
        "image": {0: "batch"},
        "logits": {0: "batch"}
    }
)
