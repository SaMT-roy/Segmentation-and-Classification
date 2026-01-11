import numpy as np
from torchvision.transforms import v2 as T
import torchvision.io as io
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import cv2
import json
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image
import os

backbone = VisionTransformer(
    patch_size=16,
    embed_dim=192
)

ckpt = torch.load("dino_checkpoint.pth", map_location="cpu")

student_state = ckpt["student"]

# Extract only backbone weights
backbone_state = {
    k.replace("backbone.", ""): v
    for k, v in student_state.items()
    if k.startswith("backbone.")
}

# Load into backbone
backbone.load_state_dict(backbone_state, strict=True)
backbone.eval()


# 1. Setup
IMG_SIZE = 2048
transform_high_res = T.Compose([
    T.Resize(IMG_SIZE, interpolation=3),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 2. Load
img_path = "/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/locllm_author/coco_val2017/000000018519.jpg" 
raw_image = Image.open(img_path).convert('RGB')
x = transform_high_res(raw_image).unsqueeze(0)

# 3. Inference
with torch.no_grad():
    # CORRECTED LINE: No [0] index here
    patch_tokens = backbone.forward(x)[0,1:]

# 4. Grid Calc
h, w = x.shape[2] // 16, x.shape[3] // 16 

# 5. PCA
patch_tokens = patch_tokens / patch_tokens.norm(dim=-1, keepdim=True)
pca = PCA(n_components=3)
pca_features = pca.fit_transform(patch_tokens.numpy())
pca_features = (pca_features - pca_features.min(0)) / (pca_features.max(0) - pca_features.min(0))

# 6. Reshape & Upsample
feature_map = pca_features.reshape(h, w, 3)
feature_tensor = torch.from_numpy(feature_map).permute(2, 0, 1).unsqueeze(0)
upsampled_features = F.interpolate(
    feature_tensor, 
    size=(IMG_SIZE, IMG_SIZE), 
    mode='bilinear', 
    align_corners=False
).squeeze(0).permute(1, 2, 0).numpy()

# 7. Plot
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].imshow(raw_image.resize((IMG_SIZE, IMG_SIZE)))
ax[0].set_title(f"Original {IMG_SIZE}x{IMG_SIZE}")
ax[0].axis('off')

ax[1].imshow(upsampled_features)
ax[1].set_title(f"DINO Features ({h}x{w} Upsampled)")
ax[1].axis('off')

# Save only (no display)
plt.tight_layout()
plt.savefig("dino_features_new.png", dpi=300, bbox_inches="tight")
plt.close(fig)
