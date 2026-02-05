import os
from PIL import Image
import torch
import torch.nn as nn
import timm
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
from tqdm import tqdm
import torchvision.transforms.functional as TF

class Config:
    model_name: str = 'mobilenetv3_small_050.lamb_in1k'
    model_path: str = "mobilenetv3_small_050_pytorch_model.bin"
    data_path: str = "/Users/saptarshimallikthakur/Pictures/VLM/sharan_startup/data"
    hidden_dim: int = 128
    classes: int = 2
    batch_size: int = 8
    global_img_size: int = 224
    lr : float = 1e-4
  

cfg = Config()
device='mps'

class AugmentedImageFolder(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = []

        self.classes = sorted(
            [d for d in os.listdir(root_dir)
             if os.path.isdir(os.path.join(root_dir, d))]
        )
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # --- Load images per class ---
        class_images = {}
        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            files = [
                os.path.join(cls_path, f)
                for f in os.listdir(cls_path)
                if f.lower().endswith(("jpg", "jpeg", "png", "bmp", "webp"))
            ]
            class_images[cls] = files

        # --- Balance dataset ---
        min_count = min(len(files) for files in class_images.values())
        print(f"Balancing dataset: using {min_count} images per class.")

        for cls, files in class_images.items():
            if len(files) > min_count:
                files = random.sample(files, min_count)
            self.samples.extend([(f, self.class_to_idx[cls]) for f in files])

        self.base_transform = transforms.Compose([
            transforms.Resize((cfg.global_img_size, cfg.global_img_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples) * 8

    def __getitem__(self, idx):
        real_index = idx // 8
        aug_index  = idx % 8

        img_path, target = self.samples[real_index]
        img = Image.open(img_path).convert("RGB")

        # --- Augment logic ---
        if aug_index == 0:
            pass  # original
        elif aug_index == 1:
            img = TF.hflip(img)
        elif aug_index == 2:
            img = TF.vflip(img)
        elif aug_index == 3:
            img = TF.rotate(img, 90)
        elif aug_index == 4:
            img = TF.rotate(img, 180)
        elif aug_index == 5:
            img = TF.rotate(img, 270)
        elif aug_index == 6:
            img = TF.rotate(img, 30)
        elif aug_index == 7:
            img = TF.rotate(img, 60)

        img = self.base_transform(img)
        return img, target


def get_loader(root, batch_size=8):
    dataset = AugmentedImageFolder(root)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, dataset.classes

train_loader, class_names = get_loader(
    cfg.data_path,
    batch_size=cfg.batch_size
)


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

class classification_model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.img_emb_model = timm.create_model(cfg.model_name, pretrained=False, features_only=True)
        checkpoint = torch.load(cfg.model_path, map_location='cpu')
        self.data_config = timm.data.resolve_model_data_config(self.img_emb_model)
        self.num_channels = self.img_emb_model.feature_info[-1]["num_chs"]
        self.input_shape  = self.img_emb_model.default_cfg['input_size'][-1]
        self.feature_size = (self.input_shape // self.img_emb_model.feature_info[-1]["reduction"])**2

        # --- Handle various checkpoint structures ---
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                # assume checkpoint itself is the state_dict
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # --- Load state dict ---
        missing, unexpected = self.img_emb_model.load_state_dict(state_dict, strict=False)

        print("✅ Missing keys:", missing)
        print("⚠️ Unexpected keys:", unexpected)

        self.dense = CNNAttentionPooling(C=self.num_channels, hidden=cfg.hidden_dim, classes=cfg.classes)

        for param in self.img_emb_model.parameters():
            param.requires_grad = False

        self.transform = T.Compose([
            T.Normalize(mean=self.data_config['mean'], std=self.data_config['std'])
        ])

    def forward(self, imgs):

        if isinstance(imgs, list):
            img_tensors = torch.stack([self.transform(im) for im in imgs])
        else:
            img_tensors = self.transform(imgs)

        with torch.no_grad():
            features = self.img_emb_model(img_tensors)[-1] 
        B, C, H, W = features.shape
        features = features.reshape(B, C, H * W)
        logits = self.dense(features)
        return logits
    
model = classification_model(cfg).to(device)

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total params     : {total:,}")
print(f"Trainable params : {trainable:,}")
print(f"Frozen params    : {total - trainable:,}")


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.dense.parameters(), lr=cfg.lr)

def train(model, loader, optimizer, criterion, device, epochs=5):

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(loader, total=len(loader), desc=f"Epoch {epoch+1}/{epochs}")

        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            # correct: batch-wise avg loss
            avg_batch_loss = total_loss / (pbar.n + 1)

            pbar.set_postfix({
                "loss": f"{avg_batch_loss:.4f}",
                "acc": f"{100 * correct / total:.2f}%"
            })

train(model, train_loader, optimizer, criterion, device, epochs=20)
torch.save(model.state_dict(), "model_state.pt")
