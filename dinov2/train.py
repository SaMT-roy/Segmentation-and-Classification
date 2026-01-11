import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from torchvision.transforms import v2 as T
import torchvision.io as io
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import cv2
import json
import os
import torch.distributed as dist
from vit import VisionTransformer

local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

dist.init_process_group(backend="nccl")
world_size = dist.get_world_size()
rank = dist.get_rank()
    

class MaskGenerator:
    def __init__(self, input_size=16, mask_ratio=0.5):
        self.input_size = input_size # e.g. 14x14 patches
        self.num_patches = input_size * input_size
        self.mask_ratio = mask_ratio

    def __call__(self, batch_size):
        num_masked = int(self.num_patches * self.mask_ratio)

        # [1, 1, 1, ..., 1, 0, 0, ..., 0] At this point, the masked patches are not random yet — they’re just at the front.
        mask = torch.hstack([
            torch.ones(batch_size, num_masked),
            torch.zeros(batch_size, self.num_patches - num_masked)
        ])

        # positions of the masked patches are random
        mask = torch.stack([row[torch.randperm(self.num_patches)] for row in mask])
        return mask.bool() # (B, N)

class DINOHead(nn.Module):
    def __init__(
        self, 
        in_dim, 
        out_dim=65280, 
        hidden_dim=1536, 
        bottleneck_dim=384
    ):
        super().__init__()

        self.mlp = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, bottleneck_dim),
                )

        # 2. The Weight Norm Layer (The "Prototypes")
        self.last_layer = torch.nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        
        # 3. DINO specific init: Freeze weight magnitude to 1
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: (B, in_dim) - comes from the ViT backbone
        x = self.mlp(x)
        
        # Normalize features to make them lie on a hypersphere
        x = F.normalize(x, dim=-1, p=2)
        
        # Dot product with normalized prototypes
        x = self.last_layer(x)
        return x

class DINOv2Wrapper(nn.Module):
    def __init__(self, backbone, dino_head, ibot_head):
        super().__init__()
        self.backbone = backbone
        self.dino_head = dino_head # For CLS token
        self.ibot_head = ibot_head # For Patch tokens (iBOT)

    def forward(self, x, masks=None, return_patches=False):
        # 1. Pass through backbone with optional masks
        features = self.backbone(x, masks)  # Output shape: (B, N+1, D)
        
        # 2. Extract CLS and Patches
        cls_feat = features[:, 0]
        patch_feat = features[:, 1:]

        # 3. Project via Separate Heads
        cls_out = self.dino_head(cls_feat)
        
        patch_out = None
        if return_patches:
            # We flatten patches to (B*N, D) or keep (B, N, D) depending on loss implementation
            # Here we apply head to the last dimension automatically
            patch_out = self.ibot_head(patch_feat)

        return cls_out, patch_out

img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


folder = '/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/locllm_author/coco_train2017'
img_paths = []
for root, _, files in os.walk(folder):
    for f in files:
        if os.path.splitext(f)[1].lower() in img_exts:
            img_paths.append(os.path.join(root, f))
df1 = pd.DataFrame({'img_path': img_paths})
if local_rank == 0:
    print('coco train :', len(df1))

root_dir = '/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/nanovlm/data/lnqa/images'
img_paths = []
for root, _, files in os.walk(root_dir):
    for file in files:
        if os.path.splitext(file)[1].lower() in img_exts:
            img_paths.append(os.path.join(root, file))
df2 = pd.DataFrame({'img_path': img_paths})
if local_rank == 0:
    print('lnqa image :', len(df2))

folder = '/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/nanovlm/data/open_images/open_images'
img_paths = []
for root, _, files in os.walk(folder):
    for f in files:
        if os.path.splitext(f)[1].lower() in img_exts:
            img_paths.append(os.path.join(root, f))
df3 = pd.DataFrame({'img_path': img_paths})
if local_rank == 0:
    print('open images test :', len(df3))


root_dir = '/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/nanovlm/data/gqa'
img_paths = []
for root, _, files in os.walk(root_dir):
    for file in files:
        if os.path.splitext(file)[1].lower() in img_exts:
            img_paths.append(os.path.join(root, file))
df4 = pd.DataFrame({'img_path': img_paths})
if local_rank == 0:
    print('gqa image :', len(df4))

df = pd.concat([df1,df2,df3,df4]).sample(frac=1).reset_index(drop=True)
del df1,df2,df3,df4

if local_rank == 0:
    print('total images :', len(df))

class DINODataAugmentation:
    def __init__(self, global_crops_number=2, local_crops_number=8):

        # 1. Global Augmentation
        self.global_transfo = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))
        ])

        # 2. Local Augmentation
        self.local_transfo = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))
        ])

        self.global_crops_number = global_crops_number
        self.local_crops_number = local_crops_number


    def __call__(self, image):
        crops = []

        # Global crops
        for _ in range(self.global_crops_number):
            img = T.RandomResizedCrop(256, scale=(0.4, 1.0))(image)
            crops.append(self.global_transfo(img))

        # Local crops
        for _ in range(self.local_crops_number):
            img = T.RandomResizedCrop(96, scale=(0.05, 0.4))(image)
            crops.append(self.local_transfo(img))

        return crops
    
class MyDataset(Dataset):
    def __init__(self, df, max_samples=None):
        self.df = df.sample(n=max_samples) if max_samples else df
        self.aug = DINODataAugmentation()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = io.read_image(row['img_path'])

        if img.shape[0] == 1:   # If grayscale → repeat channels
            img = img.repeat(3, 1, 1)
        elif img.shape[0] >= 4:
            img = img[:3,:,:]
            
        crops = self.aug(img)
        return crops
    
def dino_collate_fn(batch):
    '''
    INPUT: 
    
    [global_1, global_2,      # 2 global crops
            local_1, ..., local_8    # 8 local crops]

    batch = [
        [c1_1, c1_2, ..., c1_10],  # sample 1
        [c2_1, c2_2, ..., c2_10],  # sample 2
        ...
        [c32_1, c32_2, ..., c32_10]  # sample 32
        ]

    batch = [32 samples][10 crops each][3 × H × W]

    OUTPUT :

    transposed = [
    (c1_1, c2_1, ..., c32_1),   # crop index 0 (global)
    (c1_2, c2_2, ..., c32_2),   # crop index 1 (global)
    (c1_3, c2_3, ..., c32_3),   # local
    ...
    ]

    tensor_crops = [torch.stack(crops) for crops in transposed]

    tensor_crops = [
    Tensor(32, 3, 256, 256),  # global crop 1
    Tensor(32, 3, 256, 256),  # global crop 2
    Tensor(32, 3, 96, 96),    # local crop 1
    ...
    ]

    '''
    transposed = list(zip(*batch))
    tensor_crops = [torch.stack(crops) for crops in transposed]
    return tensor_crops

dataset = MyDataset(df)

train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    sampler=train_sampler,
    collate_fn=dino_collate_fn,
    pin_memory=True,
    drop_last=True,
    shuffle=False
)

class DINOLoss(nn.Module):
    def __init__(self, out_dim, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output, teacher_output, epoch):
        """
        student_output: [n_crops * B, out_dim] 
        teacher_output: [2 * B, out_dim]
        """
        # Softmax and Sharpening
        student_out = student_output / self.student_temp
        
        n_crops = len(student_output) // (len(teacher_output) // 2)
        student_out = student_out.chunk(n_crops) 

        # Teacher centering and sharpening
        teacher_out = F.softmax((teacher_output - self.center) / self.teacher_temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0

        # 2. Cross-Entropy
        for i_t, t_out in enumerate(teacher_out):
            for i_s, s_out in enumerate(student_out):
                if i_t == i_s:
                    continue
                
                # Now t_out is [B, dim] and s_out is [B, dim]
                loss = torch.sum(-t_out * F.log_softmax(s_out, dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1

        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output centering.
        """
        # Calculate mean of current batch
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        
        # --- FIX: Synchronize across GPUs ---
        if dist.is_initialized():
            dist.all_reduce(batch_center) # Sums "batch_center" across all GPUs
            batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
        else:
            batch_center = batch_center / len(teacher_output)
        
        # EMA update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

class iBOTPatchLoss(nn.Module):
    def __init__(self, out_dim, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_patches, teacher_patches):
        """
        Inputs are flattened tensors containing ONLY the masked tokens.
        student_patches: (N_masked_total, out_dim)
        teacher_patches: (N_masked_total, out_dim)
        """
        # 1. Student: log-softmax with temperature
        s_out = F.log_softmax(student_patches / self.student_temp, dim=-1)

        # 2. Teacher: centering + softmax with temperature (detached)
        t_out = F.softmax((teacher_patches.detach() - self.center) / self.teacher_temp, dim=-1)

        # 3. Cross-Entropy Loss
        # KL divergence = sum(-target * log_pred)
        loss = torch.sum(-t_out * s_out, dim=-1).mean()

        # 4. Update Center (EMA)
        self.update_center(teacher_patches)
        
        return loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output centering.
        """
        # Detach explicitly (even though we're in no_grad context)
        teacher_output = teacher_output.detach()
        
        # Calculate mean of current batch
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        
        # Synchronize across GPUs and normalize
        if dist.is_initialized():
            dist.all_reduce(batch_center)
            world_size = dist.get_world_size()
            batch_center = batch_center / (len(teacher_output) * world_size)
        else:
            batch_center = batch_center / len(teacher_output)

        # EMA update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

class KoLeoLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, student_output, eps=1e-8):
        """
        student_output: (B, D)
        """
        # 0. Normalize: KoLeo requires features on the hypersphere (Cosine Sim equivalent)
        # Even if your head normalizes, re-normalizing here is safe and ensures numerical stability.
        student_output = F.normalize(student_output, dim=-1, p=2)

        # 1. Pairwise distance
        pdist = torch.cdist(student_output, student_output, p=2)
        
        # 2. Mask Diagonal (distance to self is 0)
        # --- FIX: Use masked_fill to avoid in-place modification error ---
        mask = torch.eye(pdist.shape[0], device=pdist.device, dtype=torch.bool)
        pdist = pdist.masked_fill(mask, float('inf'))
        
        # 3. Find Nearest Neighbor per row
        min_dist, _ = torch.min(pdist, dim=1)
        
        # 4. Log Loss
        loss = -torch.log(min_dist + eps).mean()
        return loss

# 1. Backbones
student_backbone = VisionTransformer(patch_size=16, embed_dim=192)
teacher_backbone = VisionTransformer(patch_size=16, embed_dim=192)

# 2. Heads (UNTIED: Separate heads for DINO and iBOT)
student_dino_head = DINOHead(in_dim=192, out_dim=24000)
student_ibot_head = DINOHead(in_dim=192, out_dim=24000) # New iBOT Head

teacher_dino_head = DINOHead(in_dim=192, out_dim=24000)
teacher_ibot_head = DINOHead(in_dim=192, out_dim=24000) # New iBOT Head

# 3. Wrap (Pass both heads)
student = DINOv2Wrapper(student_backbone, student_dino_head, student_ibot_head)
teacher = DINOv2Wrapper(teacher_backbone, teacher_dino_head, teacher_ibot_head)

# 4. Sync weights & Freeze Teacher
teacher.load_state_dict(student.state_dict())
for param in teacher.parameters():
    param.requires_grad = False

student = student.to(device)
teacher = teacher.to(device)

student = torch.nn.parallel.DistributedDataParallel(
    student,
    device_ids=[local_rank],
    output_device=local_rank,
)

print("Models ready for DINO training.")

# -----------------------------------------------------------
# TRAINING LOOP
# -----------------------------------------------------------
epochs = 300
print(f"Starting training on {device}...")

dino_loss_fn = DINOLoss(out_dim=24000).to(device) # Reuse your existing class
ibot_loss_fn = iBOTPatchLoss(out_dim=24000).to(device) # Can reuse same logic for patches
koleo_loss_fn = KoLeoLoss().to(device)

mask_gen = MaskGenerator(input_size=16, mask_ratio=0.5)

optimizer = torch.optim.Adam(student.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
momentum_schedule = np.linspace(0.996, 1.0, epochs) 

for epoch in range(epochs):
    train_sampler.set_epoch(epoch)
    student.train()
    teacher.eval()
    
    total_loss = 0
    m = momentum_schedule[epoch]
    
    # 1. Wrap dataloader with tqdm
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", leave=True, disable=(local_rank != 0))

    for i, images in enumerate(pbar):
        global_imgs = torch.cat(images[:2]).to(device)
        local_imgs  = torch.cat(images[2:]).to(device)

        with torch.no_grad():
            # Teacher sees everything clean
            t_cls, t_patch = teacher(global_imgs, masks=None, return_patches=True)

        masks = mask_gen(batch_size=global_imgs.shape[0]).to(device)

        # ----------------------------------------------------
        # STUDENT FORWARD
        # ----------------------------------------------------
        # A. Global Crops (Masked) -> Used for DINO + iBOT
        s_global_cls, s_global_patch = student(global_imgs, masks=masks, return_patches=True)
        
        # B. Local Crops (Unmasked) -> Used for DINO Only
        s_local_cls, _ = student(local_imgs, masks=None, return_patches=False)
        
        # ----------------------------------------------------
        # CALCULATE LOSSES
        # ----------------------------------------------------
        
        # 1. DINO Loss (Global CLS)
        # Compare Student Global CLS + Local CLS vs Teacher Global CLS
        student_cls_all = torch.cat([s_global_cls, s_local_cls], dim=0)
        loss_dino = dino_loss_fn(student_cls_all, t_cls, epoch)

        # 2. iBOT Loss (Patch Level)
        # Only compare masked patches between Student Global and Teacher Global
        # We need to flatten to (Batch * N, Dim) to use the Loss function
        
        # Filter only masked patches
        # masks is (B, N), t_patch is (B, N, D)
        mask_bool_flat = masks.flatten().bool()
        
        t_patch_flat = t_patch.flatten(0, 1) # (B*N, D)
        s_patch_flat = s_global_patch.flatten(0, 1) # (B*N, D)
        
        # Select only the masked tokens
        t_patch_masked = t_patch_flat[mask_bool_flat]
        s_patch_masked = s_patch_flat[mask_bool_flat]

        # Calculate Cross Entropy on patches
        # NOTE: Pass dummy teacher_output to update center only once, or handle manually
        # Here we simplify and just calculate loss
        loss_ibot = ibot_loss_fn(s_patch_masked, t_patch_masked)

        # 3. KoLeo Loss
        # Applied to Student Global CLS features
        loss_koleo = koleo_loss_fn(s_global_cls)

        # ----------------------------------------------------
        # FINAL SUM
        # ----------------------------------------------------
        # Coefficients from paper (approximate)
        loss = loss_dino + loss_ibot + (loss_koleo * 0.1)
        
        optimizer.zero_grad()
        loss.backward()
        param_norms = torch.nn.utils.clip_grad_norm_(student.parameters(), 3.0)
        optimizer.step()
        
        # Update weights (EMA)
        with torch.no_grad():
            # Uses student.module because student is DDP wrapped
            # teacher is NOT DDP wrapped, so we access it directly
            for param_q, param_k in zip(student.module.parameters(), teacher.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        total_loss += loss.item()
        
        if local_rank == 0:
            # 2. Update progress bar with current loss
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        
    avg_loss = total_loss / len(dataloader)
    if local_rank == 0:
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

    # 1. Synchronization: Wait for ALL GPUs to reach this line
    dist.barrier()

    # 2. Rank Check: Only GPU 0 is allowed to write to disk
    if local_rank == 0:

        torch.save({
            'student': student.module.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, f"dino_checkpoint.pth")

        if epoch % 20 == 0:

            torch.save({
                'student': student.module.state_dict(),
                'teacher': teacher.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, f"dino_checkpoint_epoch{epoch}.pth")


# 4. Cleanup
dist.destroy_process_group()
