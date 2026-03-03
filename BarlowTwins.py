#!/usr/bin/env python
# coding: utf-8


# In[1]:
get_ipython().system('pip install -q pytorch-lightning==2.2.5 torchvision')

# In[2]:
import os, glob, math, random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torchvision
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# In[3]:
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# In[4]:
DATASET_PATH = "./ViTacTip_Dataset_Final"
CHECKPOINT_PATH = "./barlowtwins_checkpoints"
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

pl.seed_everything(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# In[5]:
# visualisation
all_images = [f for f in os.listdir(DATASET_PATH)
              if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif','.tiff'))]
if len(all_images) >= 5:
    image_files = random.sample(all_images, 5)
    rel_paths = [os.path.join(DATASET_PATH, f) for f in image_files]
else:
    def _list_images(root):
        out = []
        for ext in ('.png','.jpg','.jpeg','.bmp','.tif','.tiff'):
            out += glob.glob(os.path.join(root, '**', f'*{ext}'), recursive=True)
        return out
    rel_paths = _list_images(DATASET_PATH)[:5]

fig, axs = plt.subplots(1, len(rel_paths), figsize=(15, 3))
for i,p in enumerate(rel_paths):
    img = Image.open(p).convert("RGB")
    axs[i].imshow(img); axs[i].axis('off'); axs[i].set_title(os.path.relpath(p, DATASET_PATH), fontsize=8)
plt.suptitle("Random Sample of ViTacTip Images", fontsize=14)
plt.tight_layout(); plt.show()

# In[6]:
# parameters
IMG_SIZE      = 224
BATCH_SIZE    = 256
MAX_EPOCHS    = 100
WARMUP_EPOCHS = 10
WEIGHT_DECAY  = 1.5e-6
LAMBDA_OFFDIAG= 5e-3

BASE_MODEL    = "resnet18"
PROJ_DIM      = 8192

BASE_LR_WEIGHTS = 0.2   * (BATCH_SIZE/256.0)
BASE_LR_BIASBN  = 0.0048* (BATCH_SIZE/256.0)
COSINE_MIN_FACTOR = 1.0/1000.0

IMAGENET_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STD  = [0.5, 0.5, 0.5]

# In[7]:
class TwoCropsBT:
    def __init__(self, T1: transforms.Compose, T2: transforms.Compose):
        self.T1, self.T2 = T1, T2
    def __call__(self, x: Image.Image):
        return self.T1(x), self.T2(x)

def barlow_transforms(img_size=224):
    color_jitter = transforms.ColorJitter(0.4,0.4,0.4,0.1)
    common = [
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
    ]
    T1 = transforms.Compose([
        *common,
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    T2 = transforms.Compose([
        *common,
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.1),
        transforms.RandomSolarize(threshold=0.5, p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    EVAL_T = transforms.Compose([
        transforms.Resize(int(img_size*1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return T1, T2, EVAL_T

T1, T2, EVAL_T = barlow_transforms(IMG_SIZE)
TWOCROP = TwoCropsBT(T1, T2)

# In[8]:
def list_images(root, exts=(".png",".jpg",".jpeg",".bmp",".tif",".tiff")):
    paths = []
    for ext in exts:
        paths += glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True)
    return sorted(paths)

class FlatImageDataset(data.Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            v1, v2 = self.transform(img)
            return (v1, v2), 0
        return (img, img), 0

# In[9]:
all_paths = list_images(DATASET_PATH)
assert len(all_paths) > 0, f"No images found in {DATASET_PATH}"
train_ds = FlatImageDataset(all_paths, transform=TWOCROP)
train_loader = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, drop_last=True)

print("Train images:", len(train_ds))

# In[10]:
fig, axs = plt.subplots(2, 5, figsize=(15, 4))
mean = torch.tensor(IMAGENET_MEAN).view(3,1,1)
std  = torch.tensor(IMAGENET_STD).view(3,1,1)
for i in range(min(5, len(train_ds))):
    (v1, v2), _ = train_ds[i]
    im1 = (v1*std + mean).clamp(0,1).permute(1,2,0).numpy()
    im2 = (v2*std + mean).clamp(0,1).permute(1,2,0).numpy()
    axs[0, i].imshow(im1); axs[0, i].axis('off'); axs[0, i].set_title(f"View A {i+1}")
    axs[1, i].imshow(im2); axs[1, i].axis('off'); axs[1, i].set_title(f"View B {i+1}")
plt.suptitle("Barlow Twins Views (BYOL-style aug)")
plt.tight_layout(); plt.show()


# In[11]:
class Projector8192x3(nn.Module):
    def __init__(self, in_dim, dim=PROJ_DIM):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, dim, bias=True)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim, bias=True)
        self.bn2 = nn.BatchNorm1d(dim)
        self.fc3 = nn.Linear(dim, dim, bias=True)  # final embedding
    def forward(self, x):
        x = self.bn1(self.fc1(x)); x = F.relu(x, inplace=True)
        x = self.bn2(self.fc2(x)); x = F.relu(x, inplace=True)
        x = self.fc3(x)
        return x

class BTBackbone(nn.Module):
    def __init__(self, base_model=BASE_MODEL):
        super().__init__()
        base = getattr(torchvision.models, base_model)(weights=None)
        feat_dim = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.projector = Projector8192x3(feat_dim, PROJ_DIM)
    def forward(self, x):
        h = self.backbone(x)
        z = self.projector(h)
        return h, z

# In[12]:
class LARS(optim.Optimizer):
    def __init__(self, params, lr=0.1, momentum=0.9, weight_decay=1.5e-6, eta=0.001, eps=1e-8):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, eta=eta, eps=eps)
        super().__init__(params, defaults)
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        for group in self.param_groups:
            lr = group['lr']; momentum = group['momentum']
            wd = group['weight_decay']; eta = group['eta']; eps = group['eps']
            for p in group['params']:
                if p.grad is None: continue
                d_p = p.grad
                is_1d = (p.ndim == 1)
                if not is_1d and wd != 0:
                    d_p = d_p.add(p, alpha=wd)
                if not is_1d:
                    w_norm = torch.norm(p)
                    g_norm = torch.norm(d_p)
                    trust = eta * w_norm / (g_norm + eps) if (w_norm > 0 and g_norm > 0) else 1.0
                    d_p = d_p.mul(trust)
                state = self.state[p]
                if 'mu' not in state:
                    state['mu'] = torch.zeros_like(p)
                mu = state['mu']
                mu.mul_(momentum).add_(d_p)
                p.add_(mu, alpha=-lr)
        return loss

# In[13]:
def off_diagonal(x: torch.Tensor):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n-1, n+1)[:,1:].flatten()

# In[14]:
class BarlowTwinsPL(pl.LightningModule):
    def __init__(self,
                 base_model=BASE_MODEL,
                 proj_dim=PROJ_DIM,
                 lambda_offdiag=LAMBDA_OFFDIAG,
                 base_lr_w=BASE_LR_WEIGHTS,
                 base_lr_b=BASE_LR_BIASBN,
                 weight_decay=WEIGHT_DECAY,
                 max_epochs=MAX_EPOCHS,
                 warmup_epochs=WARMUP_EPOCHS,
                 cosine_min_factor=COSINE_MIN_FACTOR):
        super().__init__()
        self.save_hyperparameters()
        self.net = BTBackbone(base_model)
        self.train_losses = []
        self._cur_loss = []

    def _bt_loss(self, batch):
        (x1, x2), _ = batch
        _, z1 = self.net(x1)
        _, z2 = self.net(x2)

        # Normalize per-feature across batch
        z1 = (z1 - z1.mean(0)) / (z1.std(0) + 1e-12)
        z2 = (z2 - z2.mean(0)) / (z2.std(0) + 1e-12)

        N, D = z1.shape
        c = (z1.T @ z2) / N  # D x D

        on_diag  = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.hparams.lambda_offdiag * off_diag
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._bt_loss(batch)
        self._cur_loss.append(loss.item())
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        if self._cur_loss:
            self.train_losses.append(float(np.mean(self._cur_loss)))
        self._cur_loss = []

    def configure_optimizers(self):
        # Separate params: weights vs bias/BN
        params_weight, params_biasbn = [], []
        for p in self.parameters():
            if not p.requires_grad: continue
            (params_biasbn if p.ndim == 1 else params_weight).append(p)

        optim_groups = [
            {"params": params_weight, "lr": self.hparams.base_lr_w, "weight_decay": self.hparams.weight_decay},
            {"params": params_biasbn,  "lr": self.hparams.base_lr_b, "weight_decay": 0.0},
        ]
        opt = LARS(optim_groups, momentum=0.9)

        def lr_lambda(epoch):
            if epoch < self.hparams.warmup_epochs:
                return float(epoch + 1) / float(max(1, self.hparams.warmup_epochs))
            t = (epoch - self.hparams.warmup_epochs) / float(max(1, self.hparams.max_epochs - self.hparams.warmup_epochs))
            t = min(1.0, max(0.0, t))
            return self.hparams.cosine_min_factor + (1.0 - self.hparams.cosine_min_factor) * 0.5 * (1.0 + math.cos(math.pi * t))

        sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        return [opt], [sched]

# In[15]:
checkpoint_cb = ModelCheckpoint(
    dirpath=CHECKPOINT_PATH,
    save_top_k=1,
    monitor="train_loss",
    mode="min"
)

trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    logger=False,
    callbacks=[checkpoint_cb],
    enable_progress_bar=True,
    log_every_n_steps=25
)

# In[16]:
model = BarlowTwinsPL()
trainer.fit(model, train_loader)





