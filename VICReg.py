#!/usr/bin/env python
# coding: utf-8

# In[3]:


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

pl.seed_everything(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# In[3]:
DATASET_PATH   = "./ViTacTip_Dataset_Final"
CHECKPOINT_DIR = "./vicreg_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Training knobs (VICReg)
IMG_SIZE      = 224
BATCH_SIZE    = 256
MAX_EPOCHS    = 100
WARMUP_EPOCHS = 10
WEIGHT_DECAY  = 1e-6
BASE_MODEL    = "resnet18"
REP_DIM       = None
EMB_DIM       = 8192

# VICReg loss weights
LAMBDA = 25.0  # invariance (MSE)
MU     = 25.0  # variance hinge
NU     = 1.0   # covariance off-diagonal
EPS    = 1e-4

# LR schedule
BASE_LR = 0.2 * (BATCH_SIZE / 256.0)
LR_MIN_FACTOR = 0.01

# Normalization
IMAGENET_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STD  = [0.5, 0.5, 0.5]

# In[4]:
def _list_images(root, exts=(".png",".jpg",".jpeg",".bmp",".tif",".tiff")):
    out = []
    for ext in exts:
        out += glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True)
    return sorted(out)

all_imgs = _list_images(DATASET_PATH)
assert len(all_imgs) > 0, f"No images found in {DATASET_PATH}"
sample = random.sample(all_imgs, min(5, len(all_imgs)))

fig, axs = plt.subplots(1, len(sample), figsize=(15,3))
for i,p in enumerate(sample):
    axs[i].imshow(Image.open(p).convert("RGB"))
    axs[i].set_title(os.path.relpath(p, DATASET_PATH), fontsize=8)
    axs[i].axis("off")
plt.suptitle("Random sample of training images", y=1.02)
plt.tight_layout(); plt.show()

# In[5]:
class TwoCrops:
    def __init__(self, T): self.T = T
    def __call__(self, x): return self.T(x), self.T(x)

def vicreg_transform(img_size=224):
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0), ratio=(0.75, 1.3333)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
        transforms.RandomSolarize(threshold=0.5, p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

T_vicreg = TwoCrops(vicreg_transform(IMG_SIZE))

EVAL_T = transforms.Compose([
    transforms.Resize(int(IMG_SIZE*1.14)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# In[6]:
class FlatImageDataset(data.Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        x = Image.open(self.image_paths[idx]).convert("RGB")
        x = self.transform(x) if self.transform else (x, x)
        return x, 0

# 100% train — no val split
train_ds = FlatImageDataset(all_imgs, transform=T_vicreg)
train_loader = data.DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True, pin_memory=True
)
print("Train images:", len(train_ds))

# displaying augmentations
fig, axs = plt.subplots(2, 5, figsize=(15,4))
mean = torch.tensor(IMAGENET_MEAN).view(3,1,1)
std  = torch.tensor(IMAGENET_STD).view(3,1,1)
for i in range(min(5, len(train_ds))):
    (v1, v2), _ = train_ds[i]
    im1 = (v1*std + mean).clamp(0,1).permute(1,2,0).numpy()
    im2 = (v2*std + mean).clamp(0,1).permute(1,2,0).numpy()
    axs[0,i].imshow(im1); axs[0,i].axis('off'); axs[0,i].set_title(f"view A #{i+1}")
    axs[1,i].imshow(im2); axs[1,i].axis('off'); axs[1,i].set_title(f"view B #{i+1}")
plt.suptitle("VICReg symmetric views")
plt.tight_layout(); plt.show()

# In[7]:
class ExpanderMLP(nn.Module):
    def __init__(self, in_dim, hidden=EMB_DIM, out_dim=EMB_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden, bias=True),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden, bias=True),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim, bias=True),
        )
    def forward(self, x): return self.net(x)

class VICRegBackbone(nn.Module):
    def __init__(self, base_model=BASE_MODEL):
        super().__init__()
        base = getattr(torchvision.models, base_model)(weights=None)
        feat_dim = base.fc.in_features
        base.fc = nn.Identity()
        self.encoder = base
        self.expander = ExpanderMLP(in_dim=feat_dim, hidden=EMB_DIM, out_dim=EMB_DIM)
        self.feat_dim = feat_dim
        self.emb_dim = EMB_DIM
    def forward(self, x):
        y = self.encoder(x)
        if y.ndim > 2: y = y.flatten(1)
        z = self.expander(y)
        return y, z

tmp = VICRegBackbone(BASE_MODEL); REP_DIM = tmp.feat_dim; del tmp

# In[8]:
class LARS(optim.Optimizer):
    def __init__(self, params, lr=0.1, momentum=0.9, weight_decay=1e-6, eta=0.001, eps=1e-8):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, eta=eta, eps=eps)
        super().__init__(params, defaults)
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group['lr']; m = group['momentum']; wd = group['weight_decay']; eta = group['eta']; eps = group['eps']
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
                state = self.state.setdefault(p, {})
                mu = state.get('mu')
                if mu is None: mu = state.setdefault('mu', torch.zeros_like(p))
                mu.mul_(m).add_(d_p)
                p.add_(mu, alpha=-lr)
        return loss

# In[9]:
def off_diagonal(x: torch.Tensor):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n-1, n+1)[:,1:].flatten()

class VICReg_PL(pl.LightningModule):
    def __init__(self,
                 base_model=BASE_MODEL,
                 base_lr=BASE_LR,
                 weight_decay=WEIGHT_DECAY,
                 max_epochs=MAX_EPOCHS,
                 warmup_epochs=WARMUP_EPOCHS,
                 lambda_inv=LAMBDA, mu_var=MU, nu_cov=NU, eps=EPS):
        super().__init__()
        self.save_hyperparameters()
        self.net = VICRegBackbone(base_model)
        self.train_losses = []
        self._cur_loss = []

    def vicreg_loss(self, batch):
        (x1, x2), _ = batch
        _, z1 = self.net(x1)
        _, z2 = self.net(x2)

        sim_loss = F.mse_loss(z1, z2)

        def _var(z):
            std = torch.sqrt(z.var(dim=0, unbiased=False) + self.hparams.eps)
            return torch.mean(F.relu(1.0 - std))
        std_loss = _var(z1) + _var(z2)

        def _cov(z):
            z = z - z.mean(dim=0)
            cov = (z.T @ z) / (z.size(0) - 1)
            return (off_diagonal(cov).pow(2).sum()) / z.size(1)
        cov_loss = _cov(z1) + _cov(z2)

        loss = (self.hparams.lambda_inv * sim_loss
                + self.hparams.mu_var * std_loss
                + self.hparams.nu_cov * cov_loss)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.vicreg_loss(batch)
        self._cur_loss.append(loss.item())
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        if self._cur_loss:
            self.train_losses.append(float(np.mean(self._cur_loss)))
        self._cur_loss = []

    def configure_optimizers(self):
        opt = LARS(self.parameters(), lr=self.hparams.base_lr, momentum=0.9,
                   weight_decay=self.hparams.weight_decay, eta=0.001)
        def lr_lambda(epoch):
            if epoch < self.hparams.warmup_epochs:
                return float(epoch + 1) / float(max(1, self.hparams.warmup_epochs))
            T = self.hparams.max_epochs - self.hparams.warmup_epochs
            t = min(max(epoch - self.hparams.warmup_epochs, 0), max(1, T)) / float(max(1, T))
            cos = 0.5 * (1.0 + math.cos(math.pi * t))
            return LR_MIN_FACTOR + (1.0 - LR_MIN_FACTOR) * cos
        sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        return [opt], [sched]

# In[10]:
checkpoint_cb = ModelCheckpoint(
    dirpath=CHECKPOINT_DIR,
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

# In[11]:
vicreg = VICReg_PL()
trainer.fit(vicreg, train_loader)


# In[ ]:




