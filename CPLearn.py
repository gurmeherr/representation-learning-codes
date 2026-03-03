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

# Use TF32 tensor cores on A100
torch.set_float32_matmul_precision('high')

pl.seed_everything(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# In[3]:
DATASET_PATH = "./ViTacTip_Dataset_Final"
CHECKPOINT_PATH = "./cplearn_checkpoints"
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# In[4]:
def _list_images(root):
    out = []
    for ext in ('.png','.jpg','.jpeg','.bmp','.tif','.tiff'):
        out += glob.glob(os.path.join(root, '**', f'*{ext}'), recursive=True)
    return out

paths = _list_images(DATASET_PATH)
assert len(paths) > 0, f"No images found in {DATASET_PATH}"
sample_paths = random.sample(paths, min(5, len(paths)))

fig, axs = plt.subplots(1, len(sample_paths), figsize=(15,3))
for i,p in enumerate(sample_paths):
    img = Image.open(p).convert("RGB")
    axs[i].imshow(img)
    axs[i].axis('off')
    axs[i].set_title(os.path.relpath(p, DATASET_PATH), fontsize=8)
plt.suptitle("Random sample of images")
plt.tight_layout()
plt.show()

# In[5]:
# Training knobs 
IMG_SIZE = 224
BATCH_SIZE = 256
MAX_EPOCHS = 100
WARMUP_EPOCHS = 10
WEIGHT_DECAY = 1.5e-6
BASE_MODEL = "resnet18"
PROJ_OUT_DIM = 256
DICT_SIZE = 8192
BETA = 0.5
EPS = 1e-8
USE_REVERSE_KL = True
USE_TANH = False

BASE_LR = 0.2 * (BATCH_SIZE / 256.0)

IMAGENET_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STD  = [0.5, 0.5, 0.5]

# In[6]:
# Two symmetric random views
class TwoCropsSame:
    def __init__(self, T: transforms.Compose):
        self.T = T
    def __call__(self, x: Image.Image):
        return self.T(x), self.T(x)

def cplearn_transforms(img_size=224):
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    T = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    EVAL_T = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return T, EVAL_T

T_train, EVAL_T = cplearn_transforms(IMG_SIZE)
TWOCROP = TwoCropsSame(T_train)

# In[7]:
# Dataset helpers
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

# In[8]:
# Build dataset & loader
all_paths = list_images(DATASET_PATH)
assert len(all_paths) > 0, f"No images found in {DATASET_PATH}"

train_ds = FlatImageDataset(all_paths, transform=TWOCROP)
train_loader = data.DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    drop_last=True
)

print("Train samples:", len(train_ds))

# In[9]:
fig, axs = plt.subplots(2, 5, figsize=(15, 4))
mean = torch.tensor(IMAGENET_MEAN).view(3,1,1)
std  = torch.tensor(IMAGENET_STD).view(3,1,1)

for i in range(min(5, len(train_ds))):
    (v1, v2), _ = train_ds[i]
    im1 = (v1 * std + mean).clamp(0,1).permute(1,2,0).numpy()
    im2 = (v2 * std + mean).clamp(0,1).permute(1,2,0).numpy()
    axs[0, i].imshow(im1)
    axs[0, i].axis('off')
    axs[0, i].set_title(f"View A {i+1}")
    axs[1, i].imshow(im2)
    axs[1, i].axis('off')
    axs[1, i].set_title(f"View B {i+1}")

plt.suptitle("CP:Learn Augmented Views")
plt.tight_layout()
plt.show()

# In[10]:
# Utilities
def l2n(x, dim=1, eps=1e-12):
    return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)

def make_rademacher_codes(f, c, device):
    codes = torch.empty(f, c, device=device)
    codes.bernoulli_(0.5).mul_(2).sub_(1)
    return codes

# In[11]:
# Backbone + projector
class Encoder(nn.Module):
    def __init__(self, base_model=BASE_MODEL):
        super().__init__()
        base = getattr(torchvision.models, base_model)(weights=None)
        feat_dim = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.feat_dim = feat_dim

    def forward(self, x):
        return self.backbone(x)

class CPLearnProjector(nn.Module):
    """
    H = sqrt(f/n) * norm(BN(Linear(Z)))
    scores = H @ W / tau,   P = softmax(scores)
    """
    def __init__(self, in_dim, f=PROJ_OUT_DIM, c=DICT_SIZE, use_tanh=USE_TANH, eps=EPS):
        super().__init__()
        self.f, self.c, self.eps = f, c, eps
        self.fc = nn.Linear(in_dim, f, bias=False)
        self.bn = nn.BatchNorm1d(f, affine=True)
        self.use_tanh = use_tanh

        W = make_rademacher_codes(f, c, device='cpu')
        self.register_buffer("W", W)

    def forward(self, Z):
        n = Z.size(0)
        H_lin = self.fc(Z)
        H_bn  = self.bn(H_lin)

        if self.use_tanh:
            H_hat = torch.tanh(H_bn)
            H_hat = H_hat * math.sqrt(self.f / n)
        else:
            H_hat = l2n(H_bn, dim=1) * math.sqrt(self.f / n)

        denom = math.log((1.0 - self.eps * (self.c - 1)) / self.eps)
        tau = self.f / (math.sqrt(n) * max(denom, 1e-12))
        scores = (H_hat @ self.W) / tau
        P = F.softmax(scores, dim=1)
        return H_hat, scores, P, tau

# In[12]:
# Minimal LARS
class LARS(optim.Optimizer):
    def __init__(self, params, lr=0.1, momentum=0.9, weight_decay=1.5e-6, eta=0.001, eps=1e-8):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, eta=eta, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            wd = group['weight_decay']
            eta = group['eta']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
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
# LightningModule: CPLearn
class CPLearn_PL(pl.LightningModule):
    def __init__(self,
                 base_model=BASE_MODEL,
                 proj_out=PROJ_OUT_DIM,
                 dict_size=DICT_SIZE,
                 beta=BETA,
                 base_lr=BASE_LR,
                 weight_decay=WEIGHT_DECAY,
                 max_epochs=MAX_EPOCHS,
                 warmup_epochs=WARMUP_EPOCHS,
                 eps=EPS,
                 use_tanh=USE_TANH,
                 use_reverse_kl=USE_REVERSE_KL):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(base_model)
        self.projector = CPLearnProjector(
            self.encoder.feat_dim,
            f=proj_out,
            c=dict_size,
            use_tanh=use_tanh,
            eps=eps
        )

        self.train_losses = []
        self._cur_loss = []

    def _loss(self, batch):
        (x1, x2), _ = batch

        Z1 = self.encoder(x1)
        H1, S1, P1, tau1 = self.projector(Z1)

        Z2 = self.encoder(x2)
        H2, S2, P2, tau2 = self.projector(Z2)

        # invariance loss
        logP2 = F.log_softmax(S2 / 1.0, dim=1)
        inv = -(P1 * logP2).sum(dim=1).mean()

        # prior term on batch-average probs
        pbar = P1.mean(dim=0)
        if self.hparams.use_reverse_kl:
            log_q = math.log(1.0 / self.hparams.dict_size)
            prior = (pbar * (pbar.clamp_min(1e-12).log() - log_q)).sum()
        else:
            prior = - (1.0 / self.hparams.dict_size) * pbar.clamp_min(1e-12).log().sum()

        loss = self.hparams.beta * inv + prior
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._loss(batch)
        self._cur_loss.append(loss.item())
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        if self._cur_loss:
            self.train_losses.append(float(np.mean(self._cur_loss)))
        self._cur_loss = []

    def configure_optimizers(self):
        params_w, params_1d = [], []
        for p in self.parameters():
            if p.ndim == 1:
                params_1d.append(p)
            else:
                params_w.append(p)

        opt = LARS([
            {"params": params_w,  "lr": self.hparams.base_lr, "weight_decay": self.hparams.weight_decay},
            {"params": params_1d, "lr": self.hparams.base_lr, "weight_decay": 0.0},
        ], momentum=0.9)

        def lr_lambda(epoch):
            if epoch < self.hparams.warmup_epochs:
                return float(epoch + 1) / float(max(1, self.hparams.warmup_epochs))
            t = (epoch - self.hparams.warmup_epochs) / float(max(1, self.hparams.max_epochs - self.hparams.warmup_epochs))
            t = min(1.0, max(0.0, t))
            return 0.5 * (1.0 + math.cos(math.pi * t))

        sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        return [opt], [sched]

# In[14]:
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

# In[15]:
model = CPLearn_PL()
trainer.fit(model, train_loader)



