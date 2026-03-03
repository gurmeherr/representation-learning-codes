#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install -q pytorch-lightning==2.2.5 torchvision')

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


# In[6]:


DATASET_PATH = "./ViTacTip_Dataset_Final"   
CHECKPOINT_PATH = "./simclr_checkpoint"
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

pl.seed_everything(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# In[7]:


# visualize a few random images
all_images = [f for f in os.listdir(DATASET_PATH) if f.endswith('.png')]
image_files = random.sample(all_images, min(5, len(all_images)))

fig, axs = plt.subplots(1, len(image_files), figsize=(15, 3))
for idx, file_name in enumerate(image_files):
    img = Image.open(os.path.join(DATASET_PATH, file_name)).convert("RGB")
    axs[idx].imshow(img)
    axs[idx].set_title(file_name, fontsize=8)
    axs[idx].axis('off')

plt.suptitle("Random Sample of ViTacTip Images", fontsize=14)
plt.tight_layout()
plt.show()


# In[8]:


# Training parameters
IMG_SIZE      = 224
BATCH_SIZE    = 256
MAX_EPOCHS    = 100
WARMUP_EPOCHS = 10
TEMPERATURE   = 0.1
WEIGHT_DECAY  = 1e-6
BASE_MODEL    = "resnet18"
OUT_DIM       = 128

# LR scaling rule-of-thumb from SimCLR
BASE_LR = 0.3 * (BATCH_SIZE / 256.0)


# In[9]:


def ensure_odd(k):
    k = int(k)
    return k if k % 2 == 1 else k + 1

class ContrastiveTransformations:
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views
    def __call__(self, x):
        return [self.base_transforms(x) for _ in range(self.n_views)]

def simclr_transform(img_size=224, s=1.0):
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    ksize = ensure_odd(0.10 * img_size)  # ~10% of image size, must be odd
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0), ratio=(0.75, 1.3333)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=ksize, sigma=(0.1, 2.0))], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5]*3),
    ])


# In[10]:


def list_images(root, exts=(".png",".jpg",".jpeg",".bmp",".tif",".tiff")):
    paths = []
    for ext in exts:
        paths += glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True)
    return sorted(paths)

# Datasets
class FlatImageDataset(data.Dataset):
    """Reads list of image paths. For SimCLR, the transform returns [view1, view2]."""
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0

class TransformSubset(data.Dataset):
    """Wrap a Dataset and apply transforms on fetch."""
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx):
        img, lbl = self.subset[idx]   # img is PIL
        if self.transform is not None:
            img = self.transform(img)
        return img, lbl


# In[11]:


all_paths = list_images(DATASET_PATH)
assert len(all_paths) > 0, f"No images found in {DATASET_PATH}"
base_dataset = FlatImageDataset(all_paths, transform=None)


# In[12]:


train_ds = TransformSubset(base_dataset, ContrastiveTransformations(simclr_transform(IMG_SIZE)))
train_loader = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, drop_last=True)

print("Train samples:", len(train_ds))


# In[13]:


fig, axs = plt.subplots(2, 5, figsize=(15, 4))
for i in range(min(5, len(train_ds))):
    views, _ = train_ds[i]
    for j in range(2):
        img = views[j] * 0.5 + 0.5
        img = img.permute(1, 2, 0).numpy()
        axs[j, i].imshow(img)
        axs[j, i].axis('off')
        axs[j, i].set_title(f"View {j+1} - {i+1}")
plt.suptitle("Contrastive Views")
plt.tight_layout()
plt.show()


# In[14]:


# Model & Optim 
class ResNetSimCLR(nn.Module):
    def __init__(self, base_model="resnet18", out_dim=128):
        super().__init__()
        base = getattr(torchvision.models, base_model)(weights=None)
        feat_dim = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, out_dim)
        )
    def forward(self, x):
        h = self.backbone(x)
        if h.ndim > 2:
            h = torch.flatten(h, 1)
        z = self.projector(h)
        return z

class LARS(optim.Optimizer):
    """Minimal LARS with momentum; excludes 1D params (bias/norm) from WD & adaptation."""
    def __init__(self, params, lr=0.1, momentum=0.9, weight_decay=1e-6, eta=0.001, eps=1e-8):
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


# In[15]:


class SimCLR_PL(pl.LightningModule):
    def __init__(self,
                 base_model=BASE_MODEL,
                 out_dim=OUT_DIM,
                 temperature=TEMPERATURE,
                 base_lr=BASE_LR,
                 weight_decay=WEIGHT_DECAY,
                 max_epochs=MAX_EPOCHS,
                 warmup_epochs=WARMUP_EPOCHS):
        super().__init__()
        self.save_hyperparameters()
        self.model = ResNetSimCLR(base_model=base_model, out_dim=out_dim)
        self.temperature = temperature
        # history for plotting
        self.train_losses = []
        self._cur_loss = []

    def info_nce_loss(self, batch):
        (x_i, x_j), _ = batch
        x = torch.cat([x_i, x_j], dim=0)
        z = F.normalize(self.model(x), dim=1)
        sim = torch.matmul(z, z.T)  # cosine (z normalized)
        n = sim.size(0)
        eye = torch.eye(n, dtype=torch.bool, device=sim.device)
        sim.masked_fill_(eye, -9e15)
        b = n // 2
        pos_mask = eye.roll(shifts=b, dims=0)
        sim = sim / self.temperature
        nll = -sim[pos_mask] + torch.logsumexp(sim, dim=1)
        loss = nll.mean()
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.info_nce_loss(batch)
        self._cur_loss.append(loss.item())
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        if self._cur_loss:
            self.train_losses.append(np.mean(self._cur_loss))
        self._cur_loss = []

    def configure_optimizers(self):
        opt = LARS(self.parameters(), lr=self.hparams.base_lr, momentum=0.9,
                   weight_decay=self.hparams.weight_decay, eta=0.001)
        def lr_lambda(epoch):
            if epoch < self.hparams.warmup_epochs:
                return float(epoch + 1) / float(max(1, self.hparams.warmup_epochs))
            t = (epoch - self.hparams.warmup_epochs) / float(max(1, self.hparams.max_epochs - self.hparams.warmup_epochs))
            t = min(1.0, max(0.0, t))
            return 0.5 * (1.0 + math.cos(math.pi * t))  # cosine
        sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        return [opt], [sched]


# In[16]:


checkpoint_cb = ModelCheckpoint(
    dirpath=CHECKPOINT_PATH,
    save_top_k=1,
    monitor="train_loss",
    mode="min"
)


# In[17]:


trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    logger=False,
    callbacks=[checkpoint_cb],
    enable_progress_bar=True,
    log_every_n_steps=25
)


# In[18]:


model = SimCLR_PL()
trainer.fit(model, train_loader)


# In[ ]:




