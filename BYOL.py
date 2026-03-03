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
DATASET_PATH = "./ViTacTip_Dataset_Final"
CHECKPOINT_PATH = "./byol_checkpoints"
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

pl.seed_everything(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# In[4]:
# Show a few random images
all_images = [f for f in os.listdir(DATASET_PATH) if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif','.tiff'))]
if len(all_images) >= 5:
    image_files = random.sample(all_images, 5)
else:
    def _list_images(root):
        out = []
        for ext in ('.png','.jpg','.jpeg','.bmp','.tif','.tiff'):
            out += glob.glob(os.path.join(root, '**', f'*{ext}'), recursive=True)
        return out
    paths = _list_images(DATASET_PATH)
    image_files = [os.path.relpath(p, DATASET_PATH) for p in random.sample(paths, min(5, len(paths)))]

fig, axs = plt.subplots(1, len(image_files), figsize=(15, 3))
for idx, file_name in enumerate(image_files):
    img = Image.open(os.path.join(DATASET_PATH, file_name)).convert("RGB")
    axs[idx].imshow(img)
    axs[idx].set_title(file_name, fontsize=8)
    axs[idx].axis('off')
plt.suptitle("Random Sample of ViTacTip Images", fontsize=14)
plt.tight_layout()
plt.show()

# In[5]:
# Training parameters
IMG_SIZE = 224
BATCH_SIZE = 256
MAX_EPOCHS = 100
WARMUP_EPOCHS = 10
WEIGHT_DECAY = 1.5e-6
BASE_MODEL = "resnet18"
PROJ_HIDDEN = 4096
PROJ_OUT_DIM = 256
TAU_BASE = 0.996

# LR scaling rule-of-thumb (BYOL uses LARS; base 0.2@bs=256)
BASE_LR = 0.2 * (BATCH_SIZE / 256.0)

IMAGENET_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STD  = [0.5, 0.5, 0.5]

# In[6]:
# Augmentations: BYOL uses two asymmetric pipelines T and T'
class TwoCropsBYOL:
    def __init__(self, T1: transforms.Compose, T2: transforms.Compose):
        self.T1 = T1; self.T2 = T2
    def __call__(self, x: Image.Image):
        return self.T1(x), self.T2(x)

def byol_transforms(img_size=224):
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    common = [
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
    ]
    T1 = transforms.Compose([
        *common,
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),   # p=1.0
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
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return T1, T2, EVAL_T

T1, T2, EVAL_T = byol_transforms(IMG_SIZE)
TWOCROP = TwoCropsBYOL(T1, T2)

# In[7]:
# List images and dataset
def list_images(root, exts=(".png",".jpg",".jpeg",".bmp",".tif",".tiff")):
    paths = []
    for ext in exts:
        paths += glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True)
    return sorted(paths)

class FlatImageDataset(data.Dataset):
    """Reads list of image paths. For BYOL, transform returns (view1, view2)."""
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

all_paths = list_images(DATASET_PATH)
assert len(all_paths) > 0, f"No images found in {DATASET_PATH}"
base_dataset = FlatImageDataset(all_paths, transform=TWOCROP)
train_ds = base_dataset
train_loader = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
print("Train samples:", len(train_ds))

# In[9]:
# Visualize the two BYOL views
fig, axs = plt.subplots(2, 5, figsize=(15, 4))
for i in range(min(5, len(train_ds))):
    (v1, v2), _ = train_ds[i]
    mean = torch.tensor(IMAGENET_MEAN).view(3,1,1)
    std  = torch.tensor(IMAGENET_STD).view(3,1,1)
    im1 = (v1*std + mean).clamp(0,1).permute(1,2,0).numpy()
    im2 = (v2*std + mean).clamp(0,1).permute(1,2,0).numpy()
    axs[0, i].imshow(im1); axs[0, i].axis('off'); axs[0, i].set_title(f"T view {i+1}")
    axs[1, i].imshow(im2); axs[1, i].axis('off'); axs[1, i].set_title(f"T' view {i+1}")
plt.suptitle("BYOL Asymmetric Views (T vs T')")
plt.tight_layout(); plt.show()

# In[10]:
# BYOL Networks
class ProjectorMLP(nn.Module):
    def __init__(self, in_dim, hid=PROJ_HIDDEN, out_dim=PROJ_OUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.BatchNorm1d(hid),
            nn.ReLU(inplace=True),
            nn.Linear(hid, out_dim)  # no BN on output
        )
    def forward(self, x):
        return self.net(x)

class PredictorMLP(nn.Module):
    def __init__(self, in_dim=PROJ_OUT_DIM, hid=PROJ_HIDDEN, out_dim=PROJ_OUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.BatchNorm1d(hid),
            nn.ReLU(inplace=True),
            nn.Linear(hid, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class BYOLBackbone(nn.Module):
    def __init__(self, base_model=BASE_MODEL):
        super().__init__()
        base = getattr(torchvision.models, base_model)(weights=None)
        feat_dim = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.projector = ProjectorMLP(feat_dim)
    def forward(self, x):
        h = self.backbone(x)
        z = self.projector(h)
        return h, z  # z is NOT normalized here

# In[11]:
# LARS
class LARS(optim.Optimizer):
    """Minimal LARS with momentum; excludes 1D params (bias/norm) from WD & adaptation."""
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

# In[12]:
# BYOL Lightning
class BYOL_PL(pl.LightningModule):
    def __init__(self,
                 base_model=BASE_MODEL,
                 proj_hidden=PROJ_HIDDEN,
                 proj_out=PROJ_OUT_DIM,
                 base_lr=BASE_LR,
                 weight_decay=WEIGHT_DECAY,
                 max_epochs=MAX_EPOCHS,
                 warmup_epochs=WARMUP_EPOCHS,
                 tau_base=TAU_BASE):
        super().__init__()
        self.save_hyperparameters()
        # Online & target twins + predictor
        self.online = BYOLBackbone(base_model)
        self.target = BYOLBackbone(base_model)
        self.predictor = PredictorMLP(in_dim=proj_out, hid=proj_hidden, out_dim=proj_out)
        # target <- online (init)
        for p_o, p_t in zip(self.online.parameters(), self.target.parameters()):
            p_t.data.copy_(p_o.data)
            p_t.requires_grad = False
        # EMA tracking
        self.tau = tau_base
        self._global_step = 0
        self._total_steps = None
        # loss history (for optional plotting)
        self.train_losses = []
        self._cur_loss = []

    def _loss(self, batch):
        (x1, x2), _ = batch
        # Online
        _, z1 = self.online(x1)
        _, z2 = self.online(x2)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        # Target (no grad)
        with torch.no_grad():
            _, t1 = self.target(x1)
            _, t2 = self.target(x2)
        # Normalize in the loss space
        p1 = F.normalize(p1, dim=1); p2 = F.normalize(p2, dim=1)
        t1 = F.normalize(t1, dim=1); t2 = F.normalize(t2, dim=1)
        # Symmetric BYOL loss
        loss1 = 2 - 2 * (p1 * t2.detach()).sum(dim=-1)
        loss2 = 2 - 2 * (p2 * t1.detach()).sum(dim=-1)
        return (loss1 + loss2).mean()

    # Lightning hooks 
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
        opt = LARS(self.parameters(), lr=self.hparams.base_lr, momentum=0.9,
                   weight_decay=self.hparams.weight_decay, eta=0.001)
        # Warmup + cosine
        def lr_lambda(epoch):
            if epoch < self.hparams.warmup_epochs:
                return float(epoch + 1) / float(max(1, self.hparams.warmup_epochs))
            t = (epoch - self.hparams.warmup_epochs) / float(max(1, self.hparams.max_epochs - self.hparams.warmup_epochs))
            t = min(1.0, max(0.0, t))
            return 0.5 * (1.0 + math.cos(math.pi * t))
        sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        return [opt], [sched]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure, **kwargs):
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
        # Update EMA tau with cosine schedule over total steps -> 1.0
        if self._total_steps is None and self.trainer is not None:
            self._total_steps = self.trainer.estimated_stepping_batches
        self._global_step += 1
        cos = 0.5 * (1 + math.cos(math.pi * self._global_step / max(1, self._total_steps or 1)))
        self.tau = 1 - (1 - self.hparams.tau_base) * cos
        # Momentum update of target
        with torch.no_grad():
            for p_o, p_t in zip(self.online.parameters(), self.target.parameters()):
                p_t.data.mul_(self.tau).add_(p_o.data, alpha=(1 - self.tau))

# In[13]:
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

# In[14]:
model = BYOL_PL()
trainer.fit(model, train_loader)






