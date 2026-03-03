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

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# In[3]:
DATASET_PATH = "./ViTacTip_Dataset_Final"
CHECKPOINT_PATH = "./minc_checkpoints"
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

pl.seed_everything(42)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# In[4]:
flat = [f for f in os.listdir(DATASET_PATH)
        if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif','.tiff'))]
if len(flat) >= 5:
    samples = [os.path.join(DATASET_PATH, f) for f in random.sample(flat, 5)]
else:
    exts = ('.png','.jpg','.jpeg','.bmp','.tif','.tiff')
    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(DATASET_PATH, '**', f'*{e}'), recursive=True)
    samples = random.sample(paths, min(5, len(paths)))

fig, axs = plt.subplots(1, len(samples), figsize=(15,3))
for i,p in enumerate(samples):
    axs[i].imshow(Image.open(p).convert("RGB"))
    axs[i].axis('off')
    axs[i].set_title(os.path.relpath(p, DATASET_PATH), fontsize=8)
plt.suptitle("Random ViTacTip Images")
plt.tight_layout()
plt.show()

# In[5]:
IMG_SIZE      = 224
BATCH_SIZE    = 256
MAX_EPOCHS    = 300
WARMUP_EPOCHS = 10
WEIGHT_DECAY  = 1e-4
BASE_MODEL    = "resnet18"
PROJ_DIM      = 2048
ALPHA         = 2.0
BETA_AUX      = 0.8
GAMMA_TGT     = 0.996
INNER_SCALE   = 10.0

BASE_LR_WEIGHTS = 0.3 * (BATCH_SIZE / 256.0)
BASE_LR_BIASBN  = 0.0048 * (BATCH_SIZE / 256.0)

IMAGENET_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STD  = [0.5, 0.5, 0.5]

# In[6]:
class TwoCrops:
    def __init__(self, T1, T2):
        self.T1, self.T2 = T1, T2
    def __call__(self, x):
        return self.T1(x), self.T2(x)

def byol_transforms(img_size=224):
    jitter = transforms.ColorJitter(0.4,0.4,0.4,0.1)
    common = [
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
    ]
    T1 = transforms.Compose([
        *common,
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1,2.0)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    T2 = transforms.Compose([
        *common,
        transforms.RandomApply([transforms.GaussianBlur(23, (0.1,2.0))], p=0.1),
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

T1, T2, EVAL_T = byol_transforms(IMG_SIZE)
TWOCROP = TwoCrops(T1, T2)

# In[7]:
def list_images(root, exts=(".png",".jpg",".jpeg",".bmp",".tif",".tiff")):
    out = []
    for e in exts:
        out += glob.glob(os.path.join(root, "**", f"*{e}"), recursive=True)
    return sorted(out)

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
all_paths = list_images(DATASET_PATH)
assert len(all_paths) > 0, f"No images found in {DATASET_PATH}"

train_ds = FlatImageDataset(all_paths, transform=TWOCROP)

train_loader = data.DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    drop_last=True
)

print("Train images:", len(train_ds))

# In[9]:
fig, axs = plt.subplots(2, 5, figsize=(15, 4))
mean = torch.tensor(IMAGENET_MEAN).view(3,1,1)
std  = torch.tensor(IMAGENET_STD).view(3,1,1)

for i in range(min(5, len(train_ds))):
    (v1, v2), _ = train_ds[i]
    im1 = (v1*std + mean).clamp(0,1).permute(1,2,0).numpy()
    im2 = (v2*std + mean).clamp(0,1).permute(1,2,0).numpy()
    axs[0,i].imshow(im1); axs[0,i].axis('off'); axs[0,i].set_title(f"View A {i+1}")
    axs[1,i].imshow(im2); axs[1,i].axis('off'); axs[1,i].set_title(f"View B {i+1}")
plt.suptitle("BYOL-style Augs used by MINC")
plt.tight_layout()
plt.show()

# In[10]:
class ProjectorMLP(nn.Module):
    def __init__(self, in_dim, hid=2048, out_dim=PROJ_DIM):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid, bias=True)
        self.bn1 = nn.BatchNorm1d(hid)
        self.fc2 = nn.Linear(hid, hid, bias=True)
        self.bn2 = nn.BatchNorm1d(hid)
        self.fc3 = nn.Linear(hid, out_dim, bias=True)
    def forward(self, x):
        x = self.bn1(self.fc1(x)); x = F.relu(x, inplace=True)
        x = self.bn2(self.fc2(x)); x = F.relu(x, inplace=True)
        x = self.fc3(x)
        return x

class EncoderWithProjector(nn.Module):
    def __init__(self, base_model=BASE_MODEL):
        super().__init__()
        base = getattr(torchvision.models, base_model)(weights=None)
        feat_dim = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.projector = ProjectorMLP(feat_dim, hid=2048, out_dim=PROJ_DIM)
    def forward(self, x):
        h = self.backbone(x)
        z = self.projector(h)
        return h, z

# In[11]:
class LARS(optim.Optimizer):
    def __init__(self, params, lr=0.1, momentum=0.9, weight_decay=1e-4, eta=0.001, eps=1e-8):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, eta=eta, eps=eps)
        super().__init__(params, defaults)
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
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
def l2_normalize(z, dim=1, eps=1e-12):
    return z / (z.norm(dim=dim, keepdim=True) + eps)

def lower_triangular(mat):
    return torch.tril(mat)

def t_alpha(u, alpha=ALPHA, mode="alpha2_linear"):
    if mode == "alpha2_linear":
        return u
    else:
        a = alpha
        s = torch.sqrt(torch.tensor(a/2.0, device=u.device, dtype=u.dtype))
        expo = 2.0*(a-1.0)/a
        return torch.sign(u) * (torch.abs(s*u)**expo - 1.0) / (a - 1.0)

# In[13]:
class MINC_PL(pl.LightningModule):
    def __init__(self,
                 base_model=BASE_MODEL,
                 proj_dim=PROJ_DIM,
                 beta_aux=BETA_AUX,
                 gamma_tgt=GAMMA_TGT,
                 inner_scale=INNER_SCALE,
                 base_lr_w=BASE_LR_WEIGHTS,
                 base_lr_b=BASE_LR_BIASBN,
                 weight_decay=WEIGHT_DECAY,
                 max_epochs=MAX_EPOCHS,
                 warmup_epochs=WARMUP_EPOCHS):
        super().__init__()
        self.save_hyperparameters()

        self.online = EncoderWithProjector(base_model)
        self.target = EncoderWithProjector(base_model)
        for p_o, p_t in zip(self.online.parameters(), self.target.parameters()):
            p_t.data.copy_(p_o.data)
            p_t.requires_grad = False

        self.register_buffer("Lambda", torch.zeros(proj_dim, proj_dim))
        self.s = inner_scale

        self._cur_loss = []
        self.train_losses = []

    @torch.no_grad()
    def _update_target(self):
        for p_o, p_t in zip(self.online.parameters(), self.target.parameters()):
            p_t.data.mul_(self.hparams.gamma_tgt).add_(p_o.data, alpha=1 - self.hparams.gamma_tgt)

    @torch.no_grad()
    def _update_lambda(self, z_targ):
        B = z_targ.size(0)
        batch_outer = (z_targ.T @ z_targ) / B
        self.Lambda.mul_(self.hparams.beta_aux).add_(batch_outer, alpha=1 - self.hparams.beta_aux)

    def _minc_loss(self, batch):
        (x1, x2), _ = batch

        _, z2 = self.online(x2)
        z2 = l2_normalize(z2, dim=1)

        with torch.no_grad():
            _, z1_t = self.target(x1)
            z1_t = l2_normalize(z1_t, dim=1)

        dots = (z1_t * z2).sum(dim=1)
        term1 = t_alpha(self.s * dots, alpha=ALPHA, mode="alpha2_linear").mean()

        LT = lower_triangular(self.Lambda)
        quad = (z2 @ LT) * z2
        term2 = 0.5 * (self.s**2) * quad.sum(dim=1).mean()

        loss = -(term1 - term2)
        return loss, z1_t

    def training_step(self, batch, batch_idx):
        loss, z1_t = self._minc_loss(batch)
        self._cur_loss.append(loss.item())

        self._update_lambda(z1_t)

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        if self._cur_loss:
            self.train_losses.append(float(np.mean(self._cur_loss)))
        self._cur_loss = []

    def configure_optimizers(self):
        params_w, params_bb = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1:
                params_bb.append(p)
            else:
                params_w.append(p)

        opt = LARS([
            {"params": params_w,  "lr": self.hparams.base_lr_w, "weight_decay": self.hparams.weight_decay},
            {"params": params_bb, "lr": self.hparams.base_lr_b, "weight_decay": 0.0},
        ], momentum=0.9)

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
        self._update_target()

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

model = MINC_PL()
trainer.fit(model, train_loader)