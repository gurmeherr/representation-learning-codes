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
CHECKPOINT_PATH = "./swav_checkpoints"
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

pl.seed_everything(42)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# In[4]:
IMG_SIZE_GLOBAL    = 224
IMG_SIZE_LOCAL     = 96
NUM_LOCAL_CROPS    = 6
BATCH_SIZE         = 256
MAX_EPOCHS         = 100
WARMUP_EPOCHS      = 10
WEIGHT_DECAY       = 1e-6
BASE_MODEL         = "resnet18"
PROJ_OUT_DIM       = 128
N_PROTOTYPES       = 3000
TAU                = 0.1
SINKHORN_EPS       = 0.05
SINKHORN_ITERS     = 3
FREEZE_PROT_EPOCHS = 1
QUEUE_SIZE         = 0

BASE_LR = 0.3 * (BATCH_SIZE / 256.0)

IMAGENET_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STD  = [0.5, 0.5, 0.5]

# In[5]:
class MultiCropTransform:
    def __init__(self, global_crops=2, local_crops=6,
                 global_size=224, local_size=96):
        jitter = transforms.ColorJitter(0.4,0.4,0.4,0.1)

        self.global_tf = transforms.Compose([
            transforms.RandomResizedCrop(global_size, scale=(0.14, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1,2.0)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        self.local_tf = transforms.Compose([
            transforms.RandomResizedCrop(local_size, scale=(0.05, 0.14)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=9, sigma=(0.1,2.0)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.gn, self.ln = global_crops, local_crops

    def __call__(self, img: Image.Image):
        crops = [self.global_tf(img) for _ in range(self.gn)]
        crops += [self.local_tf(img) for _ in range(self.ln)]
        return crops

EVAL_T = transforms.Compose([
    transforms.Resize(int(IMG_SIZE_GLOBAL*1.14)),
    transforms.CenterCrop(IMG_SIZE_GLOBAL),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

MULTICROP = MultiCropTransform(2, NUM_LOCAL_CROPS, IMG_SIZE_GLOBAL, IMG_SIZE_LOCAL)

# In[6]:
def list_images(root, exts=(".png",".jpg",".jpeg",".bmp",".tif",".tiff")):
    out = []
    for e in exts:
        out += glob.glob(os.path.join(root, '**', f'*{e}'), recursive=True)
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
            crops = self.transform(img)
            return crops, 0
        return [img, img], 0

# In[7]:
all_paths = list_images(DATASET_PATH)
assert len(all_paths) > 0, f"No images found in {DATASET_PATH}"

train_ds = FlatImageDataset(all_paths, transform=MULTICROP)

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

# In[8]:
sample_crops, _ = train_ds[0]

fig, axs = plt.subplots(2, 4, figsize=(14, 7))

for i in range(8):
    crop = sample_crops[i]
    img = crop * 0.5 + 0.5
    img = img.permute(1, 2, 0).numpy()
    r, c = divmod(i, 4)
    axs[r, c].imshow(img)
    axs[r, c].axis("off")
    if i < 2:
        axs[r, c].set_title(f"Global crop {i+1}")
    else:
        axs[r, c].set_title(f"Local crop {i-1}")

plt.suptitle("SwAV Multi-Crop Augmentations")
plt.tight_layout()
plt.show()

# In[9]:
class ProjectorMLP(nn.Module):
    def __init__(self, in_dim, hid=2048, out_dim=PROJ_OUT_DIM):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid, bias=True)
        self.bn1 = nn.BatchNorm1d(hid)
        self.fc2 = nn.Linear(hid, out_dim, bias=True)

    def forward(self, x):
        x = self.bn1(self.fc1(x))
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        return x

class EncoderWithProjector(nn.Module):
    def __init__(self, base_model=BASE_MODEL):
        super().__init__()
        base = getattr(torchvision.models, base_model)(weights=None)
        feat_dim = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.projector = ProjectorMLP(feat_dim, 2048, PROJ_OUT_DIM)

    def forward(self, x):
        h = self.backbone(x)
        z = self.projector(h)
        return h, z

# In[10]:
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
            lr, momentum = group['lr'], group['momentum']
            wd, eta, eps = group['weight_decay'], group['eta'], group['eps']
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

# In[11]:
def l2n(x, dim=1, eps=1e-12):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

@torch.no_grad()
def sinkhorn(scores, eps=SINKHORN_EPS, iters=SINKHORN_ITERS):
    Q = torch.exp(scores / eps).t()
    Q /= Q.sum()
    K, B = Q.shape
    r = torch.ones(K, device=Q.device) / K
    c = torch.ones(B, device=Q.device) / B
    for _ in range(iters):
        u = Q.sum(dim=1)
        Q *= (r / (u + 1e-12)).unsqueeze(1)
        Q *= (c / (Q.sum(dim=0) + 1e-12)).unsqueeze(0)
    Q = Q / Q.sum(dim=0, keepdim=True)
    return Q.t()

# In[12]:
class SwAV_PL(pl.LightningModule):
    def __init__(self,
                 base_model=BASE_MODEL,
                 proj_out=PROJ_OUT_DIM,
                 n_prototypes=N_PROTOTYPES,
                 tau=TAU,
                 base_lr=BASE_LR,
                 weight_decay=WEIGHT_DECAY,
                 max_epochs=MAX_EPOCHS,
                 warmup_epochs=WARMUP_EPOCHS,
                 queue_size=QUEUE_SIZE,
                 freeze_prot_epochs=FREEZE_PROT_EPOCHS):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = EncoderWithProjector(base_model)

        self.prototypes = nn.Parameter(torch.empty(proj_out, n_prototypes))
        nn.init.normal_(self.prototypes, std=0.01)
        with torch.no_grad():
            self.prototypes.data = F.normalize(self.prototypes.data, dim=0)

        self.register_buffer("queue", torch.empty(0, proj_out))
        self._use_queue = queue_size and queue_size > 0
        self._queue_size = queue_size

        self.train_losses = []
        self._cur_loss = []

    def _scores(self, z):
        return z @ F.normalize(self.prototypes, dim=0)

    @torch.no_grad()
    def _enqueue(self, z_g):
        if not self._use_queue:
            return
        z_g = z_g.detach()
        if self.queue.numel() == 0:
            self.queue = z_g
        else:
            self.queue = torch.cat([self.queue, z_g], dim=0)
        if self.queue.size(0) > self._queue_size:
            self.queue = self.queue[-self._queue_size:]

    @torch.no_grad()
    def _codes_sinkhorn(self, scores_batch):
        if self._use_queue and self.queue.numel() > 0:
            q_scores = self._scores(self.queue)
            all_scores = torch.cat([q_scores, scores_batch], dim=0)
            Q_all = sinkhorn(all_scores, eps=SINKHORN_EPS, iters=SINKHORN_ITERS)
            return Q_all[-scores_batch.size(0):]
        else:
            return sinkhorn(scores_batch, eps=SINKHORN_EPS, iters=SINKHORN_ITERS)

    def _encode_views(self, crops_list):
        zs = []
        for x in crops_list:
            _, z = self.encoder(x)
            zs.append(l2n(z, dim=1))
        return zs

    def training_step(self, batch, batch_idx):
        crops_list, _ = batch
        num_views = len(crops_list)
        zs = self._encode_views(crops_list)

        scores = [self._scores(zv) for zv in zs]
        ps = [F.log_softmax(sv / self.hparams.tau, dim=1) for sv in scores]

        with torch.no_grad():
            q0 = self._codes_sinkhorn(scores[0].detach())
            q1 = self._codes_sinkhorn(scores[1].detach())

        self._enqueue(zs[0])
        self._enqueue(zs[1])

        loss_terms = []
        for v in range(num_views):
            if v != 0:
                loss_terms.append(-(q0 * ps[v]).sum(dim=1).mean())
            if v != 1:
                loss_terms.append(-(q1 * ps[v]).sum(dim=1).mean())
        loss = sum(loss_terms) / max(1, len(loss_terms))

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

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure, **kwargs):
        if epoch < self.hparams.freeze_prot_epochs:
            if self.prototypes.grad is not None:
                self.prototypes.grad.detach_()
                self.prototypes.grad.zero_()

        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

        with torch.no_grad():
            self.prototypes.data = F.normalize(self.prototypes.data, dim=0)

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
model = SwAV_PL()
trainer.fit(model, train_loader)