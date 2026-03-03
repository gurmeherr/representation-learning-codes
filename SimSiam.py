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
CHECKPOINT_PATH = "./simsiam_checkpoints"
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

pl.seed_everything(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# In[4]:
all_images = [f for f in os.listdir(DATASET_PATH) if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif','.tiff'))]
assert len(all_images) > 0, f"No images found in {DATASET_PATH}"
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

# In[5]:
IMG_SIZE      = 224
BATCH_SIZE    = 256
MAX_EPOCHS    = 100
WARMUP_EPOCHS = 10
WEIGHT_DECAY  = 1e-4
BASE_MODEL    = "resnet18"
OUT_DIM       = 2048

# LR linear scaling
BASE_LR = 0.05 * (BATCH_SIZE / 256.0)

# In[6]:
def ensure_odd(k):
    k = int(k)
    return k if k % 2 == 1 else k + 1

class ContrastiveTransformations:
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views
    def __call__(self, x):
        return [self.base_transforms(x) for _ in range(self.n_views)]

def simsiam_transform(img_size=224):
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    ksize = ensure_odd(0.10 * img_size)
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=ksize, sigma=(0.1, 2.0))], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5]*3),
    ])

# In[7]:
def list_images(root, exts=(".png",".jpg",".jpeg",".bmp",".tif",".tiff")):
    paths = []
    for ext in exts:
        paths += glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True)
    return sorted(paths)

class FlatImageDataset(data.Dataset):
    """Reads list of image paths. For SimSiam, the transform returns [view1, view2]."""
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
        return img, 0

# In[8]:
all_paths = list_images(DATASET_PATH)
assert len(all_paths) > 0, f"No images found in {DATASET_PATH}"

train_ds = FlatImageDataset(
    all_paths,
    transform=ContrastiveTransformations(simsiam_transform(IMG_SIZE))
)

train_loader = data.DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    drop_last=True
)

print("train size:", len(train_ds))

# In[9]:
fig, axs = plt.subplots(2, 5, figsize=(15, 4))
for i in range(min(5, len(train_ds))):
    views, _ = train_ds[i]
    for j in range(2):
        img = views[j] * 0.5 + 0.5
        img = img.permute(1, 2, 0).numpy()
        axs[j, i].imshow(img)
        axs[j, i].axis('off')
        axs[j, i].set_title(f"View {j+1} - {i+1}")
plt.suptitle("SimSiam Augmented Views")
plt.tight_layout()
plt.show()

# In[10]:
class ProjectionMLP(nn.Module):
    """
    3-layer MLP for SimSiam:
    fc-BN-ReLU -> fc-BN-ReLU -> fc-BN (no ReLU)
    """
    def __init__(self, in_dim, hidden=2048, out_dim=2048, bn_affine_output=False):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )
        self.layer3_fc = nn.Linear(hidden, out_dim, bias=False)
        self.layer3_bn = nn.BatchNorm1d(out_dim, affine=bn_affine_output)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3_bn(self.layer3_fc(x))
        return x

class PredictionMLP(nn.Module):
    """
    2-layer predictor for SimSiam:
    fc-BN-ReLU -> fc
    """
    def __init__(self, in_dim=2048, hidden=512, out_dim=2048):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(in_dim, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.out(self.hidden(x))

class ResNetSimSiam(nn.Module):
    def __init__(self, base_model="resnet18", out_dim=2048):
        super().__init__()
        base = getattr(torchvision.models, base_model)(weights=None)
        feat_dim = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.projector = ProjectionMLP(feat_dim, hidden=2048, out_dim=out_dim, bn_affine_output=False)
        self.predictor = PredictionMLP(in_dim=out_dim, hidden=512, out_dim=out_dim)

    def forward(self, x):
        h = self.backbone(x)
        if h.ndim > 2:
            h = torch.flatten(h, 1)
        z = self.projector(h)
        p = self.predictor(z)
        return z, p

# In[11]:
class SimSiam_PL(pl.LightningModule):
    def __init__(self,
                 base_model=BASE_MODEL,
                 out_dim=OUT_DIM,
                 base_lr=BASE_LR,
                 weight_decay=WEIGHT_DECAY,
                 max_epochs=MAX_EPOCHS,
                 warmup_epochs=WARMUP_EPOCHS):
        super().__init__()
        self.save_hyperparameters()
        self.model = ResNetSimSiam(base_model=base_model, out_dim=out_dim)

        self.train_losses = []
        self.train_align = []
        self.train_std = []

        self._cur_loss = []
        self._cur_align = []
        self._cur_std = []

    @staticmethod
    def neg_cosine(p, z):
        z = z.detach()
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return -(p * z).sum(dim=1).mean()

    def simsiam_step(self, batch):
        (x1, x2), _ = batch
        z1, p1 = self.model(x1)
        z2, p2 = self.model(x2)

        loss = 0.5 * self.neg_cosine(p1, z2) + 0.5 * self.neg_cosine(p2, z1)
        disp_align = 1.0 + loss.detach()

        with torch.no_grad():
            std1 = F.normalize(z1, dim=1).std(dim=0).mean().item()
            std2 = F.normalize(z2, dim=1).std(dim=0).mean().item()
            out_std = 0.5 * (std1 + std2)

        return loss, disp_align.item(), out_std

    def training_step(self, batch, batch_idx):
        loss, disp_align, out_std = self.simsiam_step(batch)

        self._cur_loss.append(loss.item())
        self._cur_align.append(disp_align)
        self._cur_std.append(out_std)

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_align_loss", disp_align, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_out_std", out_std, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        if self._cur_loss:
            self.train_losses.append(np.mean(self._cur_loss))
            self.train_align.append(np.mean(self._cur_align))
            self.train_std.append(np.mean(self._cur_std))
        self._cur_loss, self._cur_align, self._cur_std = [], [], []

    def configure_optimizers(self):
        opt = optim.SGD(
            self.parameters(),
            lr=self.hparams.base_lr,
            momentum=0.9,
            weight_decay=self.hparams.weight_decay
        )

        def lr_lambda(epoch):
            if epoch < self.hparams.warmup_epochs:
                return float(epoch + 1) / float(max(1, self.hparams.warmup_epochs))
            t = (epoch - self.hparams.warmup_epochs) / float(max(1, self.hparams.max_epochs - self.hparams.warmup_epochs))
            t = min(1.0, max(0.0, t))
            return 0.5 * (1.0 + math.cos(math.pi * t))

        sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        return [opt], [sched]

# In[12]:
checkpoint_cb = ModelCheckpoint(
    dirpath=CHECKPOINT_PATH,
    save_top_k=1,
    monitor="train_loss",
    mode="min"
)

# In[13]:
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
model = SimSiam_PL()
trainer.fit(model, train_loader)


# In[ ]:




