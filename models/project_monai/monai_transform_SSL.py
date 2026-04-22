import copy

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    RandSpatialCropd,
    RandFlipd,
    RandRotate90d,
    RandGaussianNoised,
    ToTensord
)
from monai.data import CacheDataset, DataLoader
from monai.networks.nets import resnet

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# Daten
# -------------------------
train_files = [
    {"image": "./MMDental/1/1.nii.gz"},
    {"image": "./MMDental/3/3.nii.gz"},
    {"image": "./MMDental/4/4.nii.gz"},
    {"image": "./MMDental/6/6.nii.gz"},
    {"image": "./MMDental/7/7.nii.gz"},
    {"image": "./MMDental/8/8.nii.gz"},
    {"image": "./MMDental/9/9.nii.gz"},
    {"image": "./MMDental/10/10.nii.gz"},
    {"image": "./MMDental/15/15.nii.gz"},
    {"image": "./MMDental/18/18.nii.gz"},
    {"image": "./MMDental/19/19.nii.gz"},
    {"image": "./MMDental/23/23.nii.gz"},
    {"image": "./MMDental/26/26.nii.gz"},
    {"image": "./MMDental/27/27.nii.gz"},
    {"image": "./MMDental/28/28.nii.gz"},
    {"image": "./MMDental/30/30.nii.gz"},
    {"image": "./MMDental/32/32.nii.gz"},
    {"image": "./MMDental/33/33.nii.gz"},
    {"image": "./MMDental/35/35.nii.gz"},
    {"image": "./MMDental/38/38.nii.gz"},
    {"image": "./MMDental/39/39.nii.gz"},
    {"image": "./MMDental/40/40.nii.gz"},
    {"image": "./MMDental/41/41.nii.gz"},
    {"image": "./MMDental/43/43.nii.gz"},
    {"image": "./MMDental/45/45.nii.gz"},
    {"image": "./MMDental/46/46.nii.gz"},
    {"image": "./MMDental/47/47.nii.gz"},
    {"image": "./MMDental/48/48.nii.gz"},
    {"image": "./MMDental/49/49.nii.gz"},
    {"image": "./MMDental/52/52.nii.gz"},
    {"image": "./MMDental/54/54.nii.gz"},
    {"image": "./MMDental/55/55.nii.gz"},
    {"image": "./MMDental/56/56.nii.gz"},
    {"image": "./MMDental/57/57.nii.gz"},
    {"image": "./MMDental/58/58.nii.gz"},
    {"image": "./MMDental/59/59.nii.gz"},
    {"image": "./MMDental/60/60.nii.gz"},
    {"image": "./MMDental/61/61.nii.gz"},
    {"image": "./MMDental/62/62.nii.gz"},
    {"image": "./MMDental/63/63.nii.gz"},
    {"image": "./MMDental/64/64.nii.gz"},
    {"image": "./MMDental/66/66.nii.gz"},
    {"image": "./MMDental/67/67.nii.gz"},
    {"image": "./MMDental/68/68.nii.gz"},
    {"image": "./MMDental/69/69.nii.gz"},
    {"image": "./MMDental/71/71.nii.gz"},
    {"image": "./MMDental/72/72.nii.gz"},
    {"image": "./MMDental/73/73.nii.gz"},
    {"image": "./MMDental/74/74.nii.gz"},
    {"image": "./MMDental/75/75.nii.gz"},
    {"image": "./MMDental/76/76.nii.gz"},
    {"image": "./MMDental/77/77.nii.gz"},
    {"image": "./MMDental/82/82.nii.gz"},
    {"image": "./MMDental/85/85.nii.gz"},
    {"image": "./MMDental/88/88.nii.gz"},
    {"image": "./MMDental/89/89.nii.gz"},
    {"image": "./MMDental/90/90.nii.gz"},
    {"image": "./MMDental/92/92.nii.gz"},
]


# Define transforms (Intensity_scaling, Random_spatial_crop)
base_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    ScaleIntensityd(keys=["image"]),

    RandSpatialCropd(
        keys=["image"],
        roi_size=(64, 64, 64),
        random_size=False
    ),

    ToTensord(keys=["image"])
])



def create_views(batch):
    aug = Compose([
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
        RandRotate90d(keys=["image"], prob=0.3),
        RandGaussianNoised(keys=["image"], prob=0.1, std=0.01),
    ])

    x = batch["image"]  # [B, C, D, H, W]

    x1_list = []
    x2_list = []

    for i in range(x.shape[0]):
        sample = {"image": x[i]}

        x1 = aug(copy.deepcopy(sample))["image"]
        x2 = aug(copy.deepcopy(sample))["image"]

        x1_list.append(x1)
        x2_list.append(x2)

    x1 = torch.stack(x1_list)
    x2 = torch.stack(x2_list)

    return x1, x2



dataset = CacheDataset(
    data=train_files,
    transform=base_transforms,
    cache_rate=1.0
)

loader = DataLoader(dataset, batch_size=16, shuffle=True)



class SSLModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = resnet.ResNet(
            block='basic',
            layers=[2,2,2,2],
            block_inplanes=[64,128,256,512],
            spatial_dims=3,
            n_input_channels=1,
            num_classes=512
        )

        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        h = self.encoder(x)
        return self.head(h)

# Define loss function
def contrastive_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    logits = torch.mm(z1, z2.T) / temperature
    labels = torch.arange(z1.size(0)).to(z1.device)

    loss1 = F.cross_entropy(logits, labels)
    loss2 = F.cross_entropy(logits.T, labels)

    return (loss1 + loss2) / 2

model = SSLModel().to("cuda")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Train model
for epoch in range(500):
    for batch in loader:

        x1, x2 = create_views(batch)

        x1 = x1.to(device)
        x2 = x2.to(device)

        z1 = model(x1)
        z2 = model(x2)

        loss = contrastive_loss(z1, z2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}")

