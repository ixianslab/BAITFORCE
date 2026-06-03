"""
SSL Pretraining + Downstream Kaukraft-Regression
fuer CBCT Scans (Schaedel, Ober- und Unterkiefer)

VERBESSERUNGEN v5:
  [1] Gewichteter MAE Loss — Knochen-Patches 4x wichtiger als Luft
  [2] Warmup + Cosine LR Schedule — stabiles Training von Anfang an
  [3] AdamW mit MAE-optimalen Parametern (betas, weight_decay)
  [4] Kleinere Patch-Size (4) + ROI 64^3 — feinere Strukturen sichtbar
  [5] Hybrid CNN+Transformer Encoder — lokale UND globale Features
  [6] Block-Masking — verhindert Interpolations-Schummeln
  [7] MAE-spezifische Augmentierungen — keine starken Rotationen
  [8] Training-Monitoring — Collapse-Erkennung alle 20 Epochs
  [9] InstanceNorm statt BatchNorm in CNN — stabiler bei kleinen Batches

Ausfuehren:
    python ssl_training.py --mode ssl --ssl_mode mae     # MAE (empfohlen)
    python ssl_training.py --mode ssl --ssl_mode byol    # Multi-Scale BYOL
    python ssl_training.py --mode regression --ssl_mode mae
    python ssl_training.py --mode both --ssl_mode mae

Hardware: NVIDIA RTX 4070 Ti Super (16 GB VRAM), 32 GB RAM
"""

import os
import sys
import copy
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    ScaleIntensityd,
    ScaleIntensityRangePercentilesd,
    Resized,
    RandSpatialCropd,
    CenterSpatialCropd,
    RandAffined,
    RandFlipd,
    RandShiftIntensityd,
    RandScaleIntensityd,
    RandGaussianNoised,
    RandCoarseDropoutd,
    EnsureTyped,
)

#from monai.transforms import EnsureTyped
from torch.utils.data._utils.collate import default_collate
from monai.data import PersistentDataset, DataLoader
from monai.networks.nets import resnet


# ─────────────────────────────────────────────
# Argumente
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="SSL Pretraining + Kaukraft-Regression fuer CBCT"
    )
    parser.add_argument(
        "--mode", type=str,
        choices=["ssl", "regression", "both"],
        default="both",
    )
    parser.add_argument(
        "--ssl_mode", type=str,
        choices=["mae", "byol"],
        default="mae",
        help="SSL-Methode: 'mae' (empfohlen) oder 'byol'",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────
# Debug-Info / Device
# ─────────────────────────────────────────────
print("CWD        :", os.getcwd())
print("Python     :", sys.executable)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device     :", device)
if torch.cuda.is_available():
    print("GPU        :", torch.cuda.get_device_name(0))
    print("VRAM       :", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")


# ─────────────────────────────────────────────
# Konfiguration
# ─────────────────────────────────────────────
BASE_DIR_UNLABELED  = "D:/chewing_gum/DATASETS/RAW/MMDental"
LABELS_JSON         = "D:/chewing_gum/DATASETS/labels.json"
CACHE_DIR_GLOBAL    = "./cache_global_monai"
CACHE_DIR_LOCAL     = "./cache_local_monai"
CHECKPOINT_SSL      = "./checkpoint_ssl"
CHECKPOINT_REG      = "./checkpoint_regression.pt"
MODEL_SSL_FINAL     = "./ssl_model_final.pt"
MODEL_REG_FINAL     = "./regression_model_final.pt"
MONITOR_DIR         = "./monitoring"       # Collapse-Monitoring Plots

# Multi-Scale ROI
ROI_GLOBAL = (128, 128, 128)   # ganzer Schaedel resized — Morphologie

# [4] Kleinere Patch-Size + fokussierter ROI
# 64^3 bei 0.25mm = 16mm Wuerfel — erfasst gesamten Kiefer-Querschnitt
# patch_size=4: 4 x 0.25mm = 1mm pro Patch → Kortikalis = 2 Patches sichtbar
# (64//4)^3 = 4096 Patches — VRAM-vertraeglich
ROI_LOCAL      = (64, 64, 64)
MAE_PATCH_SIZE = 8

MAE_MASK_RATIO      = 0.75     # 75% maskieren — MAE-Standard
BONE_WEIGHT_FACTOR  = 1.0      # [1] Knochen-Patches x4 gewichtet im Loss
WARMUP_EPOCHS       = 20       # [2] LR-Warmup Epochs
MONITOR_INTERVAL    = 5       # [8] Collapse-Check alle N Epochs

# SSL Training
SSL_BATCH_SIZE      = 4
ACCUMULATION_STEPS  = 4        # effektive Batch Size = 16
NUM_WORKERS         = 2
LR_SSL              = 0.5e-4   # [3] leicht hoeher als 1e-4
BYOL_TAU            = 0.996
SSL_EPOCHS          = 400      # MAE braucht 400+ Epochs
CHECKPOINT_INTERVAL = 20

# Regression
REG_EPOCHS          = 100
REG_BATCH_SIZE      = 8
LR_REG              = 1e-4
FREEZE_ENCODER      = True


# ─────────────────────────────────────────────
# Dateien einlesen
# ─────────────────────────────────────────────
def collect_nifti_files(base_dir: str) -> list:
    files = []
    for root, _, filenames in os.walk(base_dir):
        for filename in filenames:
            if filename.endswith(".nii.gz"):
                files.append({"image": os.path.join(root, filename)})
    print(f"  {len(files)} NIfTI-Dateien gefunden.")
    return files


def collect_labeled_files(labels_path: str) -> list:
    if not os.path.exists(labels_path):
        print(f"[Warnung] Labels nicht gefunden: {labels_path}")
        return []
    with open(labels_path) as f:
        label_map = json.load(f)
    files = [{"image": p, "label": float(k)}
             for p, k in label_map.items() if os.path.exists(p)]
    print(f"  {len(files)} labeled Dateien geladen.")
    return files


# ─────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────
def make_preprocessing() -> list:
    """
    Gemeinsames Preprocessing fuer alle Pipelines.
    Geraete-Harmonisierung: Percentile-Clip + Z-Score entfernt
    gerätespezifische HU-Offsets (HiRes 100kV vs Boen 90kVp).
    """
    return [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityRangePercentilesd(
            keys=["image"], lower=1, upper=99,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        NormalizeIntensityd(keys=["image"], nonzero=True),
        ScaleIntensityd(keys=["image"]),
    ]

# Global: ganzer Schaedel resized → Morphologie + Regression
global_transforms = Compose(
    make_preprocessing() + [
        Resized(keys=["image"], spatial_size=ROI_GLOBAL, mode="trilinear"),
        EnsureTyped(keys=["image"]),
    ]
)

# Local: Preprocessing ohne Resize — Crop passiert spaeter
local_transforms = Compose(
    make_preprocessing() + [
        EnsureTyped(keys=["image"]),
    ]
)

# MAE Inferenz: CenterCrop auf ROI_LOCAL
# Reproduzierbar (kein Zufall) und passend zum trainierten pos_embed
mae_infer_transforms = Compose(
    make_preprocessing() + [
        CenterSpatialCropd(keys=["image"], roi_size=ROI_LOCAL),
        EnsureTyped(keys=["image"]),
    ]
)

# RandCrop fuer lokale Views
rand_crop = RandSpatialCropd(
    keys=["image"], roi_size=ROI_LOCAL, random_size=False, random_center=True
)

# ── [7] MAE-spezifische Augmentierungen ───────────────────────
# Keine starken Rotationen — Patch-Grenzen sollen erhalten bleiben
# Starke Intensitaets-Augmentierungen OK — Encoder muss robust sein
aug_mae = Compose([
    RandAffined(
        keys=["image"], prob=0.5,
        rotate_range=(0.15, 0.15, 0.15),   # max ~3 Grad (war 0.5 = 30 Grad)
        translate_range=(3, 3, 3),
        scale_range=(0.05, 0.05, 0.05),
        mode="bilinear", padding_mode="zeros",
    ),
    # Intensitaet stark augmentieren — Geraeteunterschiede simulieren
    RandShiftIntensityd(keys=["image"], offsets=0.15, prob=0.7),
    RandScaleIntensityd(keys=["image"], factors=0.15, prob=0.7),
    RandGaussianNoised(keys=["image"], prob=0.5, std=0.02),
    # Kein CoarseDropout fuer MAE — Masking uebernimmt diese Rolle
])

# Globale Augmentierungen (BYOL)
aug_global = Compose([
    RandAffined(
        keys=["image"], prob=0.7,
        rotate_range=(0.15, 0.15, 0.15),
        translate_range=(6, 6, 6),
        scale_range=(0.1, 0.1, 0.1),
        mode="bilinear", padding_mode="zeros",
    ),
    RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
    RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
    RandGaussianNoised(keys=["image"], prob=0.2, std=0.02),
])

# Lokale Augmentierungen (BYOL)
aug_local = Compose([
    RandAffined(
        keys=["image"], prob=0.8,
        rotate_range=(0.15, 0.15, 0.15),
        translate_range=(4, 4, 4),
        scale_range=(0.15, 0.15, 0.15),
        mode="bilinear", padding_mode="zeros",
    ),
    RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
    RandShiftIntensityd(keys=["image"], offsets=0.15, prob=0.6),
    RandScaleIntensityd(keys=["image"], factors=0.15, prob=0.6),
    RandGaussianNoised(keys=["image"], prob=0.4, std=0.02),
    RandCoarseDropoutd(keys=["image"], holes=3, spatial_size=(8, 8, 8), prob=0.2),
])


def create_views_global(batch: dict):
    x = batch["image"]
    x1_list, x2_list = [], []
    for i in range(x.shape[0]):
        s = {"image": x[i]}
        x1_list.append(aug_global(copy.deepcopy(s))["image"])
        x2_list.append(aug_global(copy.deepcopy(s))["image"])
    return torch.stack(x1_list), torch.stack(x2_list)


def create_views_local(batch: dict):
    x = batch["image"]
    x1_list, x2_list = [], []
    for i in range(x.shape[0]):
        s = {"image": x[i]}
        c1 = rand_crop(copy.deepcopy(s))
        c2 = rand_crop(copy.deepcopy(s))
        x1_list.append(aug_local(c1)["image"])
        x2_list.append(aug_local(c2)["image"])
    return torch.stack(x1_list), torch.stack(x2_list)


# ─────────────────────────────────────────────
# [5] HYBRID CNN + TRANSFORMER ENCODER
#
# CNN-Backbone: lokale Features (Kortikalis, Trabekel)
#   → 3 Faltungsschichten mit InstanceNorm [9]
#   → InstanceNorm stabiler als BatchNorm bei batch_size=4
#
# Transformer: globale Aggregation
#   → lernt Beziehungen zwischen Knochen-Regionen
#   → CLS Token als globales Embedding
#
# Warum Hybrid:
#   Reiner Transformer: gut fuer globale Beziehungen, schlecht fuer lokale Textur
#   Reiner CNN:         gut fuer lokale Textur, schlecht fuer globale Beziehungen
#   Hybrid:             beides — optimal fuer CBCT Knochenstruktur
# ─────────────────────────────────────────────

class HybridMAEEncoder3D(nn.Module):
    """
    Hybrid CNN + Transformer Encoder fuer 3D CBCT MAE.

    Architektur:
      Conv3d(1→64)  + InstanceNorm + GELU  → lokale low-level Features
      Conv3d(64→128) + InstanceNorm + GELU → lokale mid-level Features
      Conv3d(128→embed_dim, stride=patch)  → Patch Embeddings
      Transformer (6 Layer)                → globale Aggregation
      CLS Token                            → globales Embedding
    """
    def __init__(self, embed_dim: int = 512, patch_size: int = MAE_PATCH_SIZE):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim  = embed_dim

        # [9] CNN-Backbone mit InstanceNorm
        # InstanceNorm normalisiert pro Sample — stabiler bei kleinen Batches
        # GELU statt ReLU — glattere Gradienten, besser fuer Transformer-Kombination
        self.cnn_backbone = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64, affine=True),
            nn.GELU(),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.InstanceNorm3d(128, affine=True),
            nn.GELU(),
            # Letzter Conv stride=patch_size → Patch Embeddings
            nn.Conv3d(128, embed_dim, kernel_size=patch_size, stride=patch_size),
        )

        # Lernbares Positional Encoding
        # Groesse dynamisch — passt zu beliebigem ROI_LOCAL
        max_patches = (max(ROI_LOCAL) // patch_size) ** 3
        self.pos_embed = nn.Parameter(torch.randn(1, max_patches, embed_dim) * 0.02)

        # CLS Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # [5] Transformer — 6 Layer (war 4)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8,
            dim_feedforward=2048, dropout=0.1,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.norm = nn.LayerNorm(embed_dim)

    def _get_pos_embed(self, N: int, device) -> torch.Tensor:
        """
        Interpoliert pos_embed auf aktuelle Patch-Anzahl N.
        Verhindert den 4096 vs 1728 Mismatch-Fehler bei unterschiedlichen ROI-Groessen.
        """
        if N == self.pos_embed.shape[1]:
            return self.pos_embed
        pos = self.pos_embed.transpose(1, 2)           # [1, E, N_orig]
        pos = F.interpolate(pos, size=N, mode="linear", align_corners=False)
        return pos.transpose(1, 2)                     # [1, N_new, E]

    def forward(self, x: torch.Tensor, mask_ratio: float = 0.0,
                use_block_masking: bool = True):
        """
        Args:
            x:                [B, 1, D, H, W]
            mask_ratio:       Anteil maskierter Patches
            use_block_masking: [6] Block-Masking statt zufaelligem Masking

        Returns:
            cls_out:     [B, E]  — CLS Token Embedding
            patch_out:   [B, N_visible, E]
            mask:        [B, N] Bool — True=maskiert
            ids_restore: [B, N]
            grid_shape:  (D', H', W')
        """
        # CNN-Features extrahieren
        feat = self.cnn_backbone(x)                    # [B, E, D', H', W']
        B, E, D, H, W = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)       # [B, N, E]
        N = tokens.shape[1]

        # Positional Encoding (interpoliert fuer beliebige Groesse)
        tokens = tokens + self._get_pos_embed(N, x.device)

        if mask_ratio > 0:
            if use_block_masking:
                # [6] Block-Masking: zusammenhaengende 3D-Bloecke maskieren
                mask, ids_shuffle = _block_masking_3d(B, D, H, W, mask_ratio, x.device)
                ids_restore = torch.argsort(ids_shuffle, dim=1)
            else:
                # Zufaelliges Masking (Fallback)
                noise = torch.rand(B, N, device=x.device)
                ids_shuffle = torch.argsort(noise, dim=1)
                ids_restore = torch.argsort(ids_shuffle, dim=1)
                mask = torch.ones(B, N, device=x.device, dtype=torch.bool)
                keep = int(N * (1 - mask_ratio))
                mask.scatter_(1, ids_shuffle[:, :keep], False)

            # Nur sichtbare Tokens behalten
            visible_mask = ~mask                       # True = sichtbar
            ids_keep = torch.where(visible_mask[0])[0] # gleich fuer alle im Batch
            tokens_visible = tokens[:, ids_keep, :]

        else:
            tokens_visible = tokens
            mask        = torch.zeros(B, N, device=x.device, dtype=torch.bool)
            ids_restore = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)

        # CLS Token prependen + Transformer
        cls = self.cls_token.expand(B, -1, -1)
        out = self.transformer(torch.cat([cls, tokens_visible], dim=1))
        out = self.norm(out)

        return out[:, 0], out[:, 1:], mask, ids_restore, (D, H, W)


def _block_masking_3d(B: int, D: int, H: int, W: int,
                      mask_ratio: float, device) -> tuple:
    """
    [6] Block-Masking: maskiert zusammenhaengende 3D-Bloecke.

    Warum besser als zufaelliges Masking:
      Zufaelliges Masking: Encoder kann fehlende Patches durch
        lokale Interpolation der Nachbarn rekonstruieren
        → lernt keine echte Struktur
      Block-Masking: gesamte Region fehlt
        → Encoder muss globales Strukturwissen nutzen
        → lernt anatomische Zusammenhaenge (Kiefer, Zahnreihe)

    Returns:
        mask:        [B, N] Bool — True=maskiert
        ids_shuffle: [B, N] sortierte Indices fuer ids_restore
    """
    N = D * H * W
    num_masked = int(N * mask_ratio)
    mask = torch.zeros(B, N, device=device, dtype=torch.bool)

    for b in range(B):
        # Zufaelliger Block-Mittelpunkt
        d0 = torch.randint(0, D, (1,)).item()
        h0 = torch.randint(0, H, (1,)).item()
        w0 = torch.randint(0, W, (1,)).item()

        # Block-Radius aus Ziel-Anzahl maskierter Patches
        r = max(1, int((num_masked * 3 / (4 * 3.14159)) ** (1/3)))

        for d in range(max(0, d0 - r), min(D, d0 + r + 1)):
            for h in range(max(0, h0 - r), min(H, h0 + r + 1)):
                for w in range(max(0, w0 - r), min(W, w0 + r + 1)):
                    idx = d * H * W + h * W + w
                    if idx < N:
                        mask[b, idx] = True

        # Falls zu wenige maskiert: Rest zufaellig auffuellen
        n_masked = mask[b].sum().item()
        if n_masked < num_masked:
            unmasked = torch.where(~mask[b])[0]
            extra = unmasked[torch.randperm(len(unmasked))[:num_masked - n_masked]]
            mask[b, extra] = True

    # ids_shuffle: maskierte Indices zuerst (fuer ids_restore Kompatibilitaet)
    ids_shuffle = torch.argsort(mask.float(), dim=1, descending=True)
    return mask, ids_shuffle


class MAEDecoder3D(nn.Module):
    """
    Leichtgewichtiger Decoder.
    Rekonstruiert maskierte Patches aus Encoder-Output.
    """
    def __init__(self, embed_dim: int = 512, decoder_dim: int = 128,
                 patch_size: int = MAE_PATCH_SIZE):
        super().__init__()
        self.patch_size = patch_size

        self.embed      = nn.Linear(embed_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.randn(1, 1, decoder_dim) * 0.02)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim, nhead=8, dim_feedforward=1024,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=1)
        self.norm = nn.LayerNorm(decoder_dim)
        self.pred = nn.Linear(decoder_dim, patch_size ** 3)

    def forward(self, patch_tokens: torch.Tensor, mask: torch.Tensor,
                ids_restore: torch.Tensor, grid_shape: tuple) -> torch.Tensor:
        B = patch_tokens.shape[0]
        N = ids_restore.shape[1]

        x = self.embed(patch_tokens)
        mask_tokens = self.mask_token.expand(B, N - x.shape[1], -1)
        x_full = torch.cat([x, mask_tokens], dim=1)

        x_full = torch.gather(
            x_full, 1,
            ids_restore.unsqueeze(-1).expand(-1, -1, x_full.shape[-1])
        )
        x_full = self.transformer(x_full)
        x_full = self.norm(x_full)
        return self.pred(x_full)                       # [B, N, patch^3]


class MAEModel(nn.Module):
    def __init__(self, embed_dim: int = 512, decoder_dim: int = 128,
                 patch_size: int = MAE_PATCH_SIZE):
        super().__init__()
        self.encoder = HybridMAEEncoder3D(embed_dim=embed_dim, patch_size=patch_size)
        self.decoder = MAEDecoder3D(embed_dim=embed_dim, decoder_dim=decoder_dim,
                                    patch_size=patch_size)
        self.patch_size = patch_size

        # Gewichtung für Zusatzverluste
        self.lambda_var = 0.1
        self.lambda_cov = 0.05

    def patchify_target(self, x: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        B, C, D, H, W = x.shape
        x = x.reshape(B, C, D//p, p, H//p, p, W//p, p)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)
        return x.reshape(B, -1, C * p * p * p)

    def forward(self, x: torch.Tensor,
                mask_ratio: float = MAE_MASK_RATIO,
                use_block_masking: bool = True):

        cls_emb, patch_tokens, mask, ids_restore, grid_shape = \
            self.encoder(x, mask_ratio=mask_ratio,
                         use_block_masking=use_block_masking)

        pred   = self.decoder(patch_tokens, mask, ids_restore, grid_shape)
        target = self.patchify_target(x)

        # ───────────────
        # MAE Reconstruction Loss
        # ───────────────
        pixel_loss      = (pred - target) ** 2
        pixel_loss_mean = pixel_loss.mean(dim=-1)

        patch_intensity = target.abs().mean(dim=-1)
        bone_weights    = 1.0 + (BONE_WEIGHT_FACTOR - 1.0) * patch_intensity

        weighted_loss = pixel_loss_mean * bone_weights

        recon_loss = (weighted_loss * mask.float()).sum() / \
                     ((mask.float() * bone_weights).sum() + 1e-8)

        # ───────────────
        # 🧠 CLS Regularization (ANTI-COLLAPSE)
        # ───────────────

        # Centering
        z = cls_emb - cls_emb.mean(dim=0)

        # Variance Loss (verhindert collapse)
        std = torch.sqrt(z.var(dim=0) + 1e-4)
        var_loss = torch.mean(F.relu(1.0 - std))

        # Covariance Loss (Dekorrelation)
        B, D = z.shape
        cov = (z.T @ z) / (B - 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        cov_loss = (off_diag ** 2).sum() / D

        # ───────────────
        # Final Loss
        # ───────────────
        loss = recon_loss \
             + self.lambda_var * var_loss \
             + self.lambda_cov * cov_loss

        return loss, cls_emb

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        cls_emb, _, _, _, _ = self.encoder(x, mask_ratio=0.0)
        return cls_emb


# ─────────────────────────────────────────────
# [2] Warmup + Cosine LR Schedule
# ─────────────────────────────────────────────
def get_warmup_cosine_scheduler(optimizer, warmup_epochs: int,
                                 total_epochs: int, min_lr: float = 1e-6):
    """
    Lineares Warmup dann Cosine Decay — MAE-Standard (He et al. 2022).

    Warum Warmup:
      Erste Epochs: Gewichte zufaellig → hohe LR → instabile Gradienten
      Warmup: LR steigt langsam → stabiler Start
      Nach Warmup: Cosine Decay → langsames Einpendeln

    Zielwerte fuer LR_SSL=1.5e-4, warmup=20:
      Epoch  0: LR = 0
      Epoch 20: LR = 1.5e-4 (voll)
      Epoch 200: LR = ~7e-5
      Epoch 400: LR = 1e-6 (min)
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine   = 0.5 * (1 + np.cos(np.pi * progress))
        scaled   = min_lr / LR_SSL + (1 - min_lr / LR_SSL) * cosine
        return scaled

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────
# Multi-Scale BYOL (Alternative zu MAE)
# ─────────────────────────────────────────────
class GlobalEncoder(nn.Module):
    def __init__(self, out_dim: int = 512):
        super().__init__()
        self.encoder = resnet.ResNet(
            block="basic", layers=[2, 2, 2, 2],
            block_inplanes=[64, 128, 256, 512],
            spatial_dims=3, n_input_channels=1, num_classes=out_dim,
        )
    def forward(self, x): return self.encoder(x)


class LocalEncoder(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.encoder = resnet.ResNet(
            block="basic", layers=[2, 2, 2, 2],
            block_inplanes=[32, 64, 128, 256],
            spatial_dims=3, n_input_channels=1, num_classes=out_dim,
        )
    def forward(self, x): return self.encoder(x)


class MultiScaleSSLModel(nn.Module):
    def __init__(self, global_dim=512, local_dim=256, proj_dim=256):
        super().__init__()
        self.global_encoder = GlobalEncoder(out_dim=global_dim)
        self.local_encoder  = LocalEncoder(out_dim=local_dim)
        fused_dim = global_dim + local_dim
        self.projector = nn.Sequential(
            nn.Linear(fused_dim, fused_dim), nn.BatchNorm1d(fused_dim), nn.ReLU(),
            nn.Linear(fused_dim, global_dim), nn.BatchNorm1d(global_dim), nn.ReLU(),
            nn.Linear(global_dim, proj_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, proj_dim // 2), nn.BatchNorm1d(proj_dim // 2), nn.ReLU(),
            nn.Linear(proj_dim // 2, proj_dim),
        )

    def forward(self, x_global, x_local):
        h = torch.cat([self.global_encoder(x_global), self.local_encoder(x_local)], dim=1)
        z = self.projector(h)
        return z, self.predictor(z)

    def encode(self, x_global, x_local):
        return torch.cat([self.global_encoder(x_global), self.local_encoder(x_local)], dim=1)


def _byol_loss(p, z):
    return 2 - 2 * (F.normalize(p, dim=-1) * F.normalize(z, dim=-1)).sum(dim=-1).mean()


class MultiScaleBYOL:
    def __init__(self, model, tau=BYOL_TAU):
        self.online = model
        self.target = copy.deepcopy(model)
        for p in self.target.parameters(): p.requires_grad_(False)
        self.tau = tau

    @torch.no_grad()
    def update_target(self):
        for op, tp in zip(self.online.parameters(), self.target.parameters()):
            tp.data = self.tau * tp.data + (1 - self.tau) * op.data

    def compute_loss(self, xg1, xg2, xl1, xl2):
        _, p1 = self.online(xg1, xl1)
        _, p2 = self.online(xg2, xl2)
        with torch.no_grad():
            z1, _ = self.target(xg1, xl1)
            z2, _ = self.target(xg2, xl2)
        return (_byol_loss(p1, z2) + _byol_loss(p2, z1)) / 2


# ─────────────────────────────────────────────
# [8] Training-Monitoring
# ─────────────────────────────────────────────
def monitor_embedding_quality(model, sample_files: list, epoch: int,
                               ssl_mode: str, output_dir: str) -> float:
    """
    Schnell-Evaluierung der Embedding-Qualitaet waehrend Training.

    Berechnet:
      - Mittlere Cosinus-Aehnlichkeit (Collapse-Indikator)
      - Anteil aktiver Dimensionen
      - Loggt Warnung wenn Collapse droht

    Returns:
        mean_sim: Mittlere Cosinus-Aehnlichkeit (< 0.95 = gut)
    """
    model.eval()
    embeddings = []
    n_samples  = min(20, len(sample_files))

    with torch.no_grad():
        for entry in sample_files[:n_samples]:
            fpath = entry["image"]
            try:
                if ssl_mode == "mae":
                    sample = mae_infer_transforms({"image": fpath})
                    x = sample["image"].unsqueeze(0).to(device)
                    emb = model.encode(x)
                else:
                    sg = global_transforms({"image": fpath})
                    sl = local_transforms({"image": fpath})
                    xg = sg["image"].unsqueeze(0).to(device)
                    xl_full = sl["image"]
                    xl = rand_crop({"image": xl_full})["image"].unsqueeze(0).to(device)
                    emb = model.encode(xg, xl)
                embeddings.append(emb.cpu().numpy().squeeze())
            except Exception:
                continue

    if len(embeddings) < 3:
        model.train()
        return 1.0

    embeddings = np.array(embeddings)
    emb_norm   = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    sims = [np.dot(emb_norm[i], emb_norm[j])
            for i in range(len(emb_norm)) for j in range(i+1, len(emb_norm))]

    mean_sim  = float(np.mean(sims))
    mean_std  = float(embeddings.std(axis=0).mean())
    active    = int((embeddings.std(axis=0) > 0.05).sum())
    total_dim = embeddings.shape[1]

    status = "OK"
    if mean_sim > 0.999:
        status = "COLLAPSE"
    elif mean_sim > 0.97:
        status = "WARNUNG"

    print(f"  [Monitor Ep{epoch:03d}] "
          f"CosSim={mean_sim:.4f} | Std={mean_std:.4f} | "
          f"Aktiv={active}/{total_dim} | {status}")

    # Zielwerte:
    # Epoch  50:  CosSim ~0.97, Aktiv ~100
    # Epoch 100:  CosSim ~0.93, Aktiv ~200
    # Epoch 200:  CosSim ~0.85, Aktiv ~300
    # Epoch 400:  CosSim ~0.65, Aktiv ~420

    if mean_sim > 0.999:
        print("  [WARNUNG] Collapse droht! Empfehlung: LR reduzieren oder Training stoppen.")

    model.train()
    return mean_sim


# ─────────────────────────────────────────────
# Checkpoint Utilities
# ─────────────────────────────────────────────
def save_checkpoint(epoch, model, optimizer, loss, path):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "loss": loss,
    }, path)
    print(f"  Checkpoint: {path}  (Epoch {epoch})")


def load_checkpoint(path, model, optimizer) -> int:
    if not os.path.exists(path):
        return 0
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    print(f"  Checkpoint geladen → Epoch {ckpt['epoch'] + 1}")
    return ckpt["epoch"] + 1


# ─────────────────────────────────────────────
# PHASE 1: SSL Training
# ─────────────────────────────────────────────
def run_ssl_training(ssl_mode: str = "mae"):
    print("\n" + "=" * 60)
    print(f"PHASE 1 — SSL Pretraining ({ssl_mode.upper()})")
    print(f"  ROI Global      : {ROI_GLOBAL}")
    print(f"  ROI Local       : {ROI_LOCAL}")
    print(f"  Patch Size      : {MAE_PATCH_SIZE}")
    print(f"  Patches (local) : {(ROI_LOCAL[0]//MAE_PATCH_SIZE)**3}")
    print(f"  Batch Size      : {SSL_BATCH_SIZE} (effektiv {SSL_BATCH_SIZE * ACCUMULATION_STEPS})")
    print(f"  Epochs          : {SSL_EPOCHS}  (Warmup: {WARMUP_EPOCHS})")
    print(f"  LR              : {LR_SSL}")
    print("=" * 60)

    train_files = collect_nifti_files(BASE_DIR_UNLABELED)
    if not train_files:
        print(f"[Fehler] Keine Dateien: {BASE_DIR_UNLABELED}")
        return

    os.makedirs(CACHE_DIR_GLOBAL, exist_ok=True)
    os.makedirs(CACHE_DIR_LOCAL,  exist_ok=True)
    os.makedirs(MONITOR_DIR,      exist_ok=True)

    local_dataset = PersistentDataset(
        data=train_files, transform=local_transforms,
        cache_dir=CACHE_DIR_LOCAL,
    )
    local_loader = DataLoader(
        local_dataset,
        batch_size=SSL_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=default_collate,
        pin_memory=False,
        persistent_workers=True,
        drop_last=True,
    )

    scaler = torch.amp.GradScaler(device="cuda")

    if ssl_mode == "mae":
        mae_model = MAEModel(embed_dim=512, decoder_dim=128).to(device)

        # [3] AdamW mit MAE-optimalen Parametern
        optimizer = torch.optim.AdamW(
            mae_model.parameters(),
            lr=LR_SSL,
            weight_decay=0.01,          # reduziert fuer kleinen Datensatz (n=133)
            betas=(0.9, 0.95),          # MAE-Standard betas
        )

        # [2] Warmup + Cosine LR Schedule
        scheduler = get_warmup_cosine_scheduler(
            optimizer, warmup_epochs=WARMUP_EPOCHS, total_epochs=SSL_EPOCHS
        )

        # changes:
        #  Load checkpoint
        # start_epoch = load_checkpoint(CHECKPOINT_SSL, mae_model, optimizer)

        # Start fresh
        start_epoch=0
        
        # Scheduler auf aktuellen Epoch bringen nach Resume
        for _ in range(start_epoch):
            scheduler.step()

        for epoch in range(start_epoch, SSL_EPOCHS):
            mae_model.train()
            epoch_loss = 0.0
            optimizer.zero_grad()

            for step, local_batch in enumerate(local_loader):
                x_local = local_batch["image"]

                # [7] MAE-Augmentierung + RandCrop auf lokales Bild
                x_list = []
                for i in range(x_local.shape[0]):
                    s = {"image": x_local[i]}
                    s = rand_crop(s)             # zufaelliger 64^3 Crop
                    s = aug_mae(s)               # MAE-spezifische Augmentierung
                    x_list.append(s["image"])
                x = torch.stack(x_list).to(device, non_blocking=True)

                with torch.amp.autocast(device_type="cuda"):
                    # [6] Block-Masking aktiviert
                    loss, _ = mae_model(x, mask_ratio=MAE_MASK_RATIO,
                                        use_block_masking=True)
                    loss = loss / ACCUMULATION_STEPS

                scaler.scale(loss).backward()

                if (step + 1) % ACCUMULATION_STEPS == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                epoch_loss += loss.item() * ACCUMULATION_STEPS

            scheduler.step()
            avg_loss   = epoch_loss / len(local_loader)
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:03d}/{SSL_EPOCHS} | "
                  f"MAE Loss: {avg_loss:.6f} | LR: {current_lr:.2e}")

            if torch.isnan(torch.tensor(avg_loss)):
                print("  [Fehler] NaN — Training abgebrochen.")
                save_checkpoint(epoch, mae_model, optimizer, avg_loss, CHECKPOINT_SSL + str(epoch) + '.pt')
                return

            # [8] Training-Monitoring
            if epoch % MONITOR_INTERVAL == 0 and epoch > 0:
                monitor_embedding_quality(
                    mae_model, train_files, epoch, "mae", MONITOR_DIR
                )

            if epoch % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(epoch, mae_model, optimizer, avg_loss, CHECKPOINT_SSL)

        torch.save(mae_model.state_dict(), MODEL_SSL_FINAL)
        print(f"\nMAE Pretraining abgeschlossen: {MODEL_SSL_FINAL}")

    else:
        # Multi-Scale BYOL
        global_dataset = PersistentDataset(
            data=train_files, transform=global_transforms,
            cache_dir=CACHE_DIR_GLOBAL,
        )
        global_loader = DataLoader(
            global_dataset,
            batch_size=SSL_BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=False,
            collate_fn=default_collate,
            persistent_workers=True,
            drop_last=True,
        )

        ms_model = MultiScaleSSLModel().to(device)
        byol     = MultiScaleBYOL(ms_model)
        optimizer = torch.optim.AdamW(
            ms_model.parameters(), lr=LR_SSL,
            weight_decay=0.01, betas=(0.9, 0.95),
        )
        scheduler = get_warmup_cosine_scheduler(
            optimizer, warmup_epochs=WARMUP_EPOCHS, total_epochs=SSL_EPOCHS
        )
        # changes
        # Load checkpoint
        start_epoch = load_checkpoint('20_epochs_mttv5_2_0_byol.pt', ms_model, optimizer)

        # Start fresh
        #start_epoch = 0
        
        for _ in range(start_epoch):
            scheduler.step()

        for epoch in range(start_epoch, SSL_EPOCHS):
            ms_model.train()
            epoch_loss = 0.0
            optimizer.zero_grad()

            for step, (global_batch, local_batch) in enumerate(
                zip(global_loader, local_loader)
            ):
                xg1, xg2 = create_views_global(global_batch)
                xl1, xl2 = create_views_local(local_batch)
                xg1, xg2 = xg1.to(device), xg2.to(device)
                xl1, xl2 = xl1.to(device), xl2.to(device)

                with torch.amp.autocast(device_type="cuda"):
                    loss = byol.compute_loss(xg1, xg2, xl1, xl2) / ACCUMULATION_STEPS

                scaler.scale(loss).backward()

                if (step + 1) % ACCUMULATION_STEPS == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    byol.update_target()

                epoch_loss += loss.item() * ACCUMULATION_STEPS

            scheduler.step()
            avg_loss = epoch_loss / len(global_loader)
            print(f"  Epoch {epoch:03d}/{SSL_EPOCHS} | BYOL Loss: {avg_loss:.6f}")

            if torch.isnan(torch.tensor(avg_loss)):
                print("  [Fehler] NaN — Training abgebrochen.")
                save_checkpoint(epoch, ms_model, optimizer, avg_loss, CHECKPOINT_SSL + str(epoch) + '.pt')
                return

            if epoch % MONITOR_INTERVAL == 0 and epoch > 0:
                monitor_embedding_quality(
                    ms_model, train_files, epoch, "byol", MONITOR_DIR
                )

            if epoch % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(epoch, ms_model, optimizer, avg_loss, CHECKPOINT_SSL)

        torch.save(ms_model.state_dict(), MODEL_SSL_FINAL)
        print(f"\nMulti-Scale BYOL abgeschlossen: {MODEL_SSL_FINAL}")


# ─────────────────────────────────────────────
# PHASE 2: Downstream Kaukraft-Regression
# ─────────────────────────────────────────────
class KaukraftRegressor(nn.Module):
    """
    Kaukraft-Regressor auf Basis des SSL-Encoders.
    MAE:  input_dim=512  (CLS Token)
    BYOL: input_dim=768  (Fused Embedding)
    """
    def __init__(self, encoder, input_dim: int = 512, freeze: bool = FREEZE_ENCODER):
        super().__init__()
        self.encoder  = encoder
        self.freeze   = freeze
        self.ssl_mode = "mae"

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            print("  Encoder eingefroren.")
        else:
            print("  Encoder wird mittrainiert (Fine-Tuning).")

        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64),        nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x_global, x_local=None):
        ctx = torch.no_grad() if self.freeze else torch.enable_grad()
        with ctx:
            if self.ssl_mode == "mae":
                features = self.encoder.encode(x_global)
            else:
                features = self.encoder.encode(x_global, x_local)
        return self.regressor(features).squeeze(1)


class LabeledCBCTDataset(torch.utils.data.Dataset):
    def __init__(self, files, transform):
        self.files = files
        self.transform = transform

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        entry  = self.files[idx]
        sample = self.transform({"image": entry["image"]})
        return sample["image"], torch.tensor(entry["label"], dtype=torch.float32)


def run_regression_training(ssl_mode: str = "mae"):
    print("\n" + "=" * 60)
    print("PHASE 2 — Kaukraft-Regression")
    print(f"  SSL Modus : {ssl_mode.upper()}")
    print(f"  Frozen    : {FREEZE_ENCODER}")
    print("=" * 60)

    if not os.path.exists(MODEL_SSL_FINAL):
        print(f"[Fehler] SSL-Modell fehlt: {MODEL_SSL_FINAL}")
        return

    labeled = collect_labeled_files(LABELS_JSON)
    if not labeled:
        print("[Fehler] Keine Labels.")
        return

    reg_dataset = LabeledCBCTDataset(labeled, global_transforms)
    reg_loader  = DataLoader(
        reg_dataset,
        batch_size=REG_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=default_collate,
        pin_memory=False,
        persistent_workers=False,
    )

    raw = torch.load(MODEL_SSL_FINAL, map_location=device)
    state = raw["model_state"] if "model_state" in raw else raw

    if ssl_mode == "mae":
        ssl_model = MAEModel(embed_dim=512).to(device)
        ssl_model.load_state_dict(state)
        regressor = KaukraftRegressor(ssl_model, input_dim=512).to(device)
        regressor.ssl_mode = "mae"
    else:
        ssl_model = MultiScaleSSLModel().to(device)
        ssl_model.load_state_dict(state)
        regressor = KaukraftRegressor(ssl_model, input_dim=768).to(device)
        regressor.ssl_mode = "byol"

    criterion = nn.HuberLoss(delta=50.0)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, regressor.parameters()), lr=LR_REG
    )
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    scaler_reg = torch.amp.GradScaler(device="cuda")
    start      = load_checkpoint(CHECKPOINT_REG, regressor, optimizer)

    for epoch in range(start, REG_EPOCHS):
        regressor.train()
        epoch_loss = 0.0

        for images, labels in reg_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda"):
                loss = criterion(regressor(images), labels)

            scaler_reg.scale(loss).backward()
            scaler_reg.step(optimizer)
            scaler_reg.update()
            epoch_loss += loss.item()

        avg = epoch_loss / len(reg_loader)
        scheduler.step(avg)
        lr  = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch:03d}/{REG_EPOCHS} | HuberLoss: {avg:.2f} N | LR: {lr:.2e}")

        if torch.isnan(torch.tensor(avg)):
            print("  [Fehler] NaN — abgebrochen.")
            save_checkpoint(epoch, regressor, optimizer, avg, CHECKPOINT_REG)
            return

        if epoch % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(epoch, regressor, optimizer, avg, CHECKPOINT_REG)

    torch.save(regressor.state_dict(), MODEL_REG_FINAL)
    print(f"\nRegression abgeschlossen: {MODEL_REG_FINAL}")


# ─────────────────────────────────────────────
# Inferenz
# ─────────────────────────────────────────────
def predict_kaukraft(scan_path: str, ssl_mode: str = "mae") -> float:
    if not os.path.exists(MODEL_REG_FINAL):
        print(f"[Fehler] {MODEL_REG_FINAL} nicht gefunden.")
        return -1.0

    if ssl_mode == "mae":
        ssl_model = MAEModel(embed_dim=512).to(device)
        reg_model = KaukraftRegressor(ssl_model, input_dim=512).to(device)
    else:
        ssl_model = MultiScaleSSLModel().to(device)
        reg_model = KaukraftRegressor(ssl_model, input_dim=768).to(device)

    reg_model.ssl_mode = ssl_mode
    raw = torch.load(MODEL_REG_FINAL, map_location=device)
    reg_model.load_state_dict(raw["model_state"] if "model_state" in raw else raw)
    reg_model.eval()

    # MAE: CenterCrop Inferenz
    transform = mae_infer_transforms if ssl_mode == "mae" else global_transforms
    sample  = transform({"image": scan_path})
    image   = sample["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        kaukraft = reg_model(image).item()

    print(f"Scan             : {os.path.basename(scan_path)}")
    print(f"Kaukraft (pred.) : {kaukraft:.1f} N")
    return kaukraft


# ─────────────────────────────────────────────
# Einstiegspunkt
# ─────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    print(f"\nModus: {args.mode.upper()}  |  SSL: {args.ssl_mode.upper()}")

    if args.mode == "ssl":
        run_ssl_training(args.ssl_mode)
    elif args.mode == "regression":
        run_regression_training(args.ssl_mode)
    elif args.mode == "both":
        run_ssl_training(args.ssl_mode)
        run_regression_training(args.ssl_mode)