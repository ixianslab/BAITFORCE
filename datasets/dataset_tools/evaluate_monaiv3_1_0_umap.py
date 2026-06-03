"""
evaluate_monaiv3_0_0.py

Evaluierung des SSL-Modells fuer CBCT Kaukraft-Regression

NEU: UMAP-Analyse zusaetzlich zu PCA und t-SNE
  - UMAP ist schneller als t-SNE bei grossen Datensaetzen
  - UMAP bewahrt globale Struktur besser als t-SNE
  - Drei Reduktionsmethoden im direkten Vergleich

Ausfuehren:
    python evaluate_monaiv3_0_0.py --mode visual   --ssl_mode byol
    python evaluate_monaiv3_0_0.py --mode visual   --ssl_mode mae
    python evaluate_monaiv3_0_0.py --mode regression --ssl_mode byol
    python evaluate_monaiv3_0_0.py --mode all      --ssl_mode byol
    python evaluate_monaiv3_0_0.py --mode visual   --ssl_mode byol --annotate_all
    python evaluate_monaiv3_0_0.py --mode visual   --ssl_mode byol --no_umap

Installation UMAP:
    pip install umap-learn

Interaktive Plots:
    evaluation/pca_interactive.html
    evaluation/tsne_interactive.html
    evaluation/umap_interactive.html
"""

import os
import sys
import json
import argparse
import numpy as np
import traceback
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from monai_transform_trainv5_2_0 import (
    MAEModel,
    MultiScaleSSLModel,
    KaukraftRegressor,
    global_transforms,
    local_transforms,
    make_preprocessing,
    #center_crop_local,
    rand_crop,
    ROI_LOCAL,
    ROI_GLOBAL,
    MAE_PATCH_SIZE,
)
from monai.transforms import Compose, CenterSpatialCropd, ToTensord, EnsureTyped

center_crop_local = Compose([
    CenterSpatialCropd(keys=["image"], roi_size=ROI_LOCAL),
    EnsureTyped(keys=["image"]),
])

matplotlib.use("Agg")

# ─────────────────────────────────────────────
# Konfiguration
# ─────────────────────────────────────────────
METADATA_JSON        = "C:/Users/zak/project_chewing_gum/current_chewing_gum/metadata_pmcanalseg.json"
MODEL_SSL            = "./runpod_checkpoints/180_checkpoint_ssl.pt"
MODEL_REG            = "./regression_model_final.pt"
OUTPUT_DIR           = "./evaluation"

TSNE_PERPLEXITY      = 30
TSNE_ITERATIONS      = 2000
RANDOM_SEED          = 42
OUTLIER_STD_THRESHOLD = 2.5

# UMAP Parameter
UMAP_N_NEIGHBORS     = 15     # Lokale Nachbarschaft — klein=lokal, gross=global
UMAP_MIN_DIST        = 0.1    # Mindestabstand im Embedding — klein=dichte Cluster
UMAP_METRIC          = "cosine"  # cosine fuer normalisierte Embeddings besser als euclidean

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ─────────────────────────────────────────────
# Argumente
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="SSL Modell Evaluierung")
    parser.add_argument("--mode",      choices=["visual", "regression", "all"], default="visual")
    parser.add_argument("--ssl_mode",  choices=["mae", "byol"], default="byol")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--annotate_all", action="store_true",
                        help="Alle Dateinummern annotieren (statt nur Ausreisser)")
    parser.add_argument("--no_umap", action="store_true",
                        help="UMAP ueberspringen (falls umap-learn nicht installiert)")
    return parser.parse_args()


# ─────────────────────────────────────────────
# UMAP verfuegbar?
# ─────────────────────────────────────────────
def check_umap_available() -> bool:
    try:
        import umap
        return True
    except ImportError:
        print("  [Info] umap-learn nicht installiert.")
        print("         pip install umap-learn")
        print("         UMAP wird uebersprungen.")
        return False


# ─────────────────────────────────────────────
# Modell laden
# ─────────────────────────────────────────────
def load_ssl_model(checkpoint_path: str, ssl_mode: str):
    if not os.path.exists(checkpoint_path):
        print(f"[Fehler] Checkpoint nicht gefunden: {checkpoint_path}")
        sys.exit(1)

    print(f"  Lade Checkpoint: {checkpoint_path}")
    raw = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(raw, dict) and "model_state" in raw:
        state_dict = raw["model_state"]
        loss_val   = raw.get("loss", "?")
        loss_str   = f"{loss_val:.6f}" if isinstance(loss_val, float) else str(loss_val)
        print(f"  Checkpoint-Dict | Epoch: {raw.get('epoch','?')} | Loss: {loss_str}")
    else:
        state_dict = raw
        print("  Direktes state_dict.")

    if ssl_mode == "mae":
        model     = MAEModel(embed_dim=512, decoder_dim=256).to(device)
        embed_dim = 512
    else:
        model     = MultiScaleSSLModel(global_dim=512, local_dim=256, proj_dim=256).to(device)
        embed_dim = 768

    try:
        model.load_state_dict(state_dict, strict=True)
        print(f"  Modell geladen (strict=True) | Embedding-Dim: {embed_dim}")
    except RuntimeError as e:
        print(f"  [Warnung] strict=True fehlgeschlagen: {e}")
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model, embed_dim


# ─────────────────────────────────────────────
# Metadaten laden
# ─────────────────────────────────────────────
def load_metadata(path: str) -> tuple:
    if not os.path.exists(path):
        print(f"[Fehler] metadata.json nicht gefunden: {path}")
        sys.exit(1)

    with open(path) as f:
        raw = json.load(f)

    file_paths, metadata = [], []
    for fpath, meta in raw.items():
        fpath = fpath.replace("\\", "/")
        if os.path.exists(fpath):
            file_paths.append(fpath)
            metadata.append(meta)
        else:
            print(f"  [Warnung] Datei nicht gefunden: {fpath}")

    print(f"  {len(file_paths)} Patienten geladen.")
    return file_paths, metadata


# ─────────────────────────────────────────────
# Embeddings extrahieren
# ─────────────────────────────────────────────
def extract_embeddings(file_paths, model, ssl_mode, embed_dim):
    mae_infer_transforms = None

    if ssl_mode == "mae":
        n_pos            = model.encoder.pos_embed.shape[1]
        patches_per_side = int(round(n_pos ** (1/3)))
        expected_voxels  = patches_per_side * MAE_PATCH_SIZE
        print(f"  MAE pos_embed: {n_pos} ({patches_per_side}^3) → Crop {expected_voxels}^3")
        mae_infer_transforms = Compose(
            make_preprocessing() + [
                CenterSpatialCropd(keys=["image"], roi_size=(expected_voxels,)*3),
                ToTensord(keys=["image"]),
            ]
        )

    model.eval()
    embeddings = []
    failed     = []

    for i, fpath in enumerate(file_paths):
        fpath = fpath.replace("\\", "/")
        print(f"  [{i+1:3d}/{len(file_paths)}] {os.path.basename(fpath)}")

        try:
            with torch.no_grad():
                if ssl_mode == "mae":
                    s   = mae_infer_transforms({"image": fpath})
                    x   = s["image"].unsqueeze(0).to(device)
                    emb = model.encode(x)
                else:
                    # Global: 128^3 resized
                    sg  = global_transforms({"image": fpath})
                    xg  = sg["image"].unsqueeze(0).to(device)
                    # Lokal: CenterCrop fuer reproduzierbare Embeddings
                    sl  = local_transforms({"image": fpath})
                    xl  = center_crop_local({"image": sl["image"]})["image"].unsqueeze(0).to(device)
                    emb = model.encode(xg, xl)

            embeddings.append(emb.cpu().numpy().squeeze())

        except Exception as e:
            print(f"    [Fehler] {os.path.basename(fpath)}: {e}")
            traceback.print_exc()
            embeddings.append(np.zeros(embed_dim))
            failed.append(fpath)

    if failed:
        print(f"\n  {len(failed)} Scans fehlgeschlagen:")
        for f in failed:
            print(f"    {f}")

    return np.array(embeddings)


# ─────────────────────────────────────────────
# Ausreißer erkennen
# ─────────────────────────────────────────────
def find_outliers(coords: np.ndarray,
                  threshold: float = OUTLIER_STD_THRESHOLD) -> np.ndarray:
    center       = coords.mean(axis=0)
    dists        = np.linalg.norm(coords - center, axis=1)
    outlier_mask = dists > (dists.mean() + threshold * dists.std())
    return np.where(outlier_mask)[0]


def save_outlier_report(outlier_indices, file_paths, metadata,
                        coords, projection, output_dir):
    import csv
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"outliers_{projection}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "filename", "age", "sex", "kaukraft",
                         f"{projection}_x", f"{projection}_y"])
        for idx in outlier_indices:
            meta = metadata[idx]
            writer.writerow([
                idx,
                os.path.basename(file_paths[idx]),
                meta.get("age", ""),
                meta.get("sex", ""),
                meta.get("kaukraft", ""),
                f"{coords[idx, 0]:.2f}",
                f"{coords[idx, 1]:.2f}",
            ])
    print(f"  Ausreisser-Report ({len(outlier_indices)}): {path}")


# ─────────────────────────────────────────────
# Interaktiver HTML Plot
# ─────────────────────────────────────────────
def save_interactive_html(coords, file_paths, metadata, title,
                           output_path, x_label="Dim 1", y_label="Dim 2"):
    points_male, points_female, points_unknown = [], [], []

    for i, (fpath, meta) in enumerate(zip(file_paths, metadata)):
        fname    = os.path.basename(fpath)
        file_id  = fname.replace(".nii.gz", "").replace(".nii", "")
        age      = meta.get("age", "N/A")
        sex      = meta.get("sex", "unknown").lower()
        kaukraft = meta.get("kaukraft", "N/A")

        tooltip = (f"<b>{fname}</b><br>"
                   f"ID: {file_id}<br>"
                   f"Alter: {age}<br>"
                   f"Geschlecht: {sex}<br>"
                   f"Kaukraft: {kaukraft} N")

        point = {"x": float(coords[i, 0]), "y": float(coords[i, 1]),
                 "text": tooltip, "label": file_id}

        if sex == "m":
            points_male.append(point)
        elif sex == "f":
            points_female.append(point)
        else:
            points_unknown.append(point)

    def make_trace(points, name, color):
        if not points:
            return ""
        return f"""
        {{
            x: {json.dumps([p["x"] for p in points])},
            y: {json.dumps([p["y"] for p in points])},
            mode: 'markers+text',
            type: 'scatter',
            name: '{name}',
            text: {json.dumps([p["label"] for p in points])},
            textposition: 'top center',
            textfont: {{size: 8, color: '{color}'}},
            hovertext: {json.dumps([p["text"] for p in points])},
            hoverinfo: 'text',
            marker: {{color: '{color}', size: 9, opacity: 0.85,
                      line: {{color: 'white', width: 0.8}}}}
        }}"""

    traces = [t for t in [
        make_trace(points_male,    "Maennlich", "#2196F3"),
        make_trace(points_female,  "Weiblich",  "#E91E63"),
        make_trace(points_unknown, "Unbekannt", "#9E9E9E"),
    ] if t]

    title_esc = title.replace("'", "\\'")

    html = f"""<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; background: #1a1a2e;
               color: #eee; margin: 0; padding: 20px; }}
        h2   {{ text-align: center; color: #90caf9; margin-bottom: 5px; }}
        p.subtitle {{ text-align: center; color: #aaa; font-size: 13px; margin-top: 0; }}
        #plot {{ width: 100%; height: 85vh; }}
        .info-box {{ background: #16213e; border: 1px solid #2196F3;
                    border-radius: 8px; padding: 10px 20px; margin: 10px auto;
                    max-width: 600px; font-size: 13px; color: #aaa; }}
    </style>
</head>
<body>
    <h2>{title}</h2>
    <p class="subtitle">Hover fuer Details &nbsp;|&nbsp; Scrollen zum Zoomen &nbsp;|&nbsp; Doppelklick zum Zuruecksetzen</p>
    <div class="info-box">
        ℹ️ Tooltip: Dateiname · ID · Alter · Geschlecht · Kaukraft<br>
        🔵 Maennlich &nbsp; 🔴 Weiblich &nbsp; n = {len(file_paths)} Patienten
    </div>
    <div id="plot"></div>
    <script>
        Plotly.newPlot('plot',
            [{",".join(traces)}],
            {{
                paper_bgcolor: '#1a1a2e', plot_bgcolor: '#16213e',
                font: {{color: '#eee', family: 'Arial'}},
                xaxis: {{title: '{x_label}', gridcolor: '#2a2a4a', zerolinecolor: '#3a3a6a'}},
                yaxis: {{title: '{y_label}', gridcolor: '#2a2a4a', zerolinecolor: '#3a3a6a'}},
                legend: {{bgcolor: '#16213e', bordercolor: '#2196F3', borderwidth: 1}},
                hoverlabel: {{bgcolor: '#0d0d1a', bordercolor: '#2196F3',
                              font: {{color: '#eee', size: 13}}}},
                hovermode: 'closest'
            }},
            {{responsive: true, displayModeBar: true,
              modeBarButtonsToRemove: ['lasso2d', 'select2d'],
              toImageButtonOptions: {{format: 'png', filename: '{title_esc}',
                                     height: 900, width: 1400, scale: 2}}}}
        );
    </script>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Interaktiver Plot: {output_path}")


# ─────────────────────────────────────────────
# Gemeinsame Plot-Helfer
# ─────────────────────────────────────────────
SEX_COLORS = {"m": "#2196F3", "f": "#E91E63", "unknown": "#9E9E9E"}
SEX_LABELS = {"m": "Maennlich", "f": "Weiblich", "unknown": "Unbekannt"}


def _add_outlier_annotations(ax, coords, file_paths, metadata,
                              outlier_indices, annotate_all=False):
    indices = range(len(file_paths)) if annotate_all else outlier_indices
    for idx in indices:
        fname   = os.path.basename(file_paths[idx])
        file_id = fname.replace(".nii.gz", "").replace(".nii", "")
        x, y    = coords[idx, 0], coords[idx, 1]
        is_out  = idx in outlier_indices

        if is_out:
            ax.scatter(x, y, s=120, facecolors="none",
                       edgecolors="red", linewidths=2.0, zorder=5)
            ax.annotate(file_id, xy=(x, y), xytext=(8, 8),
                        textcoords="offset points", fontsize=7, fontweight="bold",
                        color="red",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                  alpha=0.7, ec="red"), zorder=6)
        elif annotate_all:
            ax.annotate(file_id, xy=(x, y), xytext=(4, 4),
                        textcoords="offset points", fontsize=6,
                        color="#333333", alpha=0.8, zorder=4)


def _scatter_sex(ax, coords, metadata):
    for i, meta in enumerate(metadata):
        sex = meta.get("sex", "unknown").lower()
        ax.scatter(coords[i, 0], coords[i, 1],
                   c=SEX_COLORS.get(sex, "#9E9E9E"),
                   s=50, alpha=0.85, edgecolors="white", linewidths=0.4)
    patches = [mpatches.Patch(color=v, label=SEX_LABELS[k])
               for k, v in SEX_COLORS.items()
               if any(m.get("sex", "unknown").lower() == k for m in metadata)]
    ax.legend(handles=patches, fontsize=9)


def _scatter_age(ax, fig, coords, metadata):
    ages = np.array([m.get("age", np.nan) for m in metadata], dtype=float)
    if not np.all(np.isnan(ages)):
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=ages, cmap="plasma",
                        s=50, alpha=0.85, edgecolors="white", linewidths=0.4)
        fig.colorbar(sc, ax=ax, label="Alter (Jahre)")


def _scatter_force(ax, fig, coords, metadata):
    forces = np.array([m.get("kaukraft", np.nan) for m in metadata], dtype=float)
    if not np.all(np.isnan(forces)):
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=forces, cmap="RdYlGn",
                        s=50, alpha=0.85, edgecolors="white", linewidths=0.4)
        fig.colorbar(sc, ax=ax, label="Kaukraft (N)")
    else:
        ax.text(0.5, 0.5, "Keine Kaukraft-Daten\n(nur SSL-Phase abgeschlossen)",
                transform=ax.transAxes, ha="center", fontsize=10)


# ─────────────────────────────────────────────
# Embedding-Qualität
# ─────────────────────────────────────────────
def check_embedding_collapse(embeddings):
    print("\n--- Embedding Qualitaet ---")
    std_per_dim = embeddings.std(axis=0)
    print(f"  Mittlere Std pro Dimension   : {std_per_dim.mean():.4f}")
    print(f"  Min Std                      : {std_per_dim.min():.4f}")
    print(f"  Dim. mit Std < 0.01          : {(std_per_dim < 0.01).sum()} / {len(std_per_dim)}")

    n = min(len(embeddings), 20)
    emb_norm = embeddings[:n] / (np.linalg.norm(embeddings[:n], axis=1, keepdims=True) + 1e-8)
    sims     = [np.dot(emb_norm[i], emb_norm[j])
                for i in range(n) for j in range(i+1, n)]
    mean_sim = np.mean(sims)
    print(f"  Mittlere Cosinus-Aehnlichkeit: {mean_sim:.4f}")

    if mean_sim > 0.99:
        print("  [WARNUNG] Collapse!")
    elif mean_sim > 0.9:
        print("  [WARNUNG] Partieller Collapse.")
    else:
        print("  [OK] Kein Collapse.")

    active = (std_per_dim > 0.05).sum()
    print(f"  Aktive Dimensionen (Std>0.05): {active} / {len(std_per_dim)}")


# ─────────────────────────────────────────────
# PCA
# ─────────────────────────────────────────────
def plot_pca(embeddings, file_paths, metadata, output_dir, annotate_all=False):
    print("\n--- PCA ---")
    os.makedirs(output_dir, exist_ok=True)

    n_comp   = min(10, len(embeddings) - 1)
    pca_full = PCA(n_components=n_comp, random_state=RANDOM_SEED)
    pca_full.fit(embeddings)
    var_ratio = pca_full.explained_variance_ratio_

    print("  Erklaerte Varianz:")
    cumvar = 0
    for k, v in enumerate(var_ratio):
        cumvar += v
        print(f"    PC{k+1}: {v*100:.1f}%  (kumulativ: {cumvar*100:.1f}%)")

    coords          = pca_full.transform(embeddings)[:, :2]
    var             = var_ratio[:2]
    outlier_indices = find_outliers(coords)
    print(f"  Ausreisser: {len(outlier_indices)}")

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle("PCA der SSL-Embeddings (CBCT Schaedel)", fontsize=14, fontweight="bold")

    _scatter_sex(axes[0], coords, metadata)
    _add_outlier_annotations(axes[0], coords, file_paths, metadata,
                             outlier_indices, annotate_all)
    axes[0].set_title("Geschlecht  (rot = Ausreisser)", fontsize=11)
    axes[0].set_xlabel(f"PC1 ({var[0]*100:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({var[1]*100:.1f}%)")
    axes[0].grid(True, alpha=0.3)

    _scatter_age(axes[1], fig, coords, metadata)
    _add_outlier_annotations(axes[1], coords, file_paths, metadata,
                             outlier_indices, annotate_all)
    axes[1].set_title("Alter  (rot = Ausreisser)", fontsize=11)
    axes[1].set_xlabel(f"PC1 ({var[0]*100:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({var[1]*100:.1f}%)")
    axes[1].grid(True, alpha=0.3)

    _scatter_force(axes[2], fig, coords, metadata)
    _add_outlier_annotations(axes[2], coords, file_paths, metadata,
                             outlier_indices, annotate_all)
    axes[2].set_title("Kaukraft (N)", fontsize=11)
    axes[2].set_xlabel(f"PC1 ({var[0]*100:.1f}%)")
    axes[2].set_ylabel(f"PC2 ({var[1]*100:.1f}%)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    png_path = os.path.join(output_dir, "pca_embeddings.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  PNG: {png_path}")

    _plot_scree(var_ratio, output_dir)

    save_interactive_html(
        coords, file_paths, metadata,
        title=f"PCA der SSL-Embeddings (n={len(file_paths)})",
        output_path=os.path.join(output_dir, "pca_interactive.html"),
        x_label=f"PC1 ({var[0]*100:.1f}%)",
        y_label=f"PC2 ({var[1]*100:.1f}%)",
    )
    save_outlier_report(outlier_indices, file_paths, metadata, coords, "pca", output_dir)


def _plot_scree(var_ratio, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    k = np.arange(1, len(var_ratio) + 1)
    ax.bar(k, var_ratio * 100, color="#2196F3", alpha=0.8, label="Einzeln")
    ax.plot(k, np.cumsum(var_ratio) * 100, "ro-", linewidth=2, label="Kumulativ")
    ax.axhline(y=90, color="gray", linestyle="--", alpha=0.5, label="90% Schwelle")
    ax.set_xlabel("Hauptkomponente")
    ax.set_ylabel("Erklaerte Varianz (%)")
    ax.set_title("Scree-Plot — Varianzverteilung\n"
                 "(Gesund: viele PCs aktiv  |  Problem: PC1 dominiert)", fontsize=11)
    ax.legend(); ax.grid(True, alpha=0.3)
    out_path = os.path.join(output_dir, "pca_scree.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Scree-Plot: {out_path}")


# ─────────────────────────────────────────────
# t-SNE
# ─────────────────────────────────────────────
def plot_tsne(embeddings, file_paths, metadata, output_dir, annotate_all=False):
    print("\n--- t-SNE ---")
    os.makedirs(output_dir, exist_ok=True)

    n          = len(embeddings)
    perplexity = min(TSNE_PERPLEXITY, n - 1, 50)
    print(f"  n={n}  Perplexity={perplexity}  Iterationen={TSNE_ITERATIONS}")

    pre_pca     = PCA(n_components=min(50, n-1), random_state=RANDOM_SEED)
    emb_reduced = pre_pca.fit_transform(embeddings)

    tsne   = TSNE(n_components=2, perplexity=perplexity, max_iter=TSNE_ITERATIONS,
                  random_state=RANDOM_SEED, init="pca", learning_rate="auto")
    coords = tsne.fit_transform(emb_reduced)
    print(f"  KL-Divergenz: {tsne.kl_divergence_:.4f}")

    outlier_indices = find_outliers(coords)
    print(f"  Ausreisser: {len(outlier_indices)}")

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle(f"t-SNE der SSL-Embeddings (n={n})", fontsize=14, fontweight="bold")

    for ax, scatter_fn, title in zip(
        axes,
        [lambda ax: (_scatter_sex(ax, coords, metadata),
                     _add_outlier_annotations(ax, coords, file_paths, metadata,
                                             outlier_indices, annotate_all)),
         lambda ax: (_scatter_age(ax, fig, coords, metadata),
                     _add_outlier_annotations(ax, coords, file_paths, metadata,
                                             outlier_indices, annotate_all)),
         lambda ax: (_scatter_force(ax, fig, coords, metadata),
                     _add_outlier_annotations(ax, coords, file_paths, metadata,
                                             outlier_indices, annotate_all))],
        ["Geschlecht  (rot = Ausreisser)", "Alter  (rot = Ausreisser)", "Kaukraft (N)"]
    ):
        scatter_fn(ax)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    png_path = os.path.join(output_dir, "tsne_embeddings.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  PNG: {png_path}")

    save_interactive_html(
        coords, file_paths, metadata,
        title=f"t-SNE der SSL-Embeddings (n={n})",
        output_path=os.path.join(output_dir, "tsne_interactive.html"),
        x_label="t-SNE 1", y_label="t-SNE 2",
    )
    save_outlier_report(outlier_indices, file_paths, metadata, coords, "tsne", output_dir)


# ─────────────────────────────────────────────
# UMAP
# ─────────────────────────────────────────────
def plot_umap(embeddings, file_paths, metadata, output_dir, annotate_all=False):
    """
    UMAP-Analyse der SSL-Embeddings.

    Vorteile gegenüber t-SNE:
      - Schneller bei grossen Datensaetzen (O(n) statt O(n^2))
      - Bewahrt globale Struktur besser
      - Deterministischer mit random_state
      - Besser geeignet fuer Downstream-Regression-Analyse

    Wichtige Parameter:
      n_neighbors:  Grosse Nachbarschaft (15-50) → globale Struktur
                    Kleine Nachbarschaft (5-10)  → lokale Cluster
      min_dist:     Klein (0.0-0.1) → dichte Cluster
                    Gross (0.5-1.0) → gleichmaessige Verteilung
      metric:       cosine fuer normalisierte high-dim Embeddings besser
                    als euclidean (curse of dimensionality)
    """
    print("\n--- UMAP ---")
    print(f"  n_neighbors : {UMAP_N_NEIGHBORS}")
    print(f"  min_dist    : {UMAP_MIN_DIST}")
    print(f"  metric      : {UMAP_METRIC}")
    os.makedirs(output_dir, exist_ok=True)

    import umap

    # PCA-Vorverarbeitung beschleunigt UMAP (wie bei t-SNE)
    n = len(embeddings)
    pre_pca     = PCA(n_components=min(50, n-1), random_state=RANDOM_SEED)
    emb_reduced = pre_pca.fit_transform(embeddings)
    print(f"  PCA Vorreduktion: {embeddings.shape[1]} → {emb_reduced.shape[1]} Dim.")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=RANDOM_SEED,
        verbose=False,
    )
    coords = reducer.fit_transform(emb_reduced)
    print(f"  UMAP abgeschlossen. Shape: {coords.shape}")

    outlier_indices = find_outliers(coords)
    print(f"  Ausreisser: {len(outlier_indices)}")
    for idx in outlier_indices:
        fname = os.path.basename(file_paths[idx])
        meta  = metadata[idx]
        print(f"    [{idx}] {fname} | Alter={meta.get('age','?')} | Sex={meta.get('sex','?')}")

    # ── Statische PNG ──
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle(
        f"UMAP der SSL-Embeddings (CBCT Schaedel, n={n})\n"
        f"n_neighbors={UMAP_N_NEIGHBORS}, min_dist={UMAP_MIN_DIST}, metric={UMAP_METRIC}",
        fontsize=13, fontweight="bold"
    )

    # Plot 1: Geschlecht
    _scatter_sex(axes[0], coords, metadata)
    _add_outlier_annotations(axes[0], coords, file_paths, metadata,
                             outlier_indices, annotate_all)
    axes[0].set_title("Geschlecht  (rot = Ausreisser)", fontsize=11)
    axes[0].set_xlabel("UMAP 1"); axes[0].set_ylabel("UMAP 2")
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Alter
    _scatter_age(axes[1], fig, coords, metadata)
    _add_outlier_annotations(axes[1], coords, file_paths, metadata,
                             outlier_indices, annotate_all)
    axes[1].set_title("Alter  (rot = Ausreisser)", fontsize=11)
    axes[1].set_xlabel("UMAP 1"); axes[1].set_ylabel("UMAP 2")
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Kaukraft
    _scatter_force(axes[2], fig, coords, metadata)
    _add_outlier_annotations(axes[2], coords, file_paths, metadata,
                             outlier_indices, annotate_all)
    axes[2].set_title("Kaukraft (N)", fontsize=11)
    axes[2].set_xlabel("UMAP 1"); axes[2].set_ylabel("UMAP 2")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    png_path = os.path.join(output_dir, "umap_embeddings.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  PNG: {png_path}")

    # ── UMAP mit verschiedenen n_neighbors vergleichen ──
    _plot_umap_sensitivity(emb_reduced, file_paths, metadata, output_dir)

    # ── Interaktiver HTML Plot ──
    save_interactive_html(
        coords, file_paths, metadata,
        title=f"UMAP der SSL-Embeddings (n={n})",
        output_path=os.path.join(output_dir, "umap_interactive.html"),
        x_label="UMAP 1", y_label="UMAP 2",
    )
    save_outlier_report(outlier_indices, file_paths, metadata, coords, "umap", output_dir)

    return coords


def _plot_umap_sensitivity(emb_reduced, file_paths, metadata, output_dir):
    """
    Zeigt UMAP mit 3 verschiedenen n_neighbors Werten.
    Hilft zu beurteilen ob Cluster-Struktur robust oder artefaktisch ist.

    Interpretation:
      Robuste Cluster:   erscheinen bei allen n_neighbors-Werten
      Artefakt-Cluster: nur bei einem bestimmten n_neighbors-Wert
    """
    import umap

    n_neighbors_list = [5, 15, 30]
    print(f"\n  UMAP Sensitivitaetsanalyse: n_neighbors = {n_neighbors_list}")

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle(
        "UMAP Sensitivitaet: n_neighbors Vergleich\n"
        "(Robuste Cluster erscheinen bei allen Werten)",
        fontsize=13, fontweight="bold"
    )

    for ax, n_nb in zip(axes, n_neighbors_list):
        reducer = umap.UMAP(
            n_components=2, n_neighbors=n_nb,
            min_dist=UMAP_MIN_DIST, metric=UMAP_METRIC,
            random_state=RANDOM_SEED, verbose=False,
        )
        coords_nb = reducer.fit_transform(emb_reduced)

        _scatter_age(ax, fig, coords_nb, metadata)
        ax.set_title(f"n_neighbors={n_nb}", fontsize=12)
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "umap_sensitivity.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Sensitivitaets-Plot: {out_path}")


# ─────────────────────────────────────────────
# Methoden-Vergleich: PCA vs t-SNE vs UMAP
# ─────────────────────────────────────────────
def plot_method_comparison(pca_coords, tsne_coords, umap_coords,
                           file_paths, metadata, output_dir):
    """
    Vergleichs-Plot aller drei Dimensionsreduktionsmethoden.
    Eingefaerbt nach Alter — zeigt wie jede Methode Altersstruktur kodiert.
    Hilft zu verstehen welche Methode die Manifold am besten erfasst.
    """
    print("\n--- Methoden-Vergleich PCA / t-SNE / UMAP ---")
    os.makedirs(output_dir, exist_ok=True)

    ages   = np.array([m.get("age", np.nan) for m in metadata], dtype=float)
    forces = np.array([m.get("kaukraft", np.nan) for m in metadata], dtype=float)

    fig, axes = plt.subplots(2, 3, figsize=(21, 13))
    fig.suptitle("Methoden-Vergleich: PCA vs t-SNE vs UMAP\n"
                 "Zeile 1: Alter-Gradient  |  Zeile 2: Geschlecht",
                 fontsize=14, fontweight="bold")

    method_data = [
        (pca_coords,  "PCA",   "PC1", "PC2"),
        (tsne_coords, "t-SNE", "t-SNE 1", "t-SNE 2"),
        (umap_coords, "UMAP",  "UMAP 1", "UMAP 2"),
    ]

    # Zeile 1: Alter
    for ax, (coords, name, xlabel, ylabel) in zip(axes[0], method_data):
        if not np.all(np.isnan(ages)):
            sc = ax.scatter(coords[:, 0], coords[:, 1], c=ages, cmap="plasma",
                            s=40, alpha=0.85, edgecolors="white", linewidths=0.3)
            fig.colorbar(sc, ax=ax, label="Alter (Jahre)", shrink=0.8)
        ax.set_title(f"{name} — Alter", fontsize=12, fontweight="bold")
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    # Zeile 2: Geschlecht
    for ax, (coords, name, xlabel, ylabel) in zip(axes[1], method_data):
        _scatter_sex(ax, coords, metadata)
        ax.set_title(f"{name} — Geschlecht", fontsize=12, fontweight="bold")
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "method_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Vergleichs-Plot: {out_path}")


# ─────────────────────────────────────────────
# Regression-Validierung
# ─────────────────────────────────────────────
def evaluate_regression(file_paths, metadata, output_dir, ssl_mode):
    print("\n--- Regressions-Validierung ---")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(MODEL_REG):
        print(f"  [Fehler] {MODEL_REG} nicht gefunden.")
        return

    labeled = [(fp, m) for fp, m in zip(file_paths, metadata)
               if m.get("kaukraft") is not None]
    if len(labeled) < 5:
        print(f"  [Warnung] Nur {len(labeled)} Labels.")
        return

    if ssl_mode == "mae":
        ssl_model = MAEModel(embed_dim=512).to(device)
        regressor = KaukraftRegressor(ssl_model, input_dim=512).to(device)
        regressor.ssl_mode = "mae"
    else:
        ssl_model = MultiScaleSSLModel().to(device)
        regressor = KaukraftRegressor(ssl_model, input_dim=768).to(device)
        regressor.ssl_mode = "byol"

    raw   = torch.load(MODEL_REG, map_location=device, weights_only=False)
    state = raw["model_state"] if "model_state" in raw else raw
    regressor.load_state_dict(state, strict=False)
    regressor.eval()

    predictions, ground_truth, patient_ids = [], [], []

    for fpath, meta in labeled:
        fpath = fpath.replace("\\", "/")
        try:
            s     = global_transforms({"image": fpath})
            image = s["image"].unsqueeze(0).to(device)
            with torch.no_grad():
                pred = regressor(image).item()
            predictions.append(pred)
            ground_truth.append(meta["kaukraft"])
            patient_ids.append(os.path.basename(fpath))
        except Exception as e:
            print(f"  [Fehler] {os.path.basename(fpath)}: {e}")

    if not predictions:
        return

    predictions  = np.array(predictions)
    ground_truth = np.array(ground_truth)

    mae   = mean_absolute_error(ground_truth, predictions)
    r2    = r2_score(ground_truth, predictions)
    r, pv = pearsonr(ground_truth, predictions)

    print(f"\n  Patienten  : {len(predictions)}")
    print(f"  MAE        : {mae:.1f} N")
    print(f"  R²         : {r2:.4f}")
    print(f"  Pearson r  : {r:.4f}  (p={pv:.4f})")

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(ground_truth, predictions, s=80, alpha=0.8,
               color="#1976D2", edgecolors="white", linewidths=0.5, zorder=3)
    for i, pid in enumerate(patient_ids):
        ax.annotate(pid.replace(".nii.gz",""), (ground_truth[i], predictions[i]),
                    fontsize=6, xytext=(4, 4), textcoords="offset points", alpha=0.7)

    lims = [min(ground_truth.min(), predictions.min()) * 0.9,
            max(ground_truth.max(), predictions.max()) * 1.1]
    ax.plot(lims, lims, "k--", alpha=0.4, linewidth=1.5, label="Ideal (y=x)")
    coeffs = np.polyfit(ground_truth, predictions, 1)
    x_line = np.linspace(lims[0], lims[1], 100)
    ax.plot(x_line, np.polyval(coeffs, x_line), "r-", linewidth=2,
            label=f"Regression (r={r:.3f})")
    ax.set_xlabel("Echte Kaukraft (N)", fontsize=13)
    ax.set_ylabel("Vorhergesagte Kaukraft (N)", fontsize=13)
    ax.set_title("Kaukraft-Vorhersage vs. Messung", fontsize=14, fontweight="bold")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    stats = (f"MAE={mae:.1f}N  R²={r2:.3f}\n"
             f"Pearson r={r:.3f}  p={pv:.4f}\nn={len(predictions)}")
    ax.text(0.05, 0.95, stats, transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", alpha=0.8))
    plt.tight_layout()
    out_path = os.path.join(output_dir, "regression_scatter.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot: {out_path}")

    import csv
    csv_path = os.path.join(output_dir, "predictions.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient", "kaukraft_echt", "kaukraft_pred", "fehler_N"])
        for pid, gt, pred in zip(patient_ids, ground_truth, predictions):
            writer.writerow([pid, f"{gt:.1f}", f"{pred:.1f}", f"{abs(gt-pred):.1f}"])
    print(f"  CSV: {csv_path}")


# ─────────────────────────────────────────────
# Einstiegspunkt
# ─────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    checkpoint_path = args.checkpoint if args.checkpoint else MODEL_SSL
    umap_available  = False if args.no_umap else check_umap_available()

    print(f"\nModus     : {args.mode.upper()}")
    print(f"SSL-Modus : {args.ssl_mode.upper()}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"UMAP      : {'Ja' if umap_available else 'Nein (--no_umap oder nicht installiert)'}")

    print("\nMetadaten laden:")
    file_paths, metadata = load_metadata(METADATA_JSON)

    if args.mode in ("visual", "all"):
        print("\nModell laden:")
        ssl_model, embed_dim = load_ssl_model(checkpoint_path, args.ssl_mode)

        print("\nEmbeddings extrahieren:")
        embeddings = extract_embeddings(
            file_paths, ssl_model, args.ssl_mode, embed_dim
        )

        check_embedding_collapse(embeddings)

        # PCA-Koordinaten fuer Methoden-Vergleich speichern
        pca_full = PCA(n_components=2, random_state=RANDOM_SEED)
        pca_coords = pca_full.fit_transform(embeddings)
        plot_pca(embeddings, file_paths, metadata, OUTPUT_DIR,
                 annotate_all=args.annotate_all)

        # t-SNE
        pre_pca     = PCA(n_components=min(50, len(embeddings)-1), random_state=RANDOM_SEED)
        emb_reduced = pre_pca.fit_transform(embeddings)
        perplexity  = min(TSNE_PERPLEXITY, len(embeddings)-1, 50)
        tsne        = TSNE(n_components=2, perplexity=perplexity, max_iter=TSNE_ITERATIONS,
                           random_state=RANDOM_SEED, init="pca", learning_rate="auto")
        tsne_coords = tsne.fit_transform(emb_reduced)
        plot_tsne(embeddings, file_paths, metadata, OUTPUT_DIR,
                  annotate_all=args.annotate_all)

        # UMAP
        umap_coords = None
        if umap_available:
            umap_coords = plot_umap(embeddings, file_paths, metadata, OUTPUT_DIR,
                                    annotate_all=args.annotate_all)

            # Methoden-Vergleich nur wenn alle drei vorhanden
            plot_method_comparison(
                pca_coords, tsne_coords, umap_coords,
                file_paths, metadata, OUTPUT_DIR
            )

    if args.mode in ("regression", "all"):
        evaluate_regression(file_paths, metadata, OUTPUT_DIR, args.ssl_mode)

    print(f"\nAlle Ergebnisse in: {os.path.abspath(OUTPUT_DIR)}/")
    print("\nInteraktive Plots:")
    print(f"  pca_interactive.html")
    print(f"  tsne_interactive.html")
    if umap_available:
        print(f"  umap_interactive.html")
    print("→ Im Browser oeffnen und ueber Punkte hovern")