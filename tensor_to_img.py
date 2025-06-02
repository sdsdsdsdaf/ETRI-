import os
import pickle
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
import torch
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
from matplotlib.collections import LineCollection

# ====================
# Normalize with mask
# ====================
def normalize_df_with_mask(data: np.ndarray, mask: np.ndarray, eps: float = 1e-8):
    masked = data[mask == 1]
    if len(masked) == 0:
        return np.zeros_like(data)

    mean = masked.mean()
    std = masked.std()

    centered = data - mean
    if std < eps:
        return centered
    return centered / (std + eps)

# ====================
# Subplot visualization with variable alpha fill
# ====================
def plot_modal_subplot_image(data_df: pd.DataFrame, mask_df: pd.DataFrame, resize=(224, 224)) -> Image.Image:
    numeric_cols = data_df.select_dtypes(include='number').columns
    n_features = len(numeric_cols)
    fig, axes = plt.subplots(
    n_features, 1,
    figsize=(3, 2.5 * n_features),  # 🔺 세로 크기 늘림
    squeeze=False
    )
    fig.subplots_adjust(hspace=2)

    for i, col in enumerate(numeric_cols):
        if not data_df.index.equals(mask_df.index):
            data_df = data_df.sort_index()
            mask_df = mask_df.sort_index()

        values = data_df[col].values
        mask = mask_df[col].values
        data = normalize_df_with_mask(values, mask)

        ax = axes[i, 0]
        ax.plot(data, color='blue', linewidth=1)

        alpha_vec = 1 - np.clip(mask, 0, 1)

        segments = [
            [[x, 0], [x, data[x]]]
            for x in range(len(data))
        ]
        colors = [(1, 0, 0, alpha_vec[x]) for x in range(len(data))]

        lc = LineCollection(segments, colors=colors, linewidths=1)
        ax.add_collection(lc)
        ax.set_xlim(0, len(data))

        y_min = np.min(data)
        y_max = np.max(data)

        if np.all(mask == 0):
            ax.set_facecolor("lightgray")
            ax.text(0.5, 0.5, "PADDED", color="red", fontsize=10,
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_ylim(-1, 1)
        elif np.isclose(y_min, y_max, atol=1e-10):
            margin = 1e-2 if y_min == 0 else abs(y_min * 0.01)
            ax.set_ylim(y_min - margin, y_max + margin)
        else:
            ax.set_ylim(y_min, y_max)

        ax.axis('off')
    print(f"[{col}] mask.sum(): {mask.sum()}, mask.shape: {mask.shape}, dtype: {mask.dtype}")
    
    buf = BytesIO()
    plt.tight_layout(pad=2.0)
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    return img.resize(resize)

# ====================
# Heatmap visualization
# ====================
def plot_modal_heatmap_image(data_df: pd.DataFrame, mask_df: pd.DataFrame, resize=(224, 224)) -> Image.Image:
    numeric_cols = data_df.select_dtypes(include='number').columns

    normed_list = []
    mask_list = []

    for col in numeric_cols:
        data = data_df[col].values
        mask = mask_df[col].values
        normed = normalize_df_with_mask(data, mask)
        normed_list.append(normed)
        mask_list.append(np.clip(mask, 0, 1))

    normed_stack = np.stack(normed_list)
    mask_stack = np.stack(mask_list)
    masked_data = np.ma.masked_where(mask_stack != 1, normed_stack)

    plt.figure(figsize=(2, 2))
    plt.imshow(masked_data, aspect='auto', cmap='viridis', interpolation='nearest', vmin=-2, vmax=2)
    plt.axis('off')
    buf = BytesIO()
    plt.tight_layout(pad=2.0)
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    return img.resize(resize)

# ====================
# Convert PIL Image to torch Tensor
# ====================
def image_to_tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

# ====================
# Merge a list of images vertically
# ====================
def merge_images_vertically(images: List[Image.Image]) -> Image.Image:
    total_height = sum(img.height for img in images)
    max_width = max(img.width for img in images)
    merged = Image.new("RGB", (max_width, total_height))
    y_offset = 0
    for img in images:
        merged.paste(img, (0, y_offset))
        y_offset += img.height
    return merged

def get_none_modalities(modality_dict: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]) -> List[str]:
    none_mods = []
    for name, pair in modality_dict.items():
        if (
            pair is None or
            not isinstance(pair, tuple) or
            len(pair) != 2 or
            pair[0] is None or
            pair[1] is None
        ):
            none_mods.append(name)
    return none_mods

# ====================
# Main processing function
# ====================
def process_all_samples(
    pkl_path: str,
    save_dir: str,
    resize: Tuple[int, int] = (224, 224),
    img_save: bool = False
):
    with open(pkl_path, "rb") as f:
        data_dict = pickle.load(f)

    os.makedirs(save_dir, exist_ok=True)

    for key, (sleep_dict, lifelog_dict) in data_dict.items():
        subject_id, sleep_date, lifelog_date = key


        #디버그
        df = sleep_dict['mLight'][1]
        print(df.select_dtypes(include='number').sum())
        for val in sleep_dict.values():
            if val and isinstance(val, tuple) and val[0] is not None:
                T = len(val[0])
                time_values = val[0]["date"]
                break

        all_mods = list(set(sleep_dict.keys()) | set(lifelog_dict.keys()))
        filled_sleep = {}
        filled_lifelog = {}

        for mod in all_mods:
            val_s = sleep_dict.get(mod)
            if val_s is None or not isinstance(val_s, tuple):
                df = pd.DataFrame({f"{mod}_pad": np.zeros(T), "date": time_values})
                mask = pd.DataFrame({f"{mod}_pad": np.zeros(T), "date": time_values})
            else:
                df, mask = val_s
            filled_sleep[mod] = (df, mask)

            val_l = lifelog_dict.get(mod)
            if val_l is None or not isinstance(val_l, tuple):
                df = pd.DataFrame({f"{mod}_pad": np.zeros(T), "date": time_values})
                mask = pd.DataFrame({f"{mod}_pad": np.zeros(T), "date": time_values})
            else:
                df, mask = val_l
            filled_lifelog[mod] = (df, mask)

        sample_dir = os.path.join(save_dir, f"{subject_id}_{sleep_date}_{lifelog_date}")
        os.makedirs(sample_dir, exist_ok=True)

        images_plot_sleep, images_heat_sleep = [], []
        images_plot_lifelog, images_heat_lifelog = [], []

        for modality, (df, mask) in filled_sleep.items():
            if modality == 'mBle':
                continue
            img_plot = plot_modal_subplot_image(df, mask, resize)
            img_heat = plot_modal_heatmap_image(df, mask, resize)

            if img_save:
                img_plot.save(os.path.join(sample_dir, f"sleep_{modality}_plot.png"))
                img_heat.save(os.path.join(sample_dir, f"sleep_{modality}_heatmap.png"))

            images_plot_sleep.append(img_plot)
            images_heat_sleep.append(img_heat)

        for modality, (df, mask) in filled_lifelog.items():
            if modality == 'mBle':
                continue
            img_plot = plot_modal_subplot_image(df, mask, resize)
            img_heat = plot_modal_heatmap_image(df, mask, resize)

            if img_save:
                img_plot.save(os.path.join(sample_dir, f"lifelog_{modality}_plot.png"))
                img_heat.save(os.path.join(sample_dir, f"lifelog_{modality}_heatmap.png"))

            images_plot_lifelog.append(img_plot)
            images_heat_lifelog.append(img_heat)

        merged_plot_sleep = merge_images_vertically(images_plot_sleep)
        merged_heat_sleep = merge_images_vertically(images_heat_sleep)
        merged_plot_lifelog = merge_images_vertically(images_plot_lifelog)
        merged_heat_lifelog = merge_images_vertically(images_heat_lifelog)

        tensor_plot_sleep = image_to_tensor(merged_plot_sleep)
        tensor_heat_sleep = image_to_tensor(merged_heat_sleep)
        tensor_plot_lifelog = image_to_tensor(merged_plot_lifelog)
        tensor_heat_lifelog = image_to_tensor(merged_heat_lifelog)

        mBle_tensor_sleep = torch.tensor(sleep_dict['mBle'][0].values.T)
        mBle_tensor_lifelog = torch.tensor(lifelog_dict['mBle'][1].values.T)

        meta = {
            "mBle_tensor_sleep": mBle_tensor_sleep,
            "mBle_tensor_lifelog": mBle_tensor_lifelog,
            "tensor_plot_sleep": tensor_plot_sleep,
            "tensor_heatmap_sleep": tensor_heat_sleep,
            "tensor_plot_lifelog": tensor_plot_lifelog,
            "tensor_heatmap_lifelog": tensor_heat_lifelog,
            "subject_id": subject_id,
            "sleep_date": str(sleep_date),
            "lifelog_date": str(lifelog_date),
            "modality_names": all_mods,
            "resize": resize
        }

        file_name = f"{subject_id}_{sleep_date}_{lifelog_date}.pt"
        torch.save(meta, os.path.join(save_dir, file_name))

        none_sleep_mods = get_none_modalities(sleep_dict)
        none_lifelog_mods = get_none_modalities(lifelog_dict)

        print("❌ None sleep modalities:", none_sleep_mods)
        print("❌ None lifelog modalities:", none_lifelog_mods)



# Run example
process_all_samples(
    pkl_path="train_data_subset_5.pkl",
    save_dir="Img_samples",
    resize=(224, 224),
    img_save=True
)
