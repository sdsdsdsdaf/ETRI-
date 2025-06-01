import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
import seaborn as sns
from scipy.signal import spectrogram
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import torch
from typing import List, Tuple
from torchvision.transforms.functional import to_pil_image

def load_tensor_and_show(path: str):
    data = torch.load(path)
    tensor = data["tensor"]
    image = to_pil_image(tensor)  # (C, H, W) → PIL.Image
    image.show()
    return data


def save_tensor_with_meta(
    tensor: torch.Tensor,
    path: str,
    subject_id: str,
    sleep_date: str,
    lifelog_date: str,
    modality_names: List[str],
    mask_style: str,
    resize: Tuple[int, int]
):
    meta = {
        "tensor": tensor,
        "subject_id": subject_id,
        "sleep_date": sleep_date,
        "lifelog_date": lifelog_date,
        "modality_names": modality_names,
        "mask_style": mask_style,
        "resize": resize
    }
    torch.save(meta, path)


def normalize_df_with_mask(df: pd.DataFrame, mask_df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:

    normalized_df = df.copy()
    numeric_cols = df.select_dtypes(include='number').columns
    print(f"디버깅 수치형 컬럼의 개수: {len(numeric_cols)} 종류: {numeric_cols}")

    for col in numeric_cols:
        if col not in mask_df:
            continue
        values = df[col].values
        mask = mask_df[col].values.astype(bool)

        if not mask.any():
            continue

        mean = values[mask].mean()
        std = values[mask].std()

        normalized_values = (values - mean) / (std + eps)
        normalized_values[~mask] = 0.0 

        normalized_df[col] = normalized_values

    return normalized_df

# Plot
def save_plot_image_with_mask(data, mask, path):
    data = normalize_df_with_mask(data, mask)
    mask = np.clip(mask, 0, 1)
    plt.figure(figsize=(2, 2))
    plt.plot(data, linewidth=2, alpha=0.8)
    plt.fill_between(range(len(data)), data, alpha=1 - mask, color='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Heatmap
def save_heatmap_image_with_mask(data, mask, path):
    data = normalize_df_with_mask(data, mask)
    mask = np.clip(mask, 0, 1)
    masked_data = np.ma.masked_where(mask != 1, data)
    if len(masked_data.shape) == 1:
        masked_data = masked_data.reshape(1, -1)
    plt.figure(figsize=(2, 2))
    plt.imshow(masked_data, aspect='auto', cmap='viridis')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Spectrogram
def save_spectrogram_image_with_mask(data, mask, path, fs=1.0):
    data = data * mask  # 마스크가 없는 부분은 0으로
    f, t, Sxx = signal.spectrogram(data, fs)
    plt.figure(figsize=(2, 2))
    plt.pcolormesh(t, f, Sxx, shading='gouraud', cmap='viridis')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Bar
def save_bar_trend_image_with_mask(data, mask, path):
    data = normalize_df_with_mask(data, mask)
    mask = np.clip(mask, 0, 1)
    colors = ['gray' if m < 0.5 else 'blue' for m in mask]
    plt.figure(figsize=(2, 2))
    plt.bar(np.arange(len(data)), data, color=colors)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()


def process_with_padding(data_dict, output_dir='visual_tensors', modalities_all=None, default_len=1440):
    os.makedirs(output_dir, exist_ok=True)
    idx = 1

    for (subject_id, sleep_date, lifelog_date), (sleep_dict, lifelog_dict) in data_dict.items():
        for source_name, modal_dict in [('sleep', sleep_dict), ('life', lifelog_dict)]:
            key = (subject_id, sleep_date if source_name == 'sleep' else lifelog_date)
            if key not in modal_dict:
                continue

            for modality in modalities_all:
                df, mask_df = None, None
                if modality in modal_dict[key]:
                    df, mask_df = modal_dict[key][modality]

                if df is None or mask_df is None or df.empty:
                    signal = np.zeros(default_len)
                    mask = np.ones(default_len)  # 마스크를 모두 1로
                else:
                    df = df.select_dtypes(include=np.number)
                    if df.empty:
                        signal = np.zeros(default_len)
                        mask = np.ones(default_len)
                    else:
                        signal = df.mean(axis=1).to_numpy()
                        mask = mask_df.select_dtypes(include=np.number).mean(axis=1).to_numpy() > 0.5
                        if len(signal) < default_len:
                            pad_len = default_len - len(signal)
                            signal = np.pad(signal, (0, pad_len), mode='constant')
                            mask = np.pad(mask.astype(int), (0, pad_len), constant_values=1)
                        else:
                            signal = signal[:default_len]
                            mask = mask[:default_len]
                        signal = normalize_df_with_mask(signal, mask)

                file_id = f"{idx}-{source_name}.pt"

                for mode, plot_func in {
                    'plot': lambda ax: ax.plot(signal),
                    'heatmap': lambda ax: sns.heatmap(signal[None, :], cmap='viridis', ax=ax, cbar=False),
                    'spectrogram': lambda ax: ax.pcolormesh(*spectrogram(signal)[:2], spectrogram(signal)[2], shading='gouraud'),
                    'bar': lambda ax: ax.bar(np.arange(len(signal)), signal)
                }.items():
                    fig, ax = plt.subplots()
                    try:
                        plot_func(ax)
                        path = os.path.join(output_dir, mode, modality, file_id)
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        save_image_tensor(fig, path)
                    except Exception as e:
                        plt.close(fig)
                        print(f"❌ Failed to save {file_id} ({mode}/{modality}): {e}")

        idx += 1