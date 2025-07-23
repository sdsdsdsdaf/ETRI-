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
from tqdm.auto import tqdm
import h5py
from scipy.ndimage import affine_transform
import random



# ====================
# Normalize with mask
# ====================


def normalize_batch_tensor(t: torch.Tensor) -> torch.Tensor:
    """
    Normalize (N, C, H, W) image tensor using ImageNet statistics
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=t.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=t.device).view(1, 3, 1, 1)
    return (t - mean) / std

def save_as_h5(meta: dict, save_path: str):
    with h5py.File(save_path, "w") as f:
        # 주요 텐서 저장
        f.create_dataset("tensor_plot_sleep", data=meta["tensor_plot_sleep"].numpy())
        f.create_dataset("tensor_heatmap_sleep", data=meta["tensor_heatmap_sleep"].numpy())
        f.create_dataset("tensor_plot_lifelog", data=meta["tensor_plot_lifelog"].numpy())
        f.create_dataset("tensor_heatmap_lifelog", data=meta["tensor_heatmap_lifelog"].numpy())
        f.create_dataset("mble_data_sleep", data=meta["mBle_tensor_sleep"].numpy())
        f.create_dataset("mble_data_lifelog", data=meta["mBle_tensor_lifelog"].numpy())
        f.create_dataset("label", data=meta["label"].numpy())

        # 메타데이터는 속성으로 저장
        f.attrs["subject_id"] = meta["subject_id"]
        f.attrs["sleep_date"] = meta["sleep_date"]
        f.attrs["lifelog_date"] = meta["lifelog_date"]
        f.attrs["resize"] = meta["resize"]
        f.attrs["modality_names"] = np.array(meta["modality_names"], dtype="S")

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

        alpha_vec = np.clip(mask, 0, 1)

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

        '''
        if np.all(mask == 0):
            ax.set_facecolor("lightgray")
            ax.text(0.5, 0.5, "PADDED", color="red", fontsize=10,
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_ylim(-1, 1)
        '''
        if np.isclose(y_min, y_max, atol=1e-10):
            margin = 1e-2 if y_min == 0 else abs(y_min * 0.01)
            ax.set_ylim(y_min - margin, y_max + margin)
        else:
            ax.set_ylim(y_min, y_max)

        ax.axis('off')
    #print(f"[{col}] mask.sum(): {mask.sum()}, mask.shape: {mask.shape}, dtype: {mask.dtype}")
    
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
def plot_modal_heatmap_image(data_df, mask_df, resize=(224, 224)) -> Image.Image:
    numeric_cols = data_df.select_dtypes(include='number').columns

    # 정규화 + 마스킹
    normed_list = []
    alpha_list = []

    for col in numeric_cols:
        data = data_df[col].values
        mask = mask_df[col].values  
        normed = normalize_df_with_mask(data, mask)

        normed_list.append(normed)
        alpha_list.append(np.clip(mask, 0, 1)) 

    normed_stack = np.stack(normed_list)  # (C, T)
    alpha_stack = np.stack(alpha_list)    # (C, T)

    # RGBA 이미지로 변환
    cmap = plt.colormaps['viridis']
    rgba = cmap((normed_stack - (-2)) / (2 - (-2))) 
    rgba[..., -1] = alpha_stack  # alpha 채널에 마스크 적용

    # Plot
    fig, ax = plt.subplots(figsize=(2, 2), dpi=100)
    

    '''
    if np.all(mask==0):
        # 🔻 전부 패딩인 경우 회색 배경 + "PADDED" 텍스트
        ax.set_facecolor("lightgray")
        ax.text(
            0.5, 0.5, "PADDED", color="red", fontsize=10,
            ha="center", va="center", transform=ax.transAxes
        )
    else:'''
    
    ax.imshow(rgba, aspect='auto', interpolation='nearest')
    ax.axis('off')

    buf = BytesIO()
    plt.tight_layout(pad=2.0)
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    return img.resize(resize)


# ====================
# Convert PIL Image to torch Tensor
# ====================
def image_to_tensor(images: List[Image.Image]) -> torch.Tensor:
    """
    Args:
        images: list of PIL Image 객체들

    Returns:
        torch.Tensor: (N, 3, 224, 224)
    """
    tensor_list = [
        torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        for img in images
    ]
    return torch.stack(tensor_list, dim=0)  # (N, 3, 224, 224)

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
    img_save: bool = False,
    debug = False,
    save_method = "h5",
    label_path = None
):
    with open(pkl_path, "rb") as f:
        data_dict = pickle.load(f)
    with open(label_path, "rb") as f:
        label_dict = pickle.load(f)

    os.makedirs(save_dir, exist_ok=True)

    print(f"Converting Data to Img...")
    for key, (sleep_dict, lifelog_dict) in tqdm(data_dict.items(), leave=False):
        if len(sleep_dict) == 0:
            sleep_dict = {
                'mACStatus': None, 
                'mACStatus': None, 
                'mActivity': None, 
                'mAmbience': None, 
                'mBle':None, 
                'mGps':None, 
                'mLight':None, 
                'mScreenStatus':None, 
                'mWifi':None, 
                'wPedo':None,
                'mUsageStats':None,
                'wHr': None, 
                'wLight': None}
            
        if len(lifelog_dict) == 0:
            lifelog_dict = {
                'mACStatus': None, 
                'mACStatus': None, 
                'mActivity': None, 
                'mAmbience': None, 
                'mBle':None, 
                'mGps':None, 
                'mLight':None, 
                'mScreenStatus':None, 
                'mWifi':None, 
                'wPedo':None,
                'mUsageStats':None,
                'wHr': None, 
                'wLight': None}
            
        subject_id, sleep_date, lifelog_date = key

        #디버그
        none_sleep_mods_list = []
        none_lifelog_mods_list = []
        debug_log_lines = []

        all_mods = list(set(sleep_dict.keys()) | set(lifelog_dict.keys()))
        filled_sleep = {}
        filled_lifelog = {}
        T=5

        for mod in sleep_dict.keys():
            val_s = sleep_dict.get(mod)
            if val_s is None or not isinstance(val_s, tuple):
                df = pd.DataFrame({f"{mod}_pad": np.zeros(T)})
                mask = pd.DataFrame({f"{mod}_pad": np.zeros(T)})
                #디버그
                none_sleep_mods_list.append(mod)
            else:
                df, mask = val_s
            filled_sleep[mod] = (df, mask)

            val_l = lifelog_dict.get(mod)
            if val_l is None or not isinstance(val_l, tuple):
                df = pd.DataFrame({f"{mod}_pad": np.zeros(T)})
                mask = pd.DataFrame({f"{mod}_pad": np.zeros(T)})
                #디버그
                none_lifelog_mods_list.append(mod)
            else:
                df, mask = val_l
            filled_lifelog[mod] = (df, mask)

            val_l = lifelog_dict.get(mod)
            if val_l is None or not isinstance(val_l, tuple):
                df = pd.DataFrame({f"{mod}_pad": np.zeros(T)})
                mask = pd.DataFrame({f"{mod}_pad": np.zeros(T)})
                #디버그
                none_lifelog_mods_list.append(mod)
            else:
                df, mask = val_l
            filled_lifelog[mod] = (df, mask)


        images_plot_sleep, images_heat_sleep = [], []
        images_plot_lifelog, images_heat_lifelog = [], []

        for modality, (df, mask) in filled_sleep.items():
            if modality == 'mBle':
                continue
            img_plot = plot_modal_subplot_image(df, mask, resize)
            img_heat = plot_modal_heatmap_image(df, mask, resize)

            if img_save:
                sample_dir = os.path.join(save_dir, f"{subject_id}_{sleep_date}_{lifelog_date}")
                os.makedirs(sample_dir, exist_ok=True)
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


        tensor_plot_sleep = normalize_batch_tensor(image_to_tensor(images_plot_sleep))
        tensor_heat_sleep = normalize_batch_tensor(image_to_tensor(images_heat_sleep))
        tensor_plot_lifelog = normalize_batch_tensor(image_to_tensor(images_plot_lifelog))
        tensor_heat_lifelog = normalize_batch_tensor(image_to_tensor(images_heat_lifelog))

        mBle_tensor_sleep = torch.cat([torch.tensor(filled_sleep['mBle'][0].values.T).flatten(), 
                             torch.tensor(filled_sleep['mBle'][1].values.T).flatten()], dim=0)
        mBle_tensor_lifelog = torch.cat([torch.tensor(filled_lifelog['mBle'][0].values.T).flatten(), 
                               torch.tensor(filled_lifelog['mBle'][1].values.T).flatten()], dim=0)

        label_list = torch.Tensor(list(label_dict[key].values()))

        meta = {
            "mBle_tensor_sleep": mBle_tensor_sleep,
            "mBle_tensor_lifelog": mBle_tensor_lifelog,
            "tensor_plot_sleep": tensor_plot_sleep,
            "tensor_heatmap_sleep": tensor_heat_sleep,
            "tensor_plot_lifelog": tensor_plot_lifelog,
            "tensor_heatmap_lifelog": tensor_heat_lifelog,
            "label": label_list,
            "subject_id": subject_id,
            "sleep_date": str(sleep_date),
            "lifelog_date": str(lifelog_date),
            "modality_names": all_mods,
            "resize": resize,
        }

        file_name = f"{subject_id}_{sleep_date}_{lifelog_date}.{save_method}"
        save_as_h5(meta, os.path.join(save_dir, file_name))

        '''
        none_sleep_mods = get_none_modalities(sleep_dict)
        none_lifelog_mods = get_none_modalities(lifelog_dict)

        print("❌ None sleep modalities:", none_sleep_mods)
        print("❌ None lifelog modalities:", none_lifelog_mods)
        '''

    print("Complete converting")

with open("Data_Dict/train_data_filtered_daily_linear_ratio_mask.pkl", 'rb') as f:
    data_dict = pickle.load(f)

data_dict_5_samples = {k: data_dict[k] for k in list(data_dict.keys())[:10]}

with open("train_data_subset_5.pkl", 'wb') as f:
    pickle.dump(data_dict_5_samples, f)

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Run example
process_all_samples(
    pkl_path="Data_Dict/test_data_daily_linear_ratio_mask.pkl",
    label_path = "Data_Dict/test_label_daily_linear_ratio_mask.pkl",
    save_dir="Img_Data/test",
    resize=(224, 224),
    img_save=True,
    debug = False
)

import torch, numpy
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.__file__)
print(numpy.__file__)
