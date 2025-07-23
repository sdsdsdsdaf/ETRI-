import os
import random
from typing import Tuple, List, Optional, Dict

import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import numpy as np

# photometric augment 등에 쓰이는 함수
def add_gaussian_noise(img: torch.Tensor, std: float = 0.05) -> torch.Tensor:
    noise = torch.randn_like(img) * std
    return (img + noise).clamp(0, 1)

# RandomApplyOneOrMore: transforms 리스트 중 p_each 확률로 적용, 모두 불발 시 하나 강제 적용
class RandomApplyOneOrMore:
    def __init__(self, transforms_list: List, p_each: float = 0.5):
        self.transforms_list = transforms_list
        self.p_each = p_each

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        applied = False
        for t in self.transforms_list:
            if random.random() < self.p_each:
                try:
                    img = t(img)
                    applied = True
                except:
                    pass
        if not applied and self.transforms_list:
            t = random.choice(self.transforms_list)
            try:
                img = t(img)
            except:
                pass
        return img

class H5LifelogDataset(Dataset):
    def __init__(
        self,
        h5_dir: str,
        preload: bool = True,
        input_type_dict: Optional[Dict[str, List[str]]] = None,
        device: str = 'cpu',
        augment: bool = False,
        resize: Tuple[int, int] = (224, 224),
        aug_prob: float = 0.7,
        seed: Optional[int] = None,
        debug: bool = False
    ):
        """
        h5_dir: HDF5 파일(.h5)들이 모여 있는 디렉토리
        preload: True면 초기화 시 모든 파일을 메모리에 미리 로드
        input_type_dict: {'plot': [...], 'heatmap': [...], 'mBle': [...]} 등. None이면 attrs["modality_names"] 기반 기본 설정
        device: 'cpu' 또는 'cuda'
        augment: train 시 True로 설정하여 동적 augment 적용. val/테스트 시 False.
        resize: augment 후 리사이즈가 필요하면 사용. (여기선 기본적으로 원본 크기 유지)
        aug_prob: 각 샘플별 augment 확률
        seed: optional seed for reproducibility
        debug: True면 min/max 출력
        """
        self.h5_dir = h5_dir
        self.file_list = sorted([f for f in os.listdir(h5_dir) if f.endswith('.h5')])
        self.preload = preload
        self.input_type_dict = input_type_dict
        self.device = torch.device(device)
        self.augment = augment
        self.resize = resize
        self.aug_prob = aug_prob
        self.seed = seed
        self.debug = debug

        # ImageNet mean/std for inverse/normalize
        # shape (C,1,1)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(-1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(-1, 1, 1)

        # augment용 transforms
        if self.augment:
            weak_aug_list = [
                transforms.RandomAffine(degrees=3, translate=(0.01, 0.01), scale=(0.98, 1.02)),
                transforms.RandomRotation(degrees=3),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.3)),
                transforms.RandomErasing(p=1.0, scale=(0.02, 0.05), ratio=(0.3, 3.3), value='random'),
                transforms.Lambda(lambda x: add_gaussian_noise(x, std=0.02)),
            ]
            self.random_apply = RandomApplyOneOrMore(weak_aug_list, p_each=0.5)

        # preload: HDF5에서 미리 로드
        self.data = []
        if preload:
            print("[Data preloading...]")
            for fname in self.file_list:
                path = os.path.join(h5_dir, fname)
                with h5py.File(path, 'r') as f:
                    sample, label = self._load_sample(f)
                    self.data.append((sample, label))
            if self.debug:
                global_min, global_max = float('inf'), float('-inf')
                for sample, _ in self.data:
                    if sample.get('tensor_sleep') is not None:
                        vmin = torch.min(sample['tensor_sleep'])
                        vmax = torch.max(sample['tensor_sleep'])
                        global_min = min(global_min, vmin.item())
                        global_max = max(global_max, vmax.item())
                print(f"[DEBUG] preload 후 Normalize된 상태 tensor_sleep 최대: {global_max:.4f}, 최소: {global_min:.4f}")

    def __len__(self):
        # 원본 샘플 수만 반환
        return len(self.file_list)

    def __getitem__(self, idx: int):
        # seed 재설정: reproducibility 목적
        if self.seed is not None:
            seed_val = self.seed + idx
            random.seed(seed_val)
            torch.manual_seed(seed_val)
            np.random.seed(seed_val)

        # load sample,label
        if self.preload:
            sample, label = self.data[idx]
        else:
            fname = self.file_list[idx]
            path = os.path.join(self.h5_dir, fname)
            with h5py.File(path, 'r') as f:
                sample, label = self._load_sample(f)

        def process_normed_batch(batch: torch.Tensor, field_name: str) -> torch.Tensor:
            """
            batch: (N_mod, C, H, W), 이미 (x-mean)/std 상태
            반환: (N_mod, C, H, W), Normalize 상태 유지, augment 시 inverse→augment→clamp→re-normalize
            """
            batch = batch.to(self.device)
            N_mod, C, H, W = batch.shape
            out = []
            for i in range(N_mod):
                img = batch[i]
                if self.debug:
                    mn, mx = img.min().item(), img.max().item()
                    print(f"[DEBUG] {field_name} sample[{i}] before aug min/max: {mn:.4f}/{mx:.4f}")
                    mean_pc = img.mean(dim=(1, 2)).tolist()
                    std_pc  = img.std(dim=(1, 2), unbiased=False).tolist()
                    print(f"sample[{i}] 채널별 mean: {mean_pc}, std: {std_pc}")

                # augment 적용 결정
                if self.augment and random.random() < self.aug_prob:
                    # inverse normalize → [0,1]
                    img_denorm = (img * self.std + self.mean).clamp(0, 1)
                    if self.debug:
                        mn_d, mx_d = img_denorm.min().item(), img_denorm.max().item()
                        print(f"[DEBUG] {field_name} sample[{i}] after denorm min/max: {mn_d:.4f}/{mx_d:.4f}")
                    try:
                        img_aug = self.random_apply(img_denorm)
                    except:
                        img_aug = img_denorm
                    img_aug = img_aug.clamp(0, 1)
                    # 필요 시 resize:
                    # if img_aug.shape[1:] != self.resize:
                    #     img_aug = F.interpolate(img_aug.unsqueeze(0), size=self.resize,
                    #                             mode='bilinear', align_corners=False).squeeze(0)
                    # re-normalize
                    img_t = (img_aug - self.mean) / self.std
                    if self.debug:
                        mn2, mx2 = img_t.min().item(), img_t.max().item()
                        print(f"[DEBUG] {field_name} sample[{i}] after aug norm min/max: {mn2:.4f}/{mx2:.4f}")
                else:
                    img_t = img
                    if self.debug:
                        mn2, mx2 = img_t.min().item(), img_t.max().item()
                        print(f"[DEBUG] {field_name} sample[{i}] skip aug, retain norm min/max: {mn2:.4f}/{mx2:.4f}")
                out.append(img_t)
            return torch.stack(out, dim=0)

        # process tensor_sleep, tensor_lifelog if 존재
        if sample.get('tensor_sleep') is not None:
            sample['tensor_sleep'] = process_normed_batch(sample['tensor_sleep'], 'tensor_sleep')
        if sample.get('tensor_lifelog') is not None:
            sample['tensor_lifelog'] = process_normed_batch(sample['tensor_lifelog'], 'tensor_lifelog')
        # mble_data_* 등 non-image 텐서는 필요시 .to(self.device)로 이동 후 반환
        # 예: sample['mble_data_sleep'] = sample['mble_data_sleep'].to(self.device)  (필요시)
        return sample, label

    def _load_sample(self, f: h5py.File) -> Tuple[dict, torch.Tensor]:
        """
        HDF5 파일에서 sample dict와 label tensor를 로드
        """
        sleep_tensors = []
        lifelog_tensors = []
        mble_sleep_tensors = []
        mble_lifelog_tensors = []

        modal_name = [s.decode() for s in f.attrs["modality_names"]]
        if self.input_type_dict is None:
            # 'plot'에 해당 modality 개수만큼 요소를 읽도록 가정
            self.input_type_dict = {
                'plot': [m for m in modal_name if m != "mBle"],
                'mBle': ["mBle"]
            }

        for mode, modality_list in self.input_type_dict.items():
            n = len(modality_list)
            if mode == "plot":
                full_sleep = torch.tensor(f["tensor_plot_sleep"][:]).float()
                full_lifelog = torch.tensor(f["tensor_plot_lifelog"][:]).float()
                # full_sleep: (N_mod, C, H, W)
                for i in range(n):
                    sleep_tensors.append(full_sleep[i])
                    lifelog_tensors.append(full_lifelog[i])
            elif mode == "heatmap":
                full_sleep = torch.tensor(f["tensor_heatmap_sleep"][:]).float()
                full_lifelog = torch.tensor(f["tensor_heatmap_lifelog"][:]).float()
                for i in range(n):
                    sleep_tensors.append(full_sleep[i])
                    lifelog_tensors.append(full_lifelog[i])
            elif mode == "mBle":
                sleep_tensor = torch.tensor(f["mble_data_sleep"][:]).float().unsqueeze(0)
                lifelog_tensor = torch.tensor(f["mble_data_lifelog"][:]).float().unsqueeze(0)
                for _ in modality_list:
                    mble_sleep_tensors.append(sleep_tensor)
                    mble_lifelog_tensors.append(lifelog_tensor)
            else:
                raise ValueError(f"Unsupported input type: {mode}")

        sample = {
            "tensor_sleep": torch.stack(sleep_tensors, dim=0) if sleep_tensors else None,
            "tensor_lifelog": torch.stack(lifelog_tensors, dim=0) if lifelog_tensors else None,
            "mble_data_sleep": torch.cat(mble_sleep_tensors, dim=0) if mble_sleep_tensors else None,
            "mble_data_lifelog": torch.cat(mble_lifelog_tensors, dim=0) if mble_lifelog_tensors else None,
            "modality_names": modal_name,
        }
        label = torch.tensor(f["label"][:]).long() if "label" in f else None
        if label is None:
            raise ValueError("Label not found in HDF5 file.")
        return sample, label



if __name__ == "__main__":


    def find_dataset_min_max(dataset, exclude_mble=True, device='cuda'):
        """
        dataset: H5LifelogDataset 인스턴스
        exclude_mble: True면 mBle 데이터는 무시하고 tensor_sleep, tensor_lifelog만 검사
        returns: dict with info about global min/max
        """
        # DataLoader로 배치 단위로 읽되, 배치 크기 1로 샘플 단위 처리하거나
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        global_min = float('inf')
        global_max = float('-inf')
        info_min = None
        info_max = None
        
        for idx, (sample, label) in enumerate(loader):
            # sample은 dict 내에 각 값이 리스트나 tensor일 수 있음; DataLoader 기본 collate_fn을 가정
            # 만약 sample이 list of dict이라면 sample = sample[0]
            if isinstance(sample, list):
                sample = sample[0]
            
            # 검사할 텐서들 모음
            tensors = []
            names = []
            if sample.get('tensor_sleep') is not None:
                # shape: (N_mod, C, H, W)
                tensors.append(sample['tensor_sleep'].to(device))
                names.append('tensor_sleep')
            if sample.get('tensor_lifelog') is not None:
                tensors.append(sample['tensor_lifelog'].to(device))
                names.append('tensor_lifelog')
            # mBle 제외이므로 sample['mble_data_*']는 건너뛰기
            
            for t_name, t in zip(names, tensors):
                # t: (N, C, H, W)
                # flatten해서 값 비교; 하지만 위치도 알고 싶다면 argmax/argmin 사용
                # 우선 전체 값의 max/min
                cur_max = torch.max(t)
                cur_min = torch.min(t)
                # global 비교
                if cur_max.item() > global_max:
                    global_max = cur_max.item()
                    # 위치 찾기: argmax
                    # flatten index
                    flat_idx = torch.argmax(t).item()
                    # 복원: idx_mod, channel, y, x
                    B, N_mod, C, H, W = t.shape
                    idx_mod = flat_idx // (C*H*W)
                    rem = flat_idx % (C*H*W)
                    ch = rem // (H*W)
                    rem2 = rem % (H*W)
                    y = rem2 // W
                    x = rem2 % W
                    info_max = {
                        'dataset_idx': idx,    # DataLoader 배치 idx
                        'field': t_name,
                        'modality_index': idx_mod,
                        'channel': ch,
                        'y': y,
                        'x': x,
                        'value': global_max
                    }
                if cur_min.item() < global_min:
                    global_min = cur_min.item()
                    flat_idx = torch.argmin(t).item()
                    B, N_mod, C, H, W = t.shape
                    idx_mod = flat_idx // (C*H*W)
                    rem = flat_idx % (C*H*W)
                    ch = rem // (H*W)
                    rem2 = rem % (H*W)
                    y = rem2 // W
                    x = rem2 % W
                    info_min = {
                        'dataset_idx': idx,
                        'field': t_name,
                        'modality_index': idx_mod,
                        'channel': ch,
                        'y': y,
                        'x': x,
                        'value': global_min
                    }
        return {'min': info_min, 'max': info_max}


    dataset = H5LifelogDataset(
        h5_dir="Img_Data/train",
        preload=True,
        device='cuda',
        augment=True,
        resize=(224, 224),  # 내부에선 사용되지 않음
        aug_prob=0.7,
        seed=42,
        debug=True,
    )

    result = find_dataset_min_max(dataset)
    print("Global min:", result['min'])
    print("Global max:", result['max'])
    
    