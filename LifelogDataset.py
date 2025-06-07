import pickle
from torch.utils.data import Dataset
import torch
import os
import h5py


from typing import Tuple

class H5LifelogDataset(Dataset):
    def __init__(self, h5_dir: str, transform=None, preload: bool = False, input_type_dict: dict = None):
        self.h5_dir = h5_dir
        self.file_list = [f for f in os.listdir(h5_dir) if f.endswith(".h5")]
        self.transform = transform
        self.preload = preload
        self.data = []
        self.input_type_dict = input_type_dict


        if preload:
            print("[INFO] Preloading all data into RAM...")
            for fname in self.file_list:
                path = os.path.join(h5_dir, fname)
                with h5py.File(path, 'r') as f:
                    sample = self._load_sample(f)
                    self.data.append(sample)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx: int):
        if self.preload:
            return self.data[idx]

        path = os.path.join(self.h5_dir, self.file_list[idx])
        with h5py.File(path, 'r') as f:
            return self._load_sample(f)

    def _load_sample(self, f: h5py.File) -> Tuple[dict, torch.Tensor]:
        """
        Args:
            f: HDF5 file object
            input_type_dict: dictionary like {'plot': ['mLight', 'wHr'], 'heatmap': ['mAcc'], 'mble': ['mBle']}
        Returns:
            sample: dict with keys:
                - 'tensor_sleep': (N, C, H, W) tensor
                - 'tensor_lifelog': (N, C, H, W) tensor
                - 'mble_data_sleep': (N, F) tensor
                - 'mble_data_lifelog': (N, F) tensor
                - 'modality_names': List[str]
            label: torch.Tensor (multi-task label)
        """
        modality_names = [s.decode() for s in f.attrs["modality_names"]]
        self.input_type_dict = {
            'plot': [m for m in modality_names if m != "mBle"],
            'mBle': ["mBle"]
        }

        sleep_tensors = []
        lifelog_tensors = []
        mble_sleep_tensors = []
        mble_lifelog_tensors = []

        for mode, modality_list in self.input_type_dict.items():
            n = len(modality_list)

            if mode == "plot":
                full_sleep_tensor = torch.tensor(f["tensor_plot_sleep"][:]).float()  # (N, C, H, W)
                full_lifelog_tensor = torch.tensor(f["tensor_plot_lifelog"][:]).float()
                for i in range(n):
                    sleep_tensors.append(full_sleep_tensor[i])
                    lifelog_tensors.append(full_lifelog_tensor[i])

            elif mode == "heatmap":
                full_sleep_tensor = torch.tensor(f["tensor_heatmap_sleep"][:]).float()
                full_lifelog_tensor = torch.tensor(f["tensor_heatmap_lifelog"][:]).float()
                for i in range(n):
                    sleep_tensors.append(full_sleep_tensor[i])
                    lifelog_tensors.append(full_lifelog_tensor[i])

            elif mode == "mBle":
                sleep_tensor = torch.tensor(f["mble_data_sleep"][:]).float().unsqueeze(0)  # (1, F)
                lifelog_tensor = torch.tensor(f["mble_data_lifelog"][:]).float().unsqueeze(0)
                for _ in modality_list:
                    mble_sleep_tensors.append(sleep_tensor)
                    mble_lifelog_tensors.append(lifelog_tensor)

            else:
                raise ValueError(f"Unsupported input type: {mode}")

        sample = {
            "tensor_sleep": torch.stack(sleep_tensors, dim=0) if sleep_tensors else None,  # (N, C, H, W)
            "tensor_lifelog": torch.stack(lifelog_tensors, dim=0) if lifelog_tensors else None,
            "mble_data_sleep": torch.cat(mble_sleep_tensors, dim=0) if mble_sleep_tensors else None,  # (N, F)
            "mble_data_lifelog": torch.cat(mble_lifelog_tensors, dim=0) if mble_lifelog_tensors else None,
            "modality_names": modality_names,
        }

        label = torch.tensor(f["label"][:]).int() if "label" in f else None
        if label is None:
            raise ValueError("Label not found in HDF5 file.")

        return sample, label


if __name__ == "__main__":
    data, label = next(iter(H5LifelogDataset("Img_Data/train")))