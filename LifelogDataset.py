from torch.utils.data import Dataset
from typing import Dict, Tuple, List
import torch
import pandas as pd

MASK = 1
DATA = 0

class LifelogDataset(Dataset): #TODO 후에 마스크를 한층으로 할 것도 고려-> (피처수, 1440)가 아니라 (1, 1440으로)
    """
    PyTorch Dataset for multi-modality lifelog data.
    Each sample contains:
        - A dictionary of modality data tensors (modal_name: Tensor of shape (C, T))
        - A corresponding dictionary of mask tensors
        - A label (classification target)
    
    Features:
        - Optionally exclude certain modalities
        - Optionally reduce mask tensors from (C, T) → (1, T) by averaging across channels
        - Optional per-sample transform hooks for data and mask
        - Device-aware tensor movement

    OutPut:
        Tuple[Dict[str, Tensor], Dict[str, Tensor], Tensor]: 
        - sleep_data_tensor: dict of (modal_name: Tensor (C + 1, T) or (C + C, T))
        - lifelog_data_tensor: dict of (modal_name: Tensor (C + 1, T) or (C + C, T))
        - label_tensor: int label as tensor

    """

    def __init__(
        self,
        tensor_data_dict: 
            Dict[
                Tuple[str, str, str], 
                Tuple[
                    Dict[Tuple[str], 
                         Tuple[pd.DataFrame, pd.DataFrame]],    # sleep_date_dict
                    Dict[Tuple[str], 
                         Tuple[pd.DataFrame, pd.DataFrame]]]],  # lifelog_date_dict)

        label_dict: Dict[str, int],
        mask_ch_to_1dim: bool = False,
        exclude_modalities: List = None,
        data_transforms = None,
        mask_transforms = None,  # key: subject_id, value: int label
        device = torch.device('cpu'),
        feature_dim_map = None):
    
        self.transforms = data_transforms
        self.mask_transforms = mask_transforms
        self.device = device
        self.mask_ch_to_1dim = mask_ch_to_1dim
        self.feature_dim_map = None

        self.sleep_dataes = []  # list of dict[str, Tensor]: data per sample
        self.lifelog_dataes = []  # list of dict[str, Tensor]: data per sample
        self.labels = []  # list of Tensor: labels per sample

        self.frequency = {
            'mACStatus':    1,
            'mActivity':	1,
            'mAmbience':	2,
            'mBle':	10,
            'mGps':	1,
            'mLight':	10,
            'mScreenStatus':	1,
            'mUsageStats':	10,
            'mWifi':	10,
            'wHr':	1,
            'wLight':	1,
            'wPedo':	1,
        }
        if feature_dim_map is None:
            feature_dim_map = {}
            seen_modalities = set()
            for _, (sleep_data, lifelog_data) in tensor_data_dict.items():
                for source_data in [sleep_data[DATA], lifelog_data[DATA]]:
                    for name, df in source_data.items():
                        if df is not None and name not in feature_dim_map:
                            feature_dim_map[name] = df.shape[1]
                            seen_modalities.add(name)

            expected_modalities = set(self.frequency.keys())
            missing_modalities = expected_modalities - seen_modalities
            if missing_modalities:
                raise ValueError(
                    f"Missing modalities: {missing_modalities}"
                )
            
        self.feature_dim_map = feature_dim_map
        for (subject_id, sleep_date, lifelog_date), (sleep_date_data, lifelog_date_data) in tensor_data_dict.items():

            # Ensure 'mBle' key exists (default to None if missing)
            sleep_date_data[MASK]['mBle'] = sleep_date_data[MASK].get('mBle', None)
            lifelog_date_data[MASK]['mBle'] = lifelog_date_data[MASK].get('mBle', None)

            sleep_data_dict_filtered = {
                k: v.clone() if (exclude_modalities is None or k not in exclude_modalities) and v is not None else None
                for k, v in sleep_date_data[DATA].items()
            }
            lifelog_data_dict_filtered = {
                k: v.clone() if (exclude_modalities is None or k not in exclude_modalities) and v is not None else None
                for k, v in lifelog_date_data[DATA].items()
            }

            sleep_date_mask_dict_filtered = {
                k: v.clone() if (exclude_modalities is None or k not in exclude_modalities) and v is not None else None
                for k, v in sleep_date_data[MASK].items()
            }
            lifelog_date_mask_dict_filtered = {
                k: v.clone() if (exclude_modalities is None or k not in exclude_modalities) and v is not None else None
                for k, v in lifelog_date_data[MASK].items()
            }

            # Optionally reduce channel dimension for masks (e.g., (C, T) → (1, T))
            if self.mask_ch_to_1dim:
                sleep_date_mask_dict_filtered = self.reduce_mask_channels(sleep_date_mask_dict_filtered)
                lifelog_date_mask_dict_filtered = self.reduce_mask_channels(lifelog_date_mask_dict_filtered)

            sleep_data_dict_filtered_tensor = self.dataframe_to_tensor(sleep_data_dict_filtered)
            lifelog_data_dict_filtered_tensor = self.dataframe_to_tensor(lifelog_data_dict_filtered)
            sleep_date_mask_dict_filtered_tensor = self.dataframe_to_tensor(sleep_date_mask_dict_filtered)
            lifelog_date_mask_dict_filtered_tensor = self.dataframe_to_tensor(lifelog_date_mask_dict_filtered)

            sleep_dataes_with_mask = self.combine_data_and_mask(
                sleep_data_dict_filtered_tensor,
                sleep_date_mask_dict_filtered_tensor,
                
            )
            life_dataes_with_mask = self.combine_data_and_mask(
                lifelog_data_dict_filtered_tensor,
                lifelog_date_mask_dict_filtered_tensor,
            )

            self.sleep_dataes.append(sleep_dataes_with_mask)
            self.lifelog_dataes.append(life_dataes_with_mask)
            label: torch.Tensor = torch.tensor(label_dict[subject_id], dtype=torch.long).contiguous()
            self.labels.append(label)

    def __len__(self):
        assert len(self.sleep_dataes) == len(self.lifelog_dataes) == len(self.labels), "Doesn't match Data`s shape"
        return len(self.labels)
    
    def combine_data_and_mask(
        self,
        data: Dict[str, torch.Tensor],
        mask: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        각 modality에 대해 data (C, T)와 mask (1 또는 C, T)를 붙여서
        (C+1, T) 또는 (C+C, T) 형태로 반환.

        mask가 (T,)이면 (1, T)로 reshape.
        """
        combined = {}
        for key in data:
            d = data[key]
            m = mask.get(key, None)

            if m is None:
                raise ValueError("Mask not found for key")

            # mask가 1D면 (1, T)로 reshape
            if m.dim() == 1:
                m = m.unsqueeze(0)
        
            # 최종 concat
            combined[key] = torch.cat([d, m], dim=0)

        return combined

    def dataframe_to_tensor(self, df_dict:Dict[str, pd.DataFrame]) -> torch.Tensor: #None일 경우 즉 존재하지 않을 경우 zero padding
        total_time_block = 1440 // self.frequency
        result: Dict[str, torch.Tensor] = {}
        

        for name, df in df_dict.items():
            if df is not None:
                result[name] = torch.tensor(df.values.T, dtype=torch.float32).contiguous()
            else:
                dim = self.feature_dim_map.get(name)
                if dim is None:
                    raise ValueError(f"feature_dim_map에 {name} modality 정보가 없습니다.")
                result[name] = torch.zeros((self.feature_dim_map.get(name, None), total_time_block ))
                result[name] = result[name].contiguous()


        return result

    
    def move_to_device(self, tensor_dict:Dict[str, torch.Tensor]):
        """ 
        Move all tensors in a dict to the specified device.
        Skip None entries.
        """

        return {
        k: v.to(self.device) if v is not None else None
        for k, v in tensor_dict.items()
    }

    def reduce_mask_channels(self, mask_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        For each modality's mask tensor of shape (C, T), reduce it to (1, T) via channel-wise mean.
        """

        return {
            k: v.mean(dim=0, keepdim=True) if v is not None else None  # shape: (1, T)
            for k, v in mask_dict.items()
        }
        

    def __getitem__(self, idx):
        """
        Return a single sample (data, mask, label) with optional transforms and device placement.
        """

        sleep_data_tensor = self.sleep_dataes[idx]
        lifelog_data_tensor = self.lifelog_dataes[idx]
        label_tensor:torch.Tensor = self.labels[idx]


        if self.transforms:
            sleep_data_tensor = self.transforms(sleep_data_tensor)
        if self.mask_transforms:
            lifelog_data_tensor = self.mask_transforms(lifelog_data_tensor)

        sleep_data_tensor = self.move_to_device(sleep_data_tensor)
        lifelog_data_tensor = self.move_to_device(lifelog_data_tensor)
        label_tensor = label_tensor.to(self.device)

        return sleep_data_tensor, lifelog_data_tensor, label_tensor
