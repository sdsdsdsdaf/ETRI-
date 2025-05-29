from torch.utils.data import Dataset, DataLoader


MultiModal_data_list = ['mACStatus', 'mActivity', 'mAmbience', 'mBle', 'mGps', 'mLight',
                        'mScreenStatus', 'mUsageStats', 'mWifi', 'wHr', 'wLight', 'wPedo'] 

class ETRIDataLoader(DataLoader):
    pass


