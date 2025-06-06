import pickle

class LifelogDataset:
    def __init__(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            self.data = pickle.load(f)
        self.keys = list(self.data.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        sleep_dict, lifelog_dict = self.data[key]
        return key, sleep_dict, lifelog_dict
