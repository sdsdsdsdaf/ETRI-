import pickle

with open("Data_Dict/train_data_filtered_daily_linear_bit_mask.pkl", "rb") as f:
    full_data = pickle.load(f)

ten_keys = list(full_data.keys())[:1]
ten_samples = {k: full_data[k] for k in ten_keys}

with open("ten_samples.pkl", "wb") as f:
    pickle.dump(ten_samples, f)