import numpy as np
import pandas as pd
import glob
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from typing import Optional, Union
from functools import reduce


MultiModal_data_list = ['mACStatus', 'mActivity', 'mAmbience', 'mBle', 'mGps', 'mLight',
                        'mScreenStatus', 'mUsageStats', 'mWifi', 'wHr', 'wLight', 'wPedo']

MultiModal_data_with_time = ['mGps', 'mLight', 'mScreenStatus', 'wHr', 'wLight', 'wPedo']

metrics_train = pd.read_csv('Data\ch2025_metrics_train.csv')
sample_submission = pd.read_csv('Data\ch2025_submission_sample.csv')

top_10_labels = [
    "Inside, small room", "Speech", "Silence", "Music",
    "Narration, monologue", "Child speech, kid speaking",
    "Conversation", "Speech synthesizer", "Shout", "Babbling"
]

block_num  = 6

# ✅ 기준 쌍 (subject_id, lifelog_date)
sample_submission['lifelog_date'] = pd.to_datetime(sample_submission['lifelog_date'])
test_keys = set(zip(sample_submission['subject_id'], sample_submission['lifelog_date'].dt.date)) 

def get_time_block(timestamp: Union[pd.Series, pd.Timestamp], block_num = 24) -> int:

    assert 24 % block_num == 0, "block_num must be a divisor of 24"
    hours_per_block = 24 // block_num
    return timestamp.dt.hour // hours_per_block


#TODO: TRAIN데이터에서 각 ROW마다 데이터 있는지 확인 후 INNER JOIN으로 MERGE하게 변경
def get_common_keys(preprocessed_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    """
    key_dfs = [df[['subject_id', 'date']] for df in preprocessed_dict.values()]
    common_keys = reduce(lambda left, right: pd.merge(left, right, on=['subject_id', 'date'], how='inner'), key_dfs)
    return common_keys.drop_duplicates()


def preprocess_data(df: pd.DataFrame, timestamp_col: str = 'timestamp', SD: Optional[int] = 42) -> tuple:
    # seed 고정
    if SD is not None:
        random.seed(SD)
        np.random.seed(SD)
        os.environ['PYTHONHASHSEED'] = str(SD)

    data_dir = "Data\ch2025_data_items"

    # Parquet 파일 전체 경로 리스트
    parquet_files = glob.glob(os.path.join(data_dir, 'ch2025_*.parquet'))
    parquet_files

    # Parquet 파일을 읽어 DataFrame으로 변환
    lifelog_data:dict[pd.DataFrame] = {}

    for file_path in parquet_files:
        name = os.path.basename(file_path).replace('.parquet', '').replace('ch2025_', '')
        lifelog_data[name] = pd.read_parquet(file_path)
        print(f"✅ Loaded: {name}, shape = {lifelog_data[name].shape}")


    dataFrames = {
        q: (lifelog_data[q], 'timestamp') for q in MultiModal_data_list
    }

    preprocess_data_dict = {}

    for name in MultiModal_data_list:
        preprocess_data_dict[f"{name}"] = globals()[f"process_{name}"](lifelog_data[name])
        print(f"✅ Processed: {name}, shape = {preprocess_data_dict[f'{name}'].shape}")
        print(preprocess_data_dict[f"{name}"])

    # common_keys
    common_keys = get_common_keys(preprocess_data_dict)
    print(f"✅ Common keys found: {common_keys.shape[0]}")

    for name, df in preprocess_data_dict.items():
        # Merge with common keys
        preprocess_data_dict[name] = pd.merge(common_keys, df, on=['subject_id', 'date'], how='inner')
        print(f"✅ Merged {name} with common keys, shape = {preprocess_data_dict[name].shape}")

    return 0
    

    
def split_test_train(df: pd.DataFrame, subject_col = "subject_id", timestamp_col = "timestamp") -> tuple[pd.DataFrame, pd.DataFrame]:

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
    df  = df.dropna(subset=[timestamp_col])
    df['date_only'] = df[timestamp_col].dt.date
    df['key'] = list(zip(df[subject_col], df['date_only']))

    test_df = df[df['key'].isin(test_keys)].drop(columns=['date_only', 'key'])
    train_df = df[~df['key'].isin(test_keys)].drop(columns=['date_only', 'key'])

    return train_df, test_df

def process_mACStatus(df: pd.DataFrame) -> pd.DataFrame:

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df = df.sort_values(['subject_id', 'timestamp'])

    results = []

    for (subject_id, date), group in df.groupby(['subject_id', 'date']):
        status = group['m_charging'].values

        # Charging Status Ratio
        ratio_charging = status.mean()

        # Number of Charging transition Events
        transitions = (status[1:] != status[:-1]).sum()

        lengths = []
        current_len = 0

        for val in status:
            if val == 1:
                current_len += 1
            elif current_len > 0:
                lengths.append(current_len)
                current_len = 0
            
        if current_len > 0:
            lengths.append(current_len)

        avg_charging_duration = np.mean(lengths) if lengths else 0
        max_charging_duration = np.max(lengths) if lengths else 0

        results.append({
            'subject_id': subject_id,
            'date': date,
            'charging_status_ratio': ratio_charging,
            'charging_transition_events': transitions,
            'avg_charging_duration': avg_charging_duration,
            'max_charging_duration': max_charging_duration
        })
    
    return pd.DataFrame(results)

def process_mActivity(df: pd.DataFrame) -> pd.DataFrame:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df = df.sort_values(['subject_id', 'timestamp'])

    summary = []

    for (subj, date), group in df.groupby(['subject_id', 'date']):
        counts = group['m_activity'].value_counts(normalize=True)  # 비율
        row = {'subject_id': subj, 'date': date}

        # 0~8 비율 저장
        for i in range(9):
            row[f'activity_{i}_ratio'] = counts.get(i, 0)

        # 주요 활동 정보
        row['dominant_activity'] = group['m_activity'].mode()[0]
        row['num_unique_activities'] = group['m_activity'].nunique()

        summary.append(row)

    return pd.DataFrame(summary)

def process_mAmbience_top_10(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date

    #initialize
    for label in top_10_labels:
        df[label] = 0.

    for idx, row in df.iterrows():
        parsed = ast.literal_eval(row['m_ambience']) if isinstance(row['m_ambience'], str) else row['m_ambience']
        other_prob = 0.

        for label, prob in parsed:
            prob = float(prob)
            if label in top_10_labels:
                df.at[idx, label] = prob
            else:
                other_prob += prob

        df.at[idx, 'others'] = other_prob

    return df.drop(columns=['m_ambience'])

def summarize_mAmbience(df: pd.DataFrame) -> pd.DataFrame:
    prob_cols = [col for col in df.columns if col not in ['subject_id', 'timestamp', 'date']]

    # summarize by subject_id and date
    daily_summary = df.groupby(['subject_id', 'date'])[prob_cols].mean().reset_index()
    return daily_summary

def process_mAmbience(df:pd.DataFrame) -> pd.DataFrame:
    df_top10 = process_mAmbience_top_10(df)
    summary_df = summarize_mAmbience(df_top10)
    return summary_df

def summaarize_mBle_daliy(df: pd.DataFrame) -> pd.DataFrame:

    grouped = df.groupby(['subject_id', 'date']).agg({
        'device_class_0_cnt': 'sum',
        'device_class_others_cnt': 'sum',
        'rssi_mean': 'mean',
        'rssi_min': 'min',
        'rssi_max': 'max',
    }).reset_index()

    total_cnt = grouped['device_class_0_cnt'] + grouped['device_class_others_cnt']
    grouped['device_class_0_ratio'] = grouped['device_class_0_cnt'] / total_cnt.replace(0, np.nan)
    grouped['device_class_others_ratio'] = grouped['device_class_others_cnt'] / total_cnt.replace(0, np.nan)

    grouped.drop(columns=['device_class_0_cnt', 'device_class_others_cnt'], inplace=True)

    return grouped


def process_mBle(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date

    features = []

    for idx, row in df.iterrows():
        entry = ast.literal_eval(row['m_ble']) if isinstance(row['m_ble'], str) else row['m_ble']

        rssi_list = []
        class_0_count = 0
        class_other_count = 0

        for device in entry:
            try:
                rssi = int(device['rssi'])
                rssi_list.append(rssi)

                if str(device['device_class']) == '0':
                    class_0_count += 1
                else:
                    class_other_count += 1
            except:
                continue  # malformed record

        feature = {
            'subject_id': row['subject_id'],
            'date': row['date'],
            'device_class_0_cnt': class_0_count,
            'device_class_others_cnt': class_other_count,
            'device_count': len(rssi_list),
            'rssi_mean': np.mean(rssi_list) if rssi_list else np.nan,
            'rssi_min': np.min(rssi_list) if rssi_list else np.nan,
            'rssi_max': np.max(rssi_list) if rssi_list else np.nan,
        }
        features.append(feature)

    df = pd.DataFrame(features)
    return summaarize_mBle_daliy(df)

def process_mGps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['block'] = get_time_block(df['timestamp'], block_num=8)
    df = df.sort_values(['subject_id', 'timestamp'])

    features = []

    for idx, row in df.iterrows():
        gps_list = ast.literal_eval(row['m_gps']) if isinstance(row['m_gps'], str) else row['m_gps']

        altitudes = []
        latitudes = []
        longitudes = []
        speeds = []

        for entry in gps_list:
            try:
                lat = float(entry['lat'])
                lon = float(entry['lon'])
                alt = float(entry['alt'])
                speed = float(entry['speed'])

                latitudes.append(lat)
                longitudes.append(lon)
                altitudes.append(alt)
                speeds.append(speed)
            except (KeyError, ValueError, TypeError):
                continue
        features.append({
            'subject_id': row['subject_id'],
            'date': row['date'],
            'block': row['block'],
            'altitude_mean': np.mean(altitudes) if altitudes else np.nan,
            'latitude_std': np.std(latitudes) if latitudes else np.nan,
            'longitude_std': np.std(longitudes) if longitudes else np.nan,
            'speed_mean': np.mean(speeds) if speeds else np.nan,
            'speed_max': np.max(speeds) if speeds else np.nan,
            'speed_std': np.std(speeds) if speeds else np.nan,
        })
    m_Gps_df = pd.DataFrame(features)
    m_Gps_df = m_Gps_df.groupby(['subject_id', 'date', 'block']).agg({
        'altitude_mean': 'mean',
        'latitude_std': 'mean',
        'longitude_std': 'mean',
        'speed_mean': 'mean',
        'speed_max': 'max',
        'speed_std': 'mean'
    }).reset_index()

    return m_Gps_df

def process_mLight(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['block'] = get_time_block(df['timestamp'], block_num=8)
    df['is_night'] = df['timestamp'].dt.hour.astype('int').apply(lambda h: h >= 22 or h < 6)

    hourly = df.groupby(['subject_id', 'date', 'block']).agg(
        light_mean=('m_light', 'mean'),
        light_std=('m_light', 'std'),
        light_max=('m_light', 'max'),
        light_min=('m_light', 'min'),
    ).reset_index()

    return hourly

def process_mScreenStatus(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date

    features = []

    for (subj, date), group in df.groupby(['subject_id', 'date']):
        status = group['m_screen_use'].values
        ratio_on = status.mean()
        transitions = (status[1:] != status[:-1]).sum()

        # 연속된 1 상태 길이들
        durations = []
        current = 0
        for val in status:
            if val == 1:
                current += 1
            elif current > 0:
                durations.append(current)
                current = 0
        if current > 0:
            durations.append(current)

        features.append({
            'subject_id': subj,
            'date': date,
            'screen_on_ratio': ratio_on,
            'screen_on_transitions': transitions,
            'screen_on_duration_avg': np.mean(durations) if durations else 0,
            'screen_on_duration_max': np.max(durations) if durations else 0,
        })

    return pd.DataFrame(features)

top_apps = [
    'One UI 홈', '카카오톡', '시스템 UI', 'NAVER', '캐시워크', '성경일독Q',
    'YouTube', '통화', '메시지', '타임스프레드', 'Instagram']

def process_mUsageStats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date

    features = []

    for (subj, date), group in df.groupby(['subject_id', 'date']):
        app_time = {app: 0 for app in top_apps}
        others_time = 0

        for row in group['m_usage_stats']:
            parsed = ast.literal_eval(row) if isinstance(row, str) else row
            for entry in parsed:
                app = entry.get('app_name')
                time = entry.get('total_time', 0)
                if app in top_apps:
                    app_time[app] += int(time)
                else:
                    others_time += int(time)

        feature = {
            'subject_id': subj,
            'date': date,
            'others_time': others_time
        }
        feature.update({f'{app}_time': app_time[app] for app in top_apps})

        features.append(feature)

    return pd.DataFrame(features)

def process_wHr(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['block'] = get_time_block(df['timestamp'], block_num=8)

    results = []

    for (subj, date), group in df.groupby(['subject_id', 'date']):
        block_stats = {'subject_id': subj, 'date': date}

        for block, block_group in group.groupby('block'):
            hr_all = []
            for row in block_group['heart_rate']:
                parsed = ast.literal_eval(row) if isinstance(row, str) else row
                hr_all.extend([int(h) for h in parsed if h is not None])

            if not hr_all:
                continue

            above_100 = [hr for hr in hr_all if hr > 100]
            block_stats[f'hr_{block}_mean'] = np.mean(hr_all)
            block_stats[f'hr_{block}_std'] = np.std(hr_all)
            block_stats[f'hr_{block}_max'] = np.max(hr_all)
            block_stats[f'hr_{block}_min'] = np.min(hr_all)
            block_stats[f'hr_{block}_above_100_ratio'] = len(above_100) / len(hr_all)

        results.append(block_stats)

    return pd.DataFrame(results)

def process_mWifi(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date

    results = []

    for (subj, date), group in df.groupby(['subject_id', 'date']):
        rssi_all = []

        for row in group['m_wifi']:
            parsed = ast.literal_eval(row) if isinstance(row, str) else row
            for ap in parsed:
                try:
                    rssi = int(ap['rssi'])
                    rssi_all.append(rssi)
                except:
                    continue

        results.append({
            'subject_id': subj,
            'date': date,
            'wifi_rssi_mean': np.mean(rssi_all) if rssi_all else np.nan,
            'wifi_rssi_min': np.min(rssi_all) if rssi_all else np.nan,
            'wifi_rssi_max': np.max(rssi_all) if rssi_all else np.nan,
            'wifi_detected_cnt': len(rssi_all)
        })

    return pd.DataFrame(results)

def process_wLight(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['block'] = get_time_block(df['timestamp'], block_num=8)

    results = []

    for (subj, date), group in df.groupby(['subject_id', 'date']):
        block_stats = {'subject_id': subj, 'date': date}

        for block, block_group in group.groupby('block'):
            lux = block_group['w_light'].dropna().values
            if len(lux) == 0:
                continue

            block_stats[f'wlight_{block}_mean'] = np.mean(lux)
            block_stats[f'wlight_{block}_std'] = np.std(lux)
            block_stats[f'wlight_{block}_max'] = np.max(lux)
            block_stats[f'wlight_{block}_min'] = np.min(lux)

        results.append(block_stats)

    return pd.DataFrame(results)


def process_wPedo(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['block'] = get_time_block(df['timestamp'], block_num=8)

    summary = df.groupby(['subject_id', 'date', 'block']).agg({
        'step': 'sum',
        'step_frequency': 'mean',
        'distance': 'sum',
        'speed': ['mean', 'max'],
        'burned_calories': 'sum'
    }).reset_index()

    # 컬럼 이름 정리
    summary.columns = ['subject_id', 'date', 'block',
                       'step_sum', 'step_frequency_mean',
                       'distance_sum', 'speed_mean', 'speed_max',
                       'burned_calories_sum']

    return summary

if __name__ == "__main__":

    import os
    dir_name = 'Data/Preprocessing'
    if not os.path.exists(dir_name+"/train.pkl") or not os.path.exists(dir_name+"/test.pkl"):
        print("❗ Data files not found. Please ensure 'ch2025_metrics_train.csv' and 'ch2025_submission_sample.csv' exist in the 'Data' directory.")
        train_dict, test_dict = preprocess_data(metrics_train)
        print("✅ Data preprocessing completed.")
    
    # Add more print statements for other datasets as needed



        
