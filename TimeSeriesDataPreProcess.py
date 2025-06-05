from functools import reduce
import glob, os
import pandas as pd
from datetime import datetime
from itertools import product
from IPython.display import display
from datetime import datetime
import pickle as pkl
import numpy as np
import ast


MultiModal_data_list = []
freqency = {
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

subject_id = [
    'id01', 'id02', 'id03', 'id04', 'id05', 'id06', 'id07', 'id08', 'id09', 'id10',
]

dir_name = "Data/PreProcessingData/"
file_name = "data.pkl"

top_10_labels = [
    "Inside, small room", "Speech", "Silence", "Music",
    "Narration, monologue", "Child speech, kid speaking",
    "Conversation", "Speech synthesizer", "Shout", "Babbling"
]

top_apps = [
    'One UI 홈', '카카오톡', '시스템 UI', 'NAVER', '캐시워크', '성경일독Q',
    'YouTube', '통화', '메시지', '타임스프레드', 'Instagram']

def process_mBle(df:pd.DataFrame):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date

    features = []

    for idx, row in df.iterrows():
        entry = ast.literal_eval(row['m_ble']) if isinstance(row['m_ble'], str) else row['m_ble']

        rssi_list = []
        class_0_cnt = 0
        class_other_cnt = 0

        for device in entry:
            try:
                rssi = int(device['rssi'])
                rssi_list.append(rssi)

                if str(device['device_class']) == 0:
                    class_0_cnt += 1
                else:
                    class_other_cnt += 1

            except:
                continue

        feature = {
            'subject_id': row['subject_id'],
            'date': row['date'],
            'device_class_0_cnt': class_0_cnt,
            'device_class_others_cnt': class_other_cnt,
            'device_count': len(rssi_list),
            'rssi_mean': np.mean(rssi_list) if rssi_list else np.nan,
            'rssi_min': np.min(rssi_list) if rssi_list else np.nan,
            'rssi_max': np.max(rssi_list) if rssi_list else np.nan,
        }
        features.append(feature)


    return pd.DataFrame(features)

def summarize_mBle_daily(df:pd.DataFrame):
        # row 단위 BLE feature 추출
    df = process_mBle(df)

        # 하루 단위로 cnt 합치기
    grouped = df.groupby(['subject_id', 'date']).agg({
            'device_class_0_cnt': 'sum',
            'device_class_others_cnt': 'sum',
            'rssi_mean': 'mean',
            'rssi_min': 'min',
            'rssi_max': 'max',
        }).reset_index()

        # 총합 구해서 비율 계산
    total_cnt = grouped['device_class_0_cnt'] + grouped['device_class_others_cnt']
    grouped['device_class_0_ratio'] = grouped['device_class_0_cnt'] / total_cnt.replace(0, np.nan)
    grouped['device_class_others_ratio'] = grouped['device_class_others_cnt'] / total_cnt.replace(0, np.nan)

    # 필요 없는 원래 cnt 컬럼 제거
    grouped.drop(columns=['device_class_0_cnt', 'device_class_others_cnt',], inplace=True)

    return grouped


def find_start_end_time(df: pd.DataFrame) -> tuple[datetime, datetime]:
    ts = pd.to_datetime(df['timestamp'])
    return ts.min(), ts.max()



def bulid_merge_key(start, end, freq) -> pd.DataFrame:
    timestamps = pd.date_range(start=start, end=end, freq=str(freq)+'min')

    return pd.DataFrame([
        {'subject_id': sid, 'timestamp': t}
        for sid in subject_id
        for t in timestamps
    ])

def find_drop_rows(
        df:pd.DataFrame, 
        merge_key:pd.DataFrame, 
        threshold:float=0.3, 
        continuous_time = 180, #min
        frequency = 1) -> set[tuple[str, str]]:

    continuous_time_block  = continuous_time // frequency
    drop_set = set()

    merged_df = pd.merge(df, merge_key, on=['subject_id', 'timestamp'], how='outer')
    merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'])
    merged_df['date'] = merged_df['timestamp'].dt.date

    value_cols = [col for col in merged_df.columns if col not in ['subject_id', 'timestamp', 'date']]

    for (subject_id, date), group in merged_df.groupby(['subject_id', 'date']):
        values = group[value_cols].values  # shape: (T, C)

        if values.shape[0] == 0:
            drop_set.add((str(subject_id), str(date)))
            continue

        for ch in range(values.shape[1]):
            col = values[:, ch]
            is_nan = pd.Series(col).isna().astype(int)

            ratio = is_nan.mean()
            max_gap = is_nan.groupby((is_nan != is_nan.shift()).cumsum()).sum().max()

            if ratio >= threshold or max_gap >= continuous_time_block:
                drop_set.add((subject_id, str(date)))
                break  # 한 채널이라도 걸리면 drop

    return drop_set

def filter_dropped_rows(df: pd.DataFrame, drop_set: set[tuple[str, str]]) -> pd.DataFrame:
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    # 강제 str 변환
    df['subject_id'] = df['subject_id'].astype(str)
    df['date'] = df['date'].astype(str)
    drop_set = {(str(s), str(d)) for (s, d) in drop_set}

    df = df[~df.apply(lambda row: (row['subject_id'], row['date']) in drop_set, axis=1)]

    return df


def process_mAmbience_top10(df:pd.DataFrame):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date

    # 초기화
    for label in top_10_labels + ['others']:
        df[label] = 0.0

    for idx, row in df.iterrows():
        parsed = ast.literal_eval(row['m_ambience']) if isinstance(row['m_ambience'], str) else row['m_ambience']
        others_prob = 0.0

        for label, prob in parsed:
            prob = float(prob)
            if label in top_10_labels:
                df.at[idx, label] = prob
            else:
                others_prob += prob

        df.at[idx, 'others'] = others_prob

    return df.drop(columns=['m_ambience'])

def process_mGps(df):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date

    features = []

    for idx, row in df.iterrows():
        gps_list = ast.literal_eval(row['m_gps']) if isinstance(row['m_gps'], str) else row['m_gps']

        altitudes = []
        latitudes = []
        longitudes = []
        speeds = []

        for entry in gps_list:
            try:
                altitudes.append(float(entry['altitude']))
                latitudes.append(float(entry['latitude']))
                longitudes.append(float(entry['longitude']))
                speeds.append(float(entry['speed']))
            except:
                continue
        
        features.append({
            'subject_id': row['subject_id'],
            'date': row['date'],
            'timestamp': row['timestamp'],
            'altitude': np.mean(altitudes) if altitudes else np.nan,
            'altitude_std': np.std(altitudes) if altitudes else np.nan,
            'latitude': np.mean(latitudes) if latitudes else np.nan,
            'latitude_std': np.std(latitudes) if latitudes else np.nan,
            'longitude': np.mean(longitudes) if longitudes else np.nan,
            'longitude_std': np.std(longitudes) if longitudes else np.nan,
            'speed': np.mean(speeds) if speeds else np.nan,
            'speed_std': np.std(speeds) if speeds else np.nan,
        })

    return pd.DataFrame(features)

def process_mUsageStats(df:pd.DataFrame)->pd.DataFrame:
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    features = []

    for (subj, ts), group in df.groupby(['subject_id', 'timestamp']):
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
            'timestamp': ts,
            'date': ts.date(),
            'others_time': others_time
        }
        # 각 앱별 컬럼 추가
        feature.update({f'{app}_time': app_time[app] for app in top_apps})
        features.append(feature)

    return pd.DataFrame(features)

def process_mWifi(df:pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    results = []

    for (subj, ts), group in df.groupby(['subject_id', 'timestamp']):
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
            'timestamp': ts,
            'date': ts.date(),
            'wifi_rssi_mean': np.mean(rssi_all) if rssi_all else np.nan,
            'wifi_rssi_min': np.min(rssi_all) if rssi_all else np.nan,
            'wifi_rssi_max': np.max(rssi_all) if rssi_all else np.nan,
            'wifi_detected_cnt': len(rssi_all)
        })

    return pd.DataFrame(results)

def process_wHr(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['hr_mean'] = pd.to_numeric(df['heart_rate'], errors='coerce')  # 각 row의 heart_rate 사용
    df.drop(columns=['heart_rate'], inplace=True)
    return df

def interpolates_with_mask(lifelog_data:dict[str, pd.DataFrame], method='linear'):
    
    interpolated_results = {}
    os.makedirs(dir_name, exist_ok=True)
    for name, df in lifelog_data.items():
        print(f"⏳ {name} 전처리 중..")
        if 'timestamp' in df.columns:
            df = df.sort_values(by="timestamp")
        else:
            df = df.sort_values(by="date")

        num_df = df.select_dtypes(include='number')

        nan_mask = num_df.isna()
        df = df.infer_objects(copy=False)
        df_interp = num_df.interpolate(method=method)

        # 원본 df에 보간된 수치형 덮어쓰기
        df.update(df_interp)        

        # 보간된 위치 마스크
        mask = nan_mask & df_interp.notna()
        mask_with_id = mask.copy()
        mask_with_id['subject_id'] = df['subject_id'].values
        if 'timestamp' in df.columns:
            mask_with_id['timestamp'] = df['timestamp'].values
        else:
            mask_with_id['date'] = df['date'].values

        interpolated_results[name] = {
            'data': df,
            'mask': mask_with_id
        }
        print(f"✅ {name} 전처리 완료")
        interpolated_results[name]['data'].to_csv(f"{dir_name}{name}{method}_interpolates.csv")
    # interpolation 결과 저장
    with open(f"{dir_name}Interpolates_all_day_{method}_{file_name}", 'wb') as f:
        pkl.dump(interpolated_results, f)
        print("✅ File Saved")

    return interpolated_results

def interpolates_with_mask_daily(lifelog_data:dict[str, pd.DataFrame], method='linear'):
    interpolated_results = {}
    os.makedirs(dir_name, exist_ok=True)

    for name, df in lifelog_data.items():
        print(f"⏳ {name} 전처리 중..")

        df = df.copy()
        interpolated_dfs = []
        mask_dfs = []

        for (subj, date), group in df.groupby(['subject_id', 'date']):

            if 'timestamp' in group.columns:
                group = group.sort_values(by="timestamp")
            else:
                group = group.sort_values(by="date")

            num_df = group.select_dtypes(include='number')
            nan_mask = num_df.isna()

            num_df = num_df.infer_objects(copy=False)
            df_interp = num_df.interpolate(method=method, limit_direction='both')
            df_interp.index = group.index
            mask = nan_mask & df_interp.notna()

            group.update(df_interp) 

            mask = mask.copy()
            mask['subject_id'] = group['subject_id'].values
            mask['timestamp'] = group['timestamp'].values
            mask_dfs.append(mask)

            interpolated_dfs.append(group)

        interpolated_df = pd.concat(interpolated_dfs, ignore_index=True)
        mask_df = pd.concat(mask_dfs, ignore_index=True)

        interpolated_results[name] = {
            'data': interpolated_df,
            'mask': mask_df
        }
        print(f"✅ {name} 전처리 완료")
        interpolated_results[name]['data'].to_csv(f"{dir_name}{name}_daily_{method}_interpolates.csv")
        print("✅ File Saved")

    # interpolation 결과 저장 (일별)
    with open(f"{dir_name}Interpolates_per_day_{method}_{file_name}", 'wb') as f:
        pkl.dump(interpolated_results, f)
        print("✅ File Saved")

    return interpolated_results


def preprocessing(method: str = 'linear'):
    load_path_all = f"{dir_name}Interpolates_all_day_{method}_{file_name}"
    load_path_per = f"{dir_name}Interpolates_per_day_{method}_{file_name}"

    if os.path.exists(load_path_all):
        with open(load_path_all, 'rb') as f:
            interpolated_results = pkl.load(f)
            print("✅ File Loaded")
    elif os.path.exists(load_path_per):
        with open(load_path_per, 'rb') as f:
            interpolated_results = pkl.load(f)
            print("✅ File Loaded")
    else:
        data_dir = "Data\ch2025_data_items"

        # Parquet 파일 전체 경로 리스트
        parquet_files = glob.glob(os.path.join(data_dir, 'ch2025_*.parquet'))
        parquet_files

        # Parquet 파일을 읽어 DataFrame으로 변환
        lifelog_data:dict[str, pd.DataFrame] = {}

        for file_path in parquet_files:
            name = os.path.basename(file_path).replace('.parquet', '').replace('ch2025_', '')
            lifelog_data[name] = pd.read_parquet(file_path)
            print(f"✅ Loaded: {name}, shape = {lifelog_data[name].shape}")
            MultiModal_data_list.append(name)

        print()
        for name, df in lifelog_data.items():
            if name == 'mBle':
                continue

            start_time, end_time = find_start_end_time(df)

            merge_key = bulid_merge_key(start_time, end_time, freqency[name])
            drop_set = find_drop_rows(df, merge_key, threshold=0.3, continuous_time=120, frequency=freqency[name])
            print(f"Drop Set Num: {len(drop_set)}")

            filterd_df = filter_dropped_rows(df, drop_set)
            lifelog_data[name] = filterd_df

            print(f"{name}: {filterd_df.shape}")

        lifelog_data['mBle'] = summarize_mBle_daily(lifelog_data['mBle'])
        lifelog_data['mAmbience'] = process_mAmbience_top10(lifelog_data['mAmbience'])
        lifelog_data['mGps'] = process_mGps(lifelog_data['mGps'])
        lifelog_data['mUsageStats'] = process_mUsageStats(lifelog_data['mUsageStats'])
        lifelog_data['mWifi'] = process_mWifi(lifelog_data['mWifi'])
        lifelog_data['wHr'] = process_wHr(lifelog_data['wHr'])

        #TODO 보간 함수 삽입
        interpolated_results = interpolates_with_mask(lifelog_data, method)
        #interpolated_results = interpolates_with_mask_daily(lifelog_data, method)
    
    return interpolated_results

def data_load_and_split_test_and_train():
    # 기본 보간 방법은 linear
    interpolated_results = preprocessing()

    print(interpolated_results['wHr']['data'].shape)
    print(interpolated_results['wHr']['mask'].shape)

if __name__ == "__main__":
    data_load_and_split_test_and_train()
    






            


