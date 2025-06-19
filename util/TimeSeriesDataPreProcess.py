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
from typing import Dict, Union, Tuple
from collections import defaultdict
from pandas.api.types import is_numeric_dtype
from tqdm import tqdm



#TODO: 후에 interpolation_ratio = interpolation_ratio.fillna(0) 수정 가능성 염두

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

    print("mBle Parsing start")
    for idx, row in tqdm(df.iterrows(), leave=False):
        entry = ast.literal_eval(row['m_ble']) if isinstance(row['m_ble'], str) else row['m_ble']

        rssi_list = []
        class_0_cnt = 0
        class_other_cnt = 0

        for device in entry:
            try:
                rssi = int(device['rssi'])
                rssi_list.append(rssi)

                if str(device['device_class']) == '0':
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

    print("mBle Parsing end")
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
    start_day = ts.min().replace(hour=0, minute=0, second=0, microsecond=0)
    end_day = ts.max().replace(hour=23, minute=59, second=0, microsecond=0)
    return start_day, end_day



def build_merge_key(start, end, freq) -> pd.DataFrame:
    timestamps = pd.date_range(start=start, end=end, freq=str(freq)+'min')

    return pd.DataFrame([
        {'subject_id': sid, 'timestamp': t}
        for sid in subject_id
        for t in timestamps
    ])

def find_drop_rows(
        merged_df:pd.DataFrame, 
        threshold:float=0.3, 
        continuous_time = 180, #min
        frequency = 1,
        execept = 20,) -> set[tuple[str, str]]:

    continuous_time_block  = int(continuous_time // frequency)
    max_gap_time_block = int(execept // frequency)
    drop_set = set()

    print(merged_df.isna().sum())
    merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'])
    merged_df['date'] = merged_df['timestamp'].dt.date

    value_cols = [col for col in merged_df.columns if col not in ['subject_id', 'timestamp', 'date']]

    for (subject_id, date), group in merged_df.groupby(['subject_id', 'date']):
        values = group[value_cols].values  # shape: (T, C)

        if values.shape[0] == 0:
            print(f"🚨 DROP (0 rows): {subject_id} - {date}")
            drop_set.add((str(subject_id), str(date)))
            continue

        for ch in range(values.shape[1]):
            col = values[:, ch]
            is_nan = pd.Series(col).isna().astype(int)

            ratio = is_nan.mean()
            max_gap = is_nan.groupby((is_nan != is_nan.shift()).cumsum()).sum().max()

            if (ratio >= threshold or max_gap >= continuous_time_block):
                #print(f"⚠️ DROP: {subject_id}, {date} | ratio={ratio:.2f} | max_gap={max_gap}")
                drop_set.add((subject_id, str(date)))
            else:
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

    print("mAmbience Parsing start")
    for idx, row in tqdm(df.iterrows(), leave=False):
        parsed = ast.literal_eval(row['m_ambience']) if isinstance(row['m_ambience'], str) else row['m_ambience']
        others_prob = 0.0

        for label, prob in parsed:
            prob = float(prob)
            if label in top_10_labels:
                df.at[idx, label] = prob
            else:
                others_prob += prob

        df.at[idx, 'others'] = others_prob

    print("mAmbience Parsing end")
    return df.drop(columns=['m_ambience'])

def process_mGps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date']      = df['timestamp'].dt.date
    features = []

    print("mGps Parsing start")
    for idx, row in tqdm(df.iterrows(), leave=False):
        gps_raw = row.get('m_gps', None)

        # 1) 스칼라인지 확인하고 NaN이면 건너뛰기
        if isinstance(gps_raw, float) and pd.isna(gps_raw):
            continue

        # 2) 문자열이면 literal_eval, 실패 시 건너뛰기
        if isinstance(gps_raw, str):
            try:
                gps_list = ast.literal_eval(gps_raw)
            except (ValueError, SyntaxError):
                continue
        else:
            gps_list = gps_raw

        # 3) 리스트/튜플/ndarray 타입이 아닐 경우 건너뛰기
        if not isinstance(gps_list, (list, tuple, np.ndarray)):
            continue

        # 4) 실제 데이터 파싱
        alt, lat, lon, sp = [], [], [], []
        for entry in gps_list:
            if not isinstance(entry, dict):
                continue
            try:
                alt.append(float(entry.get('altitude', np.nan)))
                lat.append(float(entry.get('latitude', np.nan)))
                lon.append(float(entry.get('longitude', np.nan)))
                sp.append(float(entry.get('speed', np.nan)))
            except:
                continue

        features.append({
            'subject_id': row['subject_id'],
            'timestamp' : row['timestamp'],
            'date'      : row['date'],
            'altitude'      : np.nanmean(alt) if alt else np.nan,
            'altitude_std'  : np.nanstd(alt) if alt else np.nan,
            'latitude'      : np.nanmean(lat) if lat else np.nan,
            'latitude_std'  : np.nanstd(lat) if lat else np.nan,
            'longitude'     : np.nanmean(lon) if lon else np.nan,
            'longitude_std' : np.nanstd(lon) if lon else np.nan,
            'speed'         : np.nanmean(sp)  if sp  else np.nan,
            'speed_std'     : np.nanstd(sp)  if sp  else np.nan,
        })

    print("mGps Parsing end")
    return pd.DataFrame(features)

def process_mUsageStats(df: pd.DataFrame):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date

    features = []
    
    print("mUsageStats Parsing start")
    for (subj, timstamp), group in tqdm(df.groupby(['subject_id', 'timestamp'])):
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
            'timestamp': timstamp,
            'date': timstamp.date(),#버그나면 ()추가가
            'others_time': others_time
        }
        # 각 앱별 컬럼 추가
        feature.update({f'{app}_time': app_time[app] for app in top_apps})

        features.append(feature)

    print("mUsageStats Parsing end")
    print("디버깅 mUsageStats 내용", features[0])
    return pd.DataFrame(features)

def process_mWifi(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    results = []
    print("mWifi Parsing start")
    for (subj, ts), group in tqdm(df.groupby(['subject_id', 'timestamp']), leave=False):
        rssi_all = []

        for raw in group.get('m_wifi', []):
            # 스칼라 NaN 검사
            if isinstance(raw, float) and pd.isna(raw):
                continue
            # 문자열 파싱
            if isinstance(raw, str):
                try:
                    parsed = ast.literal_eval(raw)
                except (ValueError, SyntaxError):
                    continue
            else:
                parsed = raw
            # 리스트/튜플 여부 확인
            if not isinstance(parsed, (list, tuple, np.ndarray)):
                continue

            # 파싱된 항목 순회
            for ap in parsed:
                if not isinstance(ap, dict):
                    continue
                r = ap.get('rssi', None)
                if r is None:
                    continue
                try:
                    rssi_all.append(int(r))
                except:
                    continue

        results.append({
            'subject_id': subj,
            'timestamp' : ts,
            'date'      : ts.date(),
            'wifi_rssi_mean'   : np.mean(rssi_all) if rssi_all else np.nan,
            'wifi_rssi_min'    : np.min(rssi_all) if rssi_all else np.nan,
            'wifi_rssi_max'    : np.max(rssi_all) if rssi_all else np.nan,
            'wifi_detected_cnt': len(rssi_all)
        })

    print("mWifi Parsing end")
    return pd.DataFrame(results)

def process_wHr(df: pd.DataFrame) -> pd.DataFrame:
    print("wHr Parsing start")

    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date']      = df['timestamp'].dt.date

    def extract_mean(x):
        # 스칼라 NaN 검사
        if isinstance(x, float) and pd.isna(x):
            return np.nan
        # 문자열 → literal_eval
        if isinstance(x, str):
            try:
                vals = ast.literal_eval(x)
            except (ValueError, SyntaxError):
                return np.nan
        else:
            vals = x
        # 리스트/튜플/ndarray 여부 확인
        if not isinstance(vals, (list, tuple, np.ndarray)):
            return np.nan
        # 값이 있으면 평균, 없으면 nan
        return np.mean(vals) if len(vals) > 0 else np.nan

    df['hr_mean'] = df.get('heart_rate', pd.Series()).apply(extract_mean)
    if 'heart_rate' in df.columns:
        df.drop(columns=['heart_rate'], inplace=True)

    print("wHr Parsing end")
    return df


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    exclude_cols = ['timestamp', 'date', 'subject_id']
    for col in df.columns:
        if col not in exclude_cols:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)  # 문자열도 float으로
            except:
                pass
    return df

def interpolates_with_mask(lifelog_data:Dict[str, pd.DataFrame] = {}, method='linear', save_csv = False, is_continuous:bool = False):
    
    print("All day interpolates start")
    mask_type = 'ratio_mask' if is_continuous else 'bit_mask'

    interpolated_results = {}
    os.makedirs(dir_name, exist_ok=True)
    for name, df in lifelog_data.items():
        print()
        print(f"⏳ {name} 전처리 중..")
        if 'timestamp' in df.columns:
            df = df.sort_values(by=['subject_id',"timestamp"])
        else:
            df = df.sort_values(by=["subject_id", "date"])

        num_df = df.select_dtypes(include='number')

        nan_mask = num_df.isna()
        df = df.infer_objects(copy=False)
        df_interp = num_df.interpolate(method=method, limit_direction='both')
        interpolation_ratio = nan_mask.mean(axis=1)

        # 원본 df에 보간된 수치형 덮어쓰기
        df.update(df_interp)        
        # 보간된 위치 마스크
        mask = ~(nan_mask & df_interp.notna()) & df_interp.notna()

        # mask: DataFrame of shape (T, F) where True는 보간된 셀
        if is_continuous:
            # 보간 위치만 실측 비율로, 나머지는 1.0
            mask_np = mask.to_numpy() if isinstance(mask, pd.DataFrame) else mask
            observation_ratio = (1.0 - nan_mask.mean(axis=0)).fillna(0)
            obs_ratio_np = observation_ratio.to_numpy().reshape(1, -1)
            final_mask = np.where(mask_np, 1.0, obs_ratio_np)

            mask = pd.DataFrame(
                final_mask,
                columns=num_df.columns,
                index=num_df.index
            )


        assert df.isna().sum().sum() == 0, "Wrong Interpolation"

        mask_with_id = mask.copy()
        mask_with_id['subject_id'] = df['subject_id'].values
        if 'timestamp' in df.columns:
            mask_with_id['timestamp'] = df['timestamp'].values
        else:
            mask_with_id['date'] = df['date'].values


        #Columns Sort
        print("컬럼 목록:", df.columns.tolist()) #디버깅
        ordered_cols = ['subject_id'] + (['timestamp'] if 'timestamp' in df.columns else ['date']) + list(num_df.columns)
        df = df[ordered_cols]

        ordered_cols = ['subject_id'] + (['timestamp'] if 'timestamp' in mask_with_id.columns else ['date']) + list(num_df.columns)
        mask_with_id = mask_with_id[ordered_cols]

        interpolated_results[name] = {
            'data': convert_numeric_columns(df),
            'mask': convert_numeric_columns(mask_with_id) 
        }
        print(f"✅ {name} 전처리 완료")
        if save_csv:
            lifelog_data[name].to_csv(f"{dir_name}{name}_all_day_{method}_{mask_type}_orginal.csv")
            interpolated_results[name]['data'].to_csv(f"{dir_name}{name}_all_day_{method}_{mask_type}_interpolates.csv")
            interpolated_results[name]['mask'].to_csv(f"{dir_name}{name}_all_day_{method}_{mask_type}_mask.csv")

            print(f"✅{name}_all_day_{method}_interpolates_{mask_type}.csv Shape:{interpolated_results[name]['data'].shape}")
        
        print(f"🟢 {name} 마스크 검사:")
        print(f"- mask shape: {mask.shape}")
        print(f"보간 전 NaN비율{lifelog_data[name].isna().mean().mean():.4f}")
        print(f"- mask_with_id shape: {mask_with_id.shape}")
        print(f"- 보간된 값의 평균 비율: {mask.mean().mean():.4f}")
        print(f"- NaN 포함 여부 (should be False): {mask.isna().any().any()}")


    with open(f"{dir_name}Interpolates_all_day_{method}_{mask_type}_{file_name}", 'wb') as f:
        pkl.dump(interpolated_results, f)
        print(f"✅ Interpolates_all_day_{method}_{mask_type}_{file_name}")

    return interpolated_results

def interpolates_with_mask_daily(lifelog_data:dict[str, pd.DataFrame], method='linear', save_csv = False, is_continuous:bool = False):
    print("Daily interpolates start")

    interpolated_results = {}
    os.makedirs(dir_name, exist_ok=True)
    mask_type = 'ratio_mask' if is_continuous else 'bit_mask'
    for name, df in lifelog_data.items():

        print(f"⏳ {name} 전처리 중..")
        print(f"📊 df shape: {df.shape}")


        df = df.copy()
        interpolated_dfs = []
        mask_dfs = []
        interpolation_ratio_list = []
        for (subj, date), group in df.groupby(['subject_id', 'date']):

            if 'timestamp' in group.columns:
                group = group.sort_values(by=["subject_id", "timestamp"])
            else:
                group = group.sort_values(by=["subject_id", "date"])


            group = group.infer_objects(copy=False)
            num_df = group.select_dtypes(include='number')
            nan_mask = num_df.isna()
            df_interp = num_df.interpolate(method=method, limit_direction='both')
            df_interp.index = group.index
            mask = nan_mask & df_interp.notna()

            # mask: DataFrame of shape (T, F) where True는 보간된 셀
            interpolated_per_col = mask.sum(axis=0)  # Series of length F

            total_per_col = mask.shape[0]
            interpolation_ratio = (interpolated_per_col / total_per_col).fillna(0)
            interpolation_ratio_list.append(interpolation_ratio.mean())

            if is_continuous:
                # 보간 위치만 실측 비율로, 나머지는 1.0
                mask_np = mask.to_numpy() if isinstance(mask, pd.DataFrame) else mask
                observation_ratio = (1.0 - nan_mask.mean(axis=0)).fillna(0)
                obs_ratio_np = observation_ratio.to_numpy().reshape(1, -1)
                final_mask = np.where(mask_np, 1.0, obs_ratio_np)

                mask = pd.DataFrame(
                    final_mask,
                    columns=num_df.columns,
                    index=num_df.index
                )


            group.update(df_interp) 

            timestamp_col = 'timestamp' if 'timestamp' in group.columns else 'date'

            mask_with_id = mask.copy()
            mask_with_id['subject_id'] = group['subject_id'].values
            mask_with_id[timestamp_col] = group[timestamp_col].values

            # Columns Sort
            ordered_cols = ['subject_id', timestamp_col] + list(num_df.columns)
            mask_with_id = mask_with_id[ordered_cols]
            group = group[ordered_cols]


            assert group.select_dtypes(include='number').isna().sum().sum() == 0, "Interpolation failed"
            #디버깅
            if len(df_interp) > 0:
                interpolated_dfs.append(group)
                mask_dfs.append(mask_with_id)
            else:
                mask_dfs.append(mask_with_id)
                interpolated_dfs.append(group)
        
        interpolated_df = pd.concat(interpolated_dfs, ignore_index=True)
        mask_df = pd.concat(mask_dfs, ignore_index=True)

        interpolated_results[name] = {
            'data': convert_numeric_columns(interpolated_df),
            'mask': convert_numeric_columns(mask_df)
        }

        if interpolated_results[name]['data'].shape[0] == 0:
            print("DataFrame is Null")
        print(f"✅ {name} 전처리 완료")
        if save_csv:
            interpolated_results[name]['data'].to_csv(f"{dir_name}{name}_daily_{method}_{mask_type}_interpolates.csv")
            interpolated_results[name]['mask'].to_csv(f"{dir_name}{name}_daily_{method}_{mask_type}_mask.csv")

            print(f"✅{name}_daily_{method}_interpolates_{mask_type}.csv File Saved Shape:{interpolated_results[name]['data'].shape}")


        interpolation_ratio_list = np.array(interpolation_ratio_list)
        ratio = interpolation_ratio_list.mean()
        print(f"🟢 {name} 전체 마스크 검사:")
        print(f"- 보간된 값의 평균 비율: {ratio:.4f}")
        print(f"- NaN 포함 여부 (should be False): {mask_df.isna().any().any()}")

    with open(f"{dir_name}Interpolates_daily_{method}_{mask_type}_{file_name}", 'wb') as f:
        pkl.dump(interpolated_results, f)
        print(f"✅Interpolates_daily_{method}_{mask_type}_{file_name} File Saved")

    return interpolated_results

def sort_interpolated_results(
        interpolated_results: Dict[str, Dict[str, pd.DataFrame]],
) -> Dict[str, Dict[str, pd.DataFrame]]:

    sorted_results: Dict[str, Dict[str, pd.DataFrame]] = {}
    for name, df in interpolated_results.items():
        data_df = df.get('data')
        mask_df = df.get('mask')
        
        sort_cols = ['subject_id'] + (['date'] if 'date' in data_df.columns else []) + (['timestamp'] if 'timestamp' in data_df.columns else [])
        sorted_data_df = data_df.sort_values(by=sort_cols, ignore_index=True)
        sorted_mask_df = mask_df.sort_values(by=sort_cols, ignore_index=True)

        sorted_results[name] = {
            'data': sorted_data_df,
            'mask': sorted_mask_df
        }

    return sorted_results
        

def reorganize_by_subject_date(
    interpolated_results: Dict[str, Dict[str, pd.DataFrame]],
    all_modalities: list[str] = None
) -> Dict[Tuple[str, str], Dict[str, Union[Tuple[pd.DataFrame, pd.DataFrame], None]]]:
    
    sorted_result = sort_interpolated_results(interpolated_results)
    if all_modalities is None:
        all_modalities = list(sorted_result.keys())

    reorganized = defaultdict(dict)

    for modality, result in sorted_result.items():

        drop_cols = ['subject_id'] + (['date'] if 'date' in result['data'].columns else []) + (['timestamp'] if 'timestamp' in result['data'] else [])
        data_df = result['data']
        mask_df = result['mask']

        data_df['date'] = data_df['date'] if 'date' in data_df.columns else data_df['timestamp'].dt.date
        mask_df['date'] = mask_df['date'] if 'date' in mask_df.columns else mask_df['timestamp'].dt.date

        for (subj, date), group in data_df.groupby(['subject_id', 'date']):
            mask_group = mask_df.loc[group.index].reset_index(drop=True).drop(columns=drop_cols)
            data_group = group.reset_index(drop=True).drop(columns=drop_cols)
            reorganized[(subj, str(date))][modality] = (data_group, mask_group)


    # 누락된 모달리티는 None으로 채우기
    for key, mod_dict in reorganized.items():
        for modality in all_modalities:
            mod_dict.setdefault(modality, None)

    return reorganized

def preprocessing_to_interpolated_results(is_daily = True, method='linear', save_csv = False, is_continuous:bool = False, threshhold = 0.3, continuous_time = 180, execpt = 20):

    global dir_name, file_name

    mask_type = 'ratio_mask' if is_continuous else 'bit_mask'   
    all_day_path = os.path.join(dir_name, f"Interpolates_all_day_{method}_{mask_type}_{file_name}")
    per_day_path = os.path.join(dir_name, f"Interpolates_daily_{method}_{mask_type}_{file_name}")

    if os.path.exists(all_day_path):
        with open(all_day_path, 'rb') as f:
            interpolated_all_day_results = pkl.load(f)
            print(f"✅ Interpolates_all_day File Loaded (method={method}, mask={mask_type})")
    if os.path.exists(per_day_path):
        with open(per_day_path, 'rb') as f:
            interpolated_daily_results = pkl.load(f)
            print(f"✅ Interpolates_daily File Loaded (method={method}, mask={mask_type})")

    if not (os.path.exists(all_day_path)) or (not os.path.exists(per_day_path)):
        data_dir = os.path.join("Data", "ch2025_data_items")

        # Parquet 파일 전체 경로 리스트
        parquet_files = glob.glob(os.path.join(data_dir, 'ch2025_*.parquet'))

        # Parquet 파일을 읽어 DataFrame으로 변환
        lifelog_data:dict[str, pd.DataFrame] = {}

        for file_path in parquet_files: 
            name = os.path.basename(file_path).replace('.parquet', '').replace('ch2025_', '')
            lifelog_data[name] = pd.read_parquet(file_path)
            print(f"✅ Loaded: {name}, shape = {lifelog_data[name].shape}")

        print("\nStart Data Parsing")
        lifelog_data['mBle'] = summarize_mBle_daily(lifelog_data['mBle'])
        lifelog_data['mAmbience'] = process_mAmbience_top10(lifelog_data['mAmbience'])
        lifelog_data['mGps'] = process_mGps(lifelog_data['mGps'])
        lifelog_data['mUsageStats'] = process_mUsageStats(lifelog_data['mUsageStats'])
        lifelog_data['mWifi'] = process_mWifi(lifelog_data['mWifi'])
        lifelog_data['wHr'] = process_wHr(lifelog_data['wHr'])
        print('End data parsing')

        print()
        for name, df in lifelog_data.items():
            if name == 'mBle':
                continue

            start_time, end_time = find_start_end_time(df)

            print()
            print(f"디버깅 머지한 후에 결측치 개수 {name}: ")
            merge_key = build_merge_key(start_time, end_time, freqency[name])
            merge_df = pd.merge(df, merge_key, on=['subject_id', 'timestamp'], how='outer')
            merge_df = safe_resample(merge_df, freq=freqency[name], agg_numeric='mean', agg_other='last')

            print(f"{name} columns: {merge_df.columns.tolist()}")
            if name == 'mAmbience': print(merge_df.shape, f"144의 배수인가? {merge_df.shape[0] % 144}") #디버깅용용
            merge_df = merge_df.sort_values(by=['subject_id', 'timestamp'])

            if name == 'wHr':
                merge_df = merge_df.sort_values(by=['subject_id', 'timestamp'])
                
                merge_df.to_csv("wHr_merge.csv")
                print("\n 데이터형\n")
                merge_df.info()
                print(f"디버깅 wHr 결측치 개수: {merge_df.isna().sum().sum()}")

                numeric_cols = merge_df.select_dtypes(include='number').columns
                merge_df = merge_df.infer_objects(copy=False)
                merge_df[numeric_cols] = merge_df[numeric_cols].interpolate(
                    method=method, limit_direction='both', limit=2
                )
                print(f"디버깅 보간 후 wHr 결측치 개수: {merge_df.isna().sum().sum()}")

            drop_set = find_drop_rows(merged_df=merge_df, threshold=threshhold, continuous_time=continuous_time, frequency=freqency[name], execept=execpt)
            print(f"\nDrop Set Num: {len(drop_set)}")

            filtered_df = filter_dropped_rows(merge_df, drop_set)
            lifelog_data[name] = filtered_df

            print("\n드롭 뒤 데이터 셋 모습")
            print(f"{name}: {filtered_df.shape}")
            #print(f"{name}`s Missing Data: {filtered_df.isna().sum().sum()}")

            #print(f"드롭 뒤 데이터 결측치 비율 {name}_df: {filtered_df.isna().sum().sum() / filtered_df.size}")
        for name, df in lifelog_data.items():
            print(f"process_{name} 뒤 데이터 결측치 비율 {name}_df: {df.isna().sum().sum() / df.size}")

            sort_col = ['subject_id', 'timestamp'] if 'timestamp' in df.columns else ['subject_id', 'date']
            lifelog_data[name] = df.sort_values(by=sort_col)

        interpolated_all_day_results = interpolates_with_mask(lifelog_data=lifelog_data, method=method,save_csv=save_csv, is_continuous=is_continuous)
        interpolated_daily_results = interpolates_with_mask_daily(lifelog_data=lifelog_data, method=method,save_csv=save_csv, is_continuous=is_continuous)
    
    return interpolated_daily_results if is_daily else interpolated_all_day_results

def is_numeric(df, col_name):
    return is_numeric_dtype(df[col_name])

def safe_resample(df: pd.DataFrame, freq=1, agg_numeric='mean', agg_other='last'):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df = df.sort_values(by=['subject_id', 'date', 'timestamp'])
    freq = f"{freq}min"

    results = []

    for (subj, date), group in df.groupby(['subject_id', 'date']):
        group = group.set_index('timestamp')

        numeric_cols = group.select_dtypes(include='number').columns
        agg_dict = {col: agg_numeric for col in numeric_cols}

        resampled = (
            group
            .resample(freq)
            .agg(agg_dict)
            .asfreq(freq)  # 🔥 NaN row 유지
        )

        resampled['subject_id'] = subj
        resampled['date'] = date
        results.append(resampled.reset_index())
    return pd.concat(results, ignore_index=True)


def get_data_by_key( sample_dict: Dict[Tuple[str, str], Dict[str, pd.DataFrame]],
    key: Tuple[str, str]) -> Dict[str, Union[Dict[str, pd.DataFrame], None]]:
    result = sample_dict.copy()
    str_key = []
    str_key = tuple(k if isinstance(k, str) else str(k) for k in key)
 
    return result.get(str_key, {})

def bulid_data_dict(df:dict, key:tuple[str, datetime, datetime]):
    data_dict = {}
    for k in tqdm(key, leave=False):
        sleep_key = (k[0], k[1])
        lifelog_key = (k[0], k[2])
        data_dict[k] = (
            get_data_by_key(df, sleep_key),
            get_data_by_key(df, lifelog_key)
        )

    return data_dict
    
def data_load_and_split_test_and_train( #TODO threshhold=0.3, continuous_time=180 인자로 받게
        is_daily = True, 
        method='linear', 
        save_csv = False,
        threshhold=0.3, 
        continuous_time=180, 
        is_continuous = False) -> Tuple[dict, dict, dict, dict]:
 
    train_data_dir = os.path.join("Data", "ch2025_metrics_train.csv")
    test_data_dir = os.path.join("Data", "ch2025_submission_sample.csv")

    train_data = pd.read_csv(train_data_dir)
    test_data = pd.read_csv(test_data_dir)


    train_data['sleep_date'] = pd.to_datetime(train_data['sleep_date']).dt.date
    train_data['lifelog_date'] = pd.to_datetime(train_data['lifelog_date']).dt.date
    test_data['sleep_date'] = pd.to_datetime(test_data['sleep_date']).dt.date
    test_data['lifelog_date'] = pd.to_datetime(test_data['lifelog_date']).dt.date

    train_keys = set(zip(train_data['subject_id'], train_data['sleep_date'], train_data['lifelog_date']))
    test_keys = set(zip(test_data['subject_id'], test_data['sleep_date'],  test_data['lifelog_date']))
    
    interpolated_results = preprocessing_to_interpolated_results(
                                    save_csv=save_csv, 
                                    is_daily=is_daily, 
                                    method=method, 
                                    threshhold=threshhold, 
                                    continuous_time=continuous_time, 
                                    is_continuous=is_continuous)
    
    reorganized_result = reorganize_by_subject_date(interpolated_results)
    print("📦 [train_dict] 생성 중 (tqdm)...")
    '''
    train_dict = {k: (get_data_by_key(reorganized_result, (k[0], k[1])), 
                      get_data_by_key(reorganized_result, (k[0], k[2])))
                  for k in train_keys}'''
    
    train_dict = bulid_data_dict(reorganized_result, train_keys)
    print("📦 [train_label] 생성 완료 (tqdm)...")

    train_label = {
        (row.subject_id, pd.to_datetime(row.sleep_date).date(), pd.to_datetime(row.lifelog_date).date()):{
            "Q1": row.Q1,
            "Q2": row.Q2,
            "Q3": row.Q3,
            "S1": row.S1,
            "S2": row.S2,
            "S3": row.S3
        }
        for _, row in train_data.iterrows()
    }
    print("📦 [test_dict] 생성 중 (tqdm)...")
    test_dict = bulid_data_dict(reorganized_result, test_keys)
    print("📦 [test_dict] 생성 완료 (tqdm)...")
    
    test_label = {
        (row.subject_id, pd.to_datetime(row.sleep_date).date(), pd.to_datetime(row.lifelog_date).date()):
            [row.Q1,
            row.Q2,
            row.Q3,
            row.S1,
            row.S2,
            row.S3]
        for _, row in test_data.iterrows()}
    
    train_dict = dict(sorted(train_dict.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])))
    train_label = dict(sorted(train_label.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])))

    test_data = dict(sorted(test_dict.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])))
    test_label = dict(sorted(test_label.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])))
    return train_dict, train_label, test_dict, test_label
    

from collections import Counter
import matplotlib.pyplot as plt

def count_unique_modalities(train_dict):

    modality_counts = []

    for (sleep_data, lifelog_data) in train_dict.values():
        active_modalities = set()

        if sleep_data:
            #active_modalities.update([k for k, v in sleep_data.items() if v is not None])
            pass
        if lifelog_data:
            active_modalities.update([k for k, v in lifelog_data.items() if v is not None])

        modality_counts.append(len(active_modalities))

    return modality_counts

def count_none_per_modality(train_dict):
    sleep_none_counts = Counter()
    lifelog_none_counts = Counter()

    for (sleep_data, lifelog_data) in train_dict.values():
        if sleep_data:
            for modality, value in sleep_data.items():
                if value is None:
                    sleep_none_counts[modality] += 1
        if lifelog_data:
            for modality, value in lifelog_data.items():
                if value is None:
                    lifelog_none_counts[modality] += 1

    return sleep_none_counts, lifelog_none_counts


def filter_data_and_labels_by_modalities(
    data_dict: dict,
    label_dict: dict,
    min_modalities_sleep: int = 6,
    min_modalities_lifelog: int = 6,
    verbose: bool = True
):
    filtered_data = {}
    filtered_labels = {}
    removed = []

    for key, (sleep_data_dict, lifelog_data_dict) in data_dict.items():
        subject_id = key[0]

        # 🛡️ None-safe: 각 dict가 None일 수 있음
        sleep_modalities = set()
        if isinstance(sleep_data_dict, dict):
            sleep_modalities = {
                m for m, val in sleep_data_dict.items()
                if val is not None and isinstance(val, tuple) and val[0] is not None and not val[0].empty
            }

        lifelog_modalities = set()
        if isinstance(lifelog_data_dict, dict):
            lifelog_modalities = {
                m for m, val in lifelog_data_dict.items()
                if val is not None and isinstance(val, tuple) and val[0] is not None and not val[0].empty
            }

        if len(sleep_modalities) >= min_modalities_sleep and len(lifelog_modalities) >= min_modalities_lifelog:
            filtered_data[key] = (sleep_data_dict, lifelog_data_dict)
            if key in label_dict:
                filtered_labels[key] = label_dict[key]
        else:
            removed.append((key, len(sleep_modalities), len(lifelog_modalities)))

    if verbose:
        print(f"🧼 필터링 전 샘플 수: {len(data_dict)}")
        print(f"✅ 필터링 후 샘플 수: {len(filtered_data)}")
        print(f"🎯 남은 라벨 수: {len(filtered_labels)}")
        print(f"🚫 제거된 샘플 수: {len(removed)}")
        if removed:
            print("  예시 →", removed[:5])

    return filtered_data, filtered_labels


if __name__ == "__main__":

    dir = 'Data_Dict'
    method = 'linear'
    is_daliy = True
    is_continuous = True
    mask_type = 'ratio_mask' if is_continuous else 'bit_mask'
    daliy_or_all_day = "daily" if is_daliy else "all_day"
    save_csv = True

    train_x_file_name =f"train_data_{daliy_or_all_day}_{method}_{mask_type}.pkl"
    train_y_file_name = f"train_label_{daliy_or_all_day}_{method}_{mask_type}.pkl"
    test_x_file_name = f"test_data_{daliy_or_all_day}_{method}_{mask_type}.pkl"
    test_y_file_name = f"test_label_{daliy_or_all_day}_{method}_{mask_type}.pkl"

    os.makedirs(dir, exist_ok=True)
    
    train_x_file_path = os.path.join(dir, train_x_file_name)
    train_y_file_path = os.path.join(dir, train_y_file_name)
    test_x_file_path = os.path.join(dir, test_x_file_name)
    test_y_file_path = os.path.join(dir, test_y_file_name)
    
    print(f'Find {daliy_or_all_day}_{method}_{mask_type} Data')
    if (not os.path.exists(train_x_file_path) or 
        not os.path.exists(train_y_file_path) or
        not os.path.exists(test_x_file_path) or
        not os.path.exists(test_y_file_path)
        ):
        print(f'{daliy_or_all_day}_{method}_{mask_type} Data Not Found')
        print(f'Create {daliy_or_all_day}_{method}_{mask_type} Data')

        train_data, train_label, test_data, test_label = data_load_and_split_test_and_train(save_csv=save_csv, is_continuous=is_continuous)
        with open(train_x_file_path, 'wb') as f:
            pkl.dump(train_data, f)
        with open(train_y_file_path, 'wb') as f:
            pkl.dump(train_label, f)
        with open(test_x_file_path, 'wb') as f:
            pkl.dump(test_data, f)
        with open(test_y_file_path, 'wb') as f:
            pkl.dump(test_label, f)
    else:
        print(f'{daliy_or_all_day}_{method}_{mask_type} Data Found')
        print(f'Load {daliy_or_all_day}_{method}_{mask_type} Data')
        with open(train_x_file_path, 'rb') as f:
            train_data = pkl.load(f)
        with open(train_y_file_path, 'rb') as f:
            train_label = pkl.load(f)
        with open(test_x_file_path, 'rb') as f:
            test_data = pkl.load(f)
        with open(test_y_file_path, 'rb') as f:
            test_label = pkl.load(f)


    print(f"[train_dict] size: {len(train_data)}")
    print(f"[train_label] size: {len(train_label)}")
    print(f"[test_dict] size: {len(test_data)}")
    print(f"[test_label] size: {len(test_label)}")

    # 예시: 첫 train 샘플 확인
    for k in list(train_data.keys())[:1]:
        print(f"\n🧪 Train Sample Key: {k}")
        print("📦 Sleep Day Modalities:", list(train_data[k][0].keys()))
        print("📦 Lifelog Day Modalities:", list(train_data[k][1].keys()))
        print("📝 Labels:", train_label.get(k, {}))

    # 🔁 적용
    sleep_none_counts, lifelog_none_counts = count_none_per_modality(train_data)

    # 📊 데이터프레임으로 보기 좋게 정리
    none_df = pd.DataFrame({
        "sleep_none_count": pd.Series(sleep_none_counts),
        "lifelog_none_count": pd.Series(lifelog_none_counts)
    }).fillna(0).astype(int)

    # ✅ 보기
    print(none_df.sort_index())

    modality_counts = count_unique_modalities(train_data)
    count_distribution = Counter(modality_counts)

    print(count_distribution)
    print(len(next(iter(train_data.values()))[0]))

    plt.figure(figsize=(8, 5))
    plt.bar(count_distribution.keys(), count_distribution.values())
    plt.xlabel("Number of Non-None Modalities in Sample")
    plt.ylabel("Number of Samples")
    plt.title("Effective Modality Count per Train Sample")
    plt.grid(True)
    plt.show()
    

    train_data, train_label = filter_data_and_labels_by_modalities(train_data, train_label)

    train_x_file_name = f"train_data_filtered_{daliy_or_all_day}_{method}_{mask_type}.pkl"
    train_x_file_path = os.path.join(dir, train_x_file_name)
    with open(train_x_file_path, 'wb') as f:
            pkl.dump(train_data, f)

    modality_counts = count_unique_modalities(train_data)
    count_distribution = Counter(modality_counts)

    print(count_distribution)
    print(len(next(iter(train_data.values()))[0]))

    plt.figure(figsize=(8, 5))
    plt.bar(count_distribution.keys(), count_distribution.values())
    plt.xlabel("Number of Non-None Modalities in Sample")
    plt.ylabel("Number of Samples")
    plt.title("Effective Modality Count per Train Sample")
    plt.grid(True)
    plt.show()

    
    sample = next(iter(train_data.values()))[0]['mLight'][1]

    print(sample)

    













            


