import os
import pandas as pd 
from util import *
import shutil
import time
import json
import numpy as np
import csv
import bisect


def load_candles(csv_filename):
    candles = []
    with open(csv_filename, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            candle = {
                "timestamp": int(row["timestamp"]),
                "open": float(row["open"]),
                "volume": float(row["volume"])
            }
            candles.append(candle)
    candles.sort(key=lambda x: x["timestamp"])
    return candles


def find_closest_candle(timestamp, candles):
    # Create a list of start times for binary search.
    starts = [candle["timestamp"] for candle in candles]
    pos = bisect.bisect_left(starts, timestamp)
    
    if pos == 0:
        return candles[0]
    elif pos == len(candles):
        return candles[-1]
    else:
        before = candles[pos - 1]
        after = candles[pos]
        if abs(before["timestamp"] - timestamp) <= abs(after["timestamp"] - timestamp):
            return before
        else:
            return after

def sum_volume_interval(start_interval, end_interval, candles):
    total = sum(candle["volume"] for candle in candles if start_interval < candle["timestamp"] <= end_interval + 10)
    return total

def get_market_data(migration_ts, candles, sol = True):
    # Find the candle closest to the migration timestamp.
    migration_candle = find_closest_candle(migration_ts, candles)
    
    # Calculate 24h metrics:
    ts_24h = migration_ts - 24 * 3600
    candle_24h = find_closest_candle(ts_24h, candles)
    # Sum volume for all candles in the 24-hour period preceding the migration timestamp.
    vol_24h = sum_volume_interval(ts_24h, migration_ts, candles)
    
    
    price_migration  = migration_candle["open"]
    price_change_24h = (migration_candle["open"] - candle_24h["open"]) / candle_24h["open"]
    
    # Calculate 7-day metrics:
    ts_7d = migration_ts - 7 * 24 * 3600

    candle_7d = find_closest_candle(ts_7d, candles)
    
    price_change_7d  = (migration_candle["open"] - candle_7d["open"]) / candle_7d["open"]
    
    if sol:
        return {
            "price_at_migration": price_migration / 100, # diving by 100 to get a sol price between 0 and 1
            "vol_24hr": vol_24h,
            "price_change_24h": price_change_24h,
            "price_change_7d": price_change_7d,
        }
    else:
        return {
            "price_at_migration": price_migration / 100000, # diving by 100,000 to get a btc price between 0 and 1
            "vol_24hr": vol_24h,
            "price_change_24h": price_change_24h,
            "price_change_7d": price_change_7d,
        }



#problem is in this function
def creator_to_coins_mapping():
    df = pd.read_csv('mastersheet_with_filenames_v2.csv')
    df = df.dropna(subset=['index'])
    print(len(df))
    df = df.dropna(subset=['file_name'])
    print(len(df))

    init_dict = df.groupby("token_creator")["file_name"].apply(list).to_dict()

    prefix = "init_data/"
    middle = "_json_padded/"

    creator_files_dict ={}
    for key in init_dict:
        files = init_dict[key]
        new_files = []
        
        for f in files:
            try:
                month = f[:3] 
                new_files.append(prefix + month + middle + f)
            except:
                print('failed')
        creator_files_dict[key] = new_files


    counts = {}
    for key in creator_files_dict:
        count = len(creator_files_dict[key])
        if count not in counts.keys():
            counts[count] = 0
        counts[count] += 1
        if count > 5:
            #print(key)
            pass
    print(creator_files_dict['Cbo6pdGQowiMuYjVtqXoZ5wSC9Z9G7vePffcvwDihkQw'])
    return creator_files_dict


def get_change(json_df, start_time, cap = None):
    movements = json_df['ohlcv_data']
    first_min_after_prediction = movements[start_time]['open']
    total_min = 10 * 60
    final_min = movements[total_min - 1]['close']
    if len(movements) != total_min:
        raise Exception("Dataframe doesn't have the right length")

    change = (final_min - first_min_after_prediction)/first_min_after_prediction
    if cap is not None:
        change = min(change, cap)
    return change


def get_all_json_files():
    data_by_month = get_folder_names(prefix = "init_data/", suffix = "_json_padded")
    all_csv = []
    month_names = ['sep', 'oct', 'nov', 'dec', 'jan', 'feb']
    for month in month_names:
        print(data_by_month[month])
        all_csv += gather_all_csv([data_by_month[month]])
    
    return all_csv




def median_interval_and_earliest(timestamps):
    """
    Computes the median inter-launch interval and the earliest timestamp from a list of launch timestamps.
    
    Parameters:
        timestamps (list or array-like): A list of numeric launch timestamps (e.g., in seconds).
    
    Returns:
        tuple: (median_interval, earliest_timestamp)
               - median_interval (float): The median of the differences between successive timestamps, or np.nan if fewer than 2 timestamps.
               - earliest_timestamp (float): The smallest timestamp in the list, or None if the list is empty.
    """
    timestamps = np.array(timestamps, dtype=float)
    if timestamps.size == 0:
        return (np.nan, None)
    
    sorted_timestamps = np.sort(timestamps)
    earliest = sorted_timestamps[0]
    
    if sorted_timestamps.size < 2:
        median_interval = np.nan
    else:
        intervals = np.diff(sorted_timestamps)
        median_interval = np.median(intervals)
        median_interval_days = median_interval / (3600 * 24) 
    
    return (median_interval_days, earliest)

def add_total_returns(json_file):
    fifteen_change = get_change(json_file, 15, cap = None)
    fifteen_change_capped = get_change(json_file, 15, cap = 5)
    thirty_change = get_change(json_file, 30, cap = None)
    thirty_change_capped = get_change(json_file, 30, cap = 5)
    json_file['processed_data']['fifteen_change'] = fifteen_change
    json_file['processed_data']['fifteen_changed_capped'] = fifteen_change_capped
    json_file['processed_data']['thirty_change'] = thirty_change
    json_file['processed_data']['thirty_change_capped'] = thirty_change_capped
    return json_file

def above_min_bar(json_file, min_in, label, bar = 1e-05):
    n_min = json_file['ohlcv_data'][min_in-1]['close']
    if n_min > bar:
        json_file['processed_data'][label] = '1'
    else:
        json_file['processed_data'][label] = '0'
    return json_file



def add_other_coins_data(json_file, migration_time, sol_candles, btc_candles):
    sol_info = get_market_data(migration_time, sol_candles, sol = True)
    btc_info = get_market_data(migration_time, btc_candles, sol = False)

    # Update the JSON data with the new market info.
    json_file["processed_data"]["sol_market_data"] = sol_info
    json_file["processed_data"]["btc_market_data"] = btc_info
    return json_file

def add_data_to_all_jsons():
    TEN_HOURS = 60 * 60 * 10 #Youssef confirm
    creator_to_coins = creator_to_coins_mapping()
    csv_files = get_all_json_files()
    sol_candles = load_candles("major_coins_data/SOL-USD.csv")
    btc_candles = load_candles("major_coins_data/BTC-USD.csv")



    for file_n in csv_files:
        df, json_file = read_our_json_file(file_n)
        if np.isnan(json_file['token_data']['created_timestamp']):
            print(f"Encountered NAN on {file_n}")
            continue

        creator = json_file['token_data']['token_creator']
        try:
            other_coins = creator_to_coins[creator]
            if file_n == "init_data/nov_json_padded/nov06_52.json":
                print(other_coins, '52')
            if file_n == "init_data/nov_json_padded/nov04_38.json":
                print(other_coins, '38')
            if file_n == "init_data/nov_json_padded/nov07_206.json":
                print(other_coins, '206')
            
        except:
            print("Error finding creator. Continuing")
            print(file_n)
            continue
        try:
            other_coins.remove(file_n)
        except:
            print("Error in list of other coins created by this person")
            continue
        og_coin_migration_time = json_file['token_data']['migration_timestamp']
        og_coin_creation_time = json_file['token_data']['created_timestamp']
        total_val_other_coins = 0
        total_other_coins_return_15 = 0
        total_other_coins_capped_return_15 = 0
        total_other_coins_return_30 = 0
        total_other_coins_capped_return_30 = 0
        
        other_coins_creation_ts = []

        json_file = add_other_coins_data(json_file, og_coin_migration_time, sol_candles, btc_candles)
        
        json_file = add_total_returns(json_file)

        for coin in other_coins:
            df, other_coin_json_file = read_our_json_file(coin)
            migration_time = other_coin_json_file['token_data']['migration_timestamp']
        
            creation_ts = other_coin_json_file['token_data']['created_timestamp']
            other_coins_creation_ts.append(creation_ts)
            if og_coin_migration_time == 1730926894:
                print(file_n)
                print(migration_time)
                print(og_coin_migration_time)
            if og_coin_migration_time - migration_time > TEN_HOURS:
                if og_coin_migration_time == 1730926894:
                    print('here')
                total_val_other_coins += 1
                total_other_coins_return_15 += get_change(other_coin_json_file, 15)
                total_other_coins_capped_return_15 += get_change(other_coin_json_file, 15, cap = 5)
                total_other_coins_return_30 += get_change(other_coin_json_file, 30)
                total_other_coins_capped_return_30 += get_change(other_coin_json_file, 30, cap = 5)

        if total_val_other_coins == 0:
            other_coins_15_return = 0
            other_coins_15_return_capped = 0
            other_coins_30_return = 0
            other_coins_30_return_capped = 0
        else:
            other_coins_15_return = total_other_coins_return_15/total_val_other_coins
            other_coins_15_return_capped = total_other_coins_capped_return_15/total_val_other_coins
            other_coins_30_return = total_other_coins_return_30/total_val_other_coins
            other_coins_30_return_capped = total_other_coins_capped_return_30/total_val_other_coins

        #days_since_first_launch, median_interval = median_interval_and_earliest(other_coins_creation_ts) 
        json_file['processed_data']['creator_data'] = {}
        json_file['processed_data']['creator_data']['other_coins_15_return'] = other_coins_15_return
        json_file['processed_data']['creator_data']['other_coins_15_return_capped'] = other_coins_15_return_capped
        json_file['processed_data']['creator_data']['other_coins_30_return'] = other_coins_30_return
        json_file['processed_data']['creator_data']['other_coins_30_return_capped'] = other_coins_30_return_capped
        json_file['processed_data']['creator_data']['other_coins_created'] = total_val_other_coins
        #json_file['processed_data']['creator_data']['days_since_first_launch'] = days_since_first_launch
        #json_file['processed_data']['creator_data']['mean_interval'] = median_interval

        final_file_path = file_n[5:]

        json_file = above_min_bar(json_file, 15, 'above_bar_15_min_in')
        json_file = above_min_bar(json_file, 30, 'above_bar_30_min_in')


        directory = os.path.dirname(final_file_path)
        os.makedirs(directory, exist_ok=True)
        with open(final_file_path, 'w') as f:
            json.dump(json_file, f, indent = 4)




def add_new_individual_df(df):
    df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
    df['usd_vol'] = df['volume'] * (df['open'] + df['close'])/2
    if (df['close'] == 0).any():
        print("Error: There are zeros in closing prices")

    df['price_change'] = (df['close'] - df['open'])/(df['open'])
    df['vol_change'] = (
    (df['usd_vol'] - df['usd_vol'].shift(1)) / df['usd_vol'].shift(1)
).replace([np.inf, -np.inf], 0).fillna(0)


    df['norm_open'] = df['open']/(df['open'].iloc[0])
    df['mov_in_min'] = (df['high'] - df['low'])/df['open']
    return df


def add_new_features(old_prefix, old_suffix, new_prefix, new_suffix):
    folders_dict = get_folder_names(prefix = old_prefix, suffix = old_suffix)
    periods_of_time = ['sep1', 'sep2', 'oct1', 'oct2', 'nov1', 'nov2', 'dec1', 'dec2', 'jan1', 'jan2', 'feb1', 'feb2']
    folder_list = []
    for month in periods_of_time:
        folder_list += folders_dict[month]

    for folder in folder_list:
        print(f'Currently running on {folder}')
        if not os.path.isdir(folder):
            print(f"Problem: Skipping {folder} as it is not a directory")
            continue
        
        old_name = folder.split("_")[0]
        new_folder = old_name + new_suffix
        os.makedirs(new_folder, exist_ok=True)
        counter = 0

        for fname in os.listdir(folder):
            if fname.lower().endswith(".csv"):
                old_path = os.path.join(folder, fname)
                new_path = os.path.join(new_folder, fname)
                
                try:
                    df = pd.read_csv(old_path)
                except Exception as e:
                    print(f'Problem: failed to read in file {old_path} with error {e}')
                    continue

                if df.empty:
                    print(f'Problem: Skipping {fname} because its empty')
                    continue

                df = add_new_individual_df(df)

                df.to_csv(new_path, index=False)

                counter += 1
                if counter % 10 == 0:
                    print(f'Saved {counter} files')




def delete_old_folders(prefix_name, suffix_name):
    folders = get_folder_names(prefix = prefix_name, suffix = suffix_name)
    periods = ['nov1', 'dec1', 'dec2', 'jan1', 'jan2', 'feb1']
    for period in periods:
        for folder in folders[period]:
            print(folder)
            try:
                shutil.rmtree(folder)
            except Exception as e:
                print(e)
                print(folder)



def reverse_df(df):
    if df.empty:
        print('Problem: Encountered empty dataframe when reversing')
    return df.iloc[::-1].reset_index(drop=True)


def last_row_before_time(time_stamps, minutes):
    first_time_stamp = time_stamps[0]
    end_of_first_x_min = first_time_stamp + (minutes * 60) - 60 #final timestamp in first_x_min
    last_time_stamp = time_stamps[-1]
    if last_time_stamp < end_of_first_x_min:
        return None

    last_index = -1
    try:
        while time_stamps[last_index+1] <= end_of_first_x_min:
            last_index += 1
    except:
        print("ERROR")
        return None
    if last_index == -1:
        raise Exception("Problematic Dataframe, No rows exist before end of training")
    
    return last_index



#pad the dataframe so it has 600 rows
def pad_timestamps(df, new_path, cut_out_data, desired_rows = 600, step = 60):
    if df.empty:
        print('Problem: Empty Dataframe When Padding')
    new_rows = []
    first_row = df.iloc[0].copy()
    new_rows.append(first_row)
    last_timestamp = first_row.iloc[0]
    real_idx = 1
    n_original = len(df)

    if cut_out_data[0]:
        train_minutes = cut_out_data[1]
        time_stamps = list(df.iloc[:, 0])
        idx = last_row_before_time(time_stamps, train_minutes)
        if idx == None:
            return None


        last_training_row = df.iloc[idx]
        last_open = last_training_row.iloc[1]
        if last_open < 1e-5:
            return None



    while len(new_rows) < desired_rows:
        if real_idx < n_original:
            next_row = df.iloc[real_idx]
            next_timestamp = next_row.iloc[0]
            gap = next_timestamp - last_timestamp
            if gap == step:
                new_rows.append(next_row.copy())
                last_timestamp = next_timestamp
                real_idx += 1
            elif gap < step:
                print('MAJOR PROBLEM')
                print('Problem: When padding difference between rows is less than 60')
                print(f'Path: {new_path} on index {len(new_rows)}')
                real_idx += 1
            else:
                filler = new_rows[-1].copy()
                filler.iloc[0] = last_timestamp + step #time
                last_close_value = new_rows[-1].iloc[4]
                filler.iloc[1] = last_close_value #open
                filler.iloc[2] = last_close_value #high
                filler.iloc[3] = last_close_value #low
                filler.iloc[4] = last_close_value #close

                filler.iloc[5] = 0 #volatility

                new_rows.append(filler)
                last_timestamp = filler.iloc[0]
        else:
            filler = new_rows[-1].copy()
            filler.iloc[0] = last_timestamp + step #time
            last_close_value = new_rows[-1].iloc[4]
            filler.iloc[1] = last_close_value #open
            filler.iloc[2] = last_close_value #high
            filler.iloc[3] = last_close_value #low
            filler.iloc[4] = last_close_value #close

            filler.iloc[5] = 0 #volatility

            new_rows.append(filler)
            last_timestamp = filler.iloc[0]
    if (len(new_rows) != desired_rows):
        print('Problem: We failed to properly pad dataframe')
    
    padded_df = pd.DataFrame(new_rows, columns = df.columns)
    return padded_df

def pad_csv_folders(folder_list, first_30 = False, desired_rows = 600, step = 60, cut_out_data = [False, 0]):
    for folder in folder_list:
        print(f'Currently running on {folder}')
        if not os.path.isdir(folder):
            print(f"Problem: Skipping {folder} as it is not a directory")
            continue
        
        suffix = "_padded"
        if first_30:
            suffix += "_first30"
        if cut_out_data[0]:
            suffix += "_low_volume_dropped"

        new_folder = folder + suffix
        os.makedirs(new_folder, exist_ok=True)
        counter = 0
        for fname in os.listdir(folder):
            if fname.lower().endswith(".csv"):
                old_path = os.path.join(folder, fname)
                new_path = os.path.join(new_folder, fname)
                
                try:
                    df = pd.read_csv(old_path)
                except Exception as e:
                    print(f'Problem: failed to read in file {old_path} with error {e}')
                    continue

                if df.empty:
                    print(f'Problem: Skipping {fname} because its empty')
                    continue

                df = reverse_df(df)

                df = pad_timestamps(df, new_path, cut_out_data, desired_rows=desired_rows, step=step)
                if not isinstance(df, pd.DataFrame):
                    continue

                if first_30:
                    df = df.head(30)
                df.to_csv(new_path, index=False)
                counter += 1
                if counter % 10 == 0:
                    print(f'Saved {counter} files')

#not currently in use
def cutOffDf(folder, numRowsToInclude):
    new_folder = folder + "_first30"
    os.makedirs(new_folder, exist_ok=True)

    for fname in os.listdir(folder):
        if fname.lower().endswith(".csv"):
            old_path = os.path.join(folder, fname)
            new_path = os.path.join(new_folder, fname)

            try:
                df = pd.read_csv(old_path)
            
            except Exception as e:
                print(f'Problem: Error reading {old_path} of {e}')
                continue

            if df.empty:
                print(f'Problem: Skipping {fname} because its empty')
                continue

            df = df.head(numRowsToInclude)
            df.to_csv(new_path, index = False)
        




def paddingMain():
    all_months = ['sep1', 'sep2', 'oct1', 'oct2', 'nov1', 'nov2', 'dec1', 'dec2', 'jan1', 'jan2', 'feb1', 'feb2']
    data_to_pad = []
    all_folder_names = get_folder_names(suffix = "_ohlcv")
    for month in all_months:
        data_to_pad += all_folder_names[month]
    pad_csv_folders(data_to_pad, False, cut_out_data = [True, 60])

add_data_to_all_jsons()