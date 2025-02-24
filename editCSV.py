import os
import pandas as pd 
from util import *
import shutil



def add_new_individual_df(df):
    df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
    df['usd_vol'] = df['volume'] * (df['open'] + df['close'])/2
    if (df['close'] == 0).any():
        print("Error: There are zeros in closing prices")

    df['price_change'] = (df['close'] - df['open'])/(df['open'])
    df['vol_change'] = ((df['usd_vol'] - df['usd_vol'].shift(1)) / df['usd_vol'].shift(1)).fillna(0)


    df['norm_open'] = df['open']/(df['open'].iloc[0])
    df['mov_in_min'] = (df['high'] - df['low'])/df['open']
    return df


def add_new_features(old_prefix, old_suffix, new_prefix, new_suffix):
    folders_dict = get_folder_names(prefix = old_prefix, suffix = old_suffix)
    periods_of_time = ['nov1', 'dec1', 'dec2', 'jan1', 'jan2', 'feb1']
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
    while time_stamps[last_index+1] <= end_of_first_x_min:
        last_index += 1
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
    data_to_pad = ['data/jan18_ohlcv']
    pad_csv_folders(data_to_pad, False, cut_out_data = [True, 30])



add_new_features(old_prefix = "data/", old_suffix = "_ohlcv_padded_low_volume_dropped",
                  new_prefix = "data/" , new_suffix = "_padded_extra_features")
