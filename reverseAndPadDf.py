import os
import pandas as pd 

def reverse_df(df):
    if df.empty:
        print('Problem: Encountered empty dataframe when reversing')
    return df.iloc[::-1].reset_index(drop=True)


#pad the dataframe so it has 600 rows
def pad_timestamps(df, new_path, desired_rows = 600, step = 60):
    if df.empty:
        print('Problem: Empty Dataframe When Padding')
    new_rows = []
    first_row = df.iloc[0].copy()
    new_rows.append(first_row)
    last_timestamp = first_row.iloc[0]
    real_idx = 1
    n_original = len(df)
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

def pad_csv_folders(folder_list, desired_rows = 600, step = 60):
    for folder in folder_list:
        print(f'Currently running on {folder}')
        if not os.path.isdir(folder):
            print(f"Problem: Skipping {folder} as it is not a directory")
            continue
        
        new_folder = folder + "_padded"
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

                df = pad_timestamps(df, new_path, desired_rows=desired_rows, step=step)

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
    folder = ["data/" +  "jan" + str(i) + "_ohlcv" for i in range(16, 27)]
    pad_csv_folders(folder)

paddingMain()
