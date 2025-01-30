import requests
import json
from dune_client.client import DuneClient
import time
import csv
from datetime import datetime
import pandas as pd

chain = "solana"

################# Getting data on graduating coins for a specific day ############################

listOfDays = ["2024-09-01", "2024-09-02", "2024-09-03", "2024-09-04", "2024-09-05",
              "2024-09-06", "2024-09-07", "2024-09-08", "2024-09-09", "2024-09-10",
              "2024-09-11", "2024-09-12", "2024-09-13", "2024-09-14", "2024-09-15",
              "2024-09-16", "2024-09-17", "2024-09-18", "2024-09-19", "2024-09-20",
              "2024-09-21", "2024-09-22", "2024-09-23", "2024-09-24", "2024-09-25",
              "2024-09-26", "2024-09-27", "2024-09-28", "2024-09-29", "2024-09-30"]

# date format: "yyyy-mm-dd"
def get_graduating_coins_DUNE(date):
    # dune API key
    # dune = DuneClient("O2ymfk7UST25dnDwOj0f00kRzcB8pc59")
    # df = dune.get_latest_result_dataframe(4000309)
    # df.to_csv(' .csv', index=True)

    # get the data for the specific day when we have the CSV
    input_file = 'allCoins.csv' 
    df = pd.read_csv(input_file)

    filtered_df = df[df['block_date'] == date]

    # Step 3: Export the filtered DataFrame to a new CSV
    month = (datetime.strptime(date, "%Y-%m-%d").strftime("%b")).lower()
    day = date.split("-")[2]
    output_file = f'{month}{day}Coins.csv'
    filtered_df.to_csv(output_file, index=False)

################################### END END ######################################################

# helper function to add hours to a Unix timestamp
def add_hours_to_unix_time(unix_time, hours):
    seconds_to_add = hours * 3600
    return unix_time + seconds_to_add

################################## Dexscreener API functions ####################################

# dexscreener API liquidity pool data
def get_LP_dexAPI(chain, CA):
    url = f"https://api.dexscreener.com/token-pairs/v1/{chain}/{CA}"
    
    response = requests.get(url)
    
    # Raise an HTTPError if the response was unsuccessful
    response.raise_for_status()
    
    data = response.json()
    
    pool = ""
    
    if not data:
        pool = 0
        # raise ValueError(f"!!!! CA WRONG !!!!{CA}")
    else:
        pool = data[0]["pairAddress"]
    
    return pool

# dexscreener API market cap data
def get_MC_dexAPI(chain, CA):
    url = f"https://api.dexscreener.com/token-pairs/v1/{chain}/{CA}"
    
    response = requests.get(url)
    
    # Raise an HTTPError if the response was unsuccessful
    response.raise_for_status()
    
    data = response.json()
        
    if not data:
        MC = 0
        # raise ValueError(f"!!!! CA WRONG !!!!{CA}")
    else:
        MC = data[0]["marketCap"]
    
    return MC

# get the date when the pair was created (in seconds Unix timestamp)
def get_datecreated_dexAPI(chain, CA):
    url = f"https://api.dexscreener.com/token-pairs/v1/{chain}/{CA}"
    
    response = requests.get(url)
    
    # Raise an HTTPError if the response was unsuccessful
    response.raise_for_status()
    
    data = response.json()
    
    if not data:
        date = 0
        # raise ValueError(f"!!!! CA WRONG !!!! {CA}")
    else:
        date = data[0]["pairCreatedAt"]
    
    # check if the date is in milliseconds or seconds and convert it to seconds (and skip if date is 0)
    if date > 1e12:
        date = date // 1000
    return date

# LIMIT: 30 CALLS PER MINUTE
# get the OHLCV data for a given pair for "hours" after the pair was created (max 10 hrs)
def ohlcv(chain, CA, hours, max_retries=20, sleep_on_error=5):
    """
    Fetch the OHLCV data. Retries on errors (network issues, rate-limits, etc.).
    
    :param chain: Network chain name
    :param CA: Contract address
    :param hours: Hours back from 'date' to fetch
    :param max_retries: Maximum number of retries before giving up 
    :param sleep_on_error: Seconds to wait between retries
    :return: List of OHLCV data or empty list if pool is invalid
    """
    pool = getLiqPool(chain, CA)
    date = getDateCreated(chain, CA)
    before_timestamp = add_hours_to_unix_time(date, hours)

    # If the pair is invalid
    if pool == 0:
        return []

    url = f"https://api.geckoterminal.com/api/v2/networks/{chain}/pools/{pool}/ohlcv/minute?aggregate=1&before_timestamp={before_timestamp}&limit=600&currency=usd&token={CA}"

    attempts = 0
    while True:
        try:
            response = requests.get(url, timeout=10)  # Include a timeout to avoid hanging
            response.raise_for_status()              # Raise an HTTPError if the response was unsuccessful
            
            data = response.json()
            data = data["data"]["attributes"]["ohlcv_list"]
            return data  # If successful, return the data immediately

        except requests.exceptions.HTTPError as http_err:
            # Handle HTTP-specific errors (e.g., 429 rate limit)
            attempts += 1
            print(f"HTTP error: {http_err} — Attempt {attempts}/{max_retries}. Retrying in {sleep_on_error}s.")
            if attempts >= max_retries:
                print("Max retries reached. Returning empty data.")
                return []
            time.sleep(sleep_on_error)

        except requests.exceptions.ConnectionError as conn_err:
            # This will catch your "RemoteDisconnected" error among other connection issues
            attempts += 1
            print(f"Connection error: {conn_err} — Attempt {attempts}/{max_retries}. Retrying in {sleep_on_error}s.")
            if attempts >= max_retries:
                print("Max retries reached. Returning empty data.")
                return []
            time.sleep(sleep_on_error)
            
        except requests.exceptions.RequestException as req_err:
            # Catches all other requests-related errors (including timeouts, too)
            attempts += 1
            print(f"Request error: {req_err} — Attempt {attempts}/{max_retries}. Retrying in {sleep_on_error}s.")
            if attempts >= max_retries:
                print("Max retries reached. Returning empty data.")
                return []
            time.sleep(sleep_on_error)


################################## End of Dexscreener API functions ##################################


###################################### CoinGecko API function ####################################

# takes datetime in the format "yyyy-mm-dd hh:mm:ss.000 UTC" and returns the Unix timestamp in seconds
def getUnixTimestamp(datetime_str):
    # Parse the datetime string into a datetime object
    dt = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S.%f UTC")
    # Convert to a Unix timestamp
    unix_timestamp = int(dt.timestamp())
    return unix_timestamp

# takes a token address and returns the pool address and fdv data
def get_pool_data_CG(CA, max_retries=20, sleep_on_error=5):
    url = f"https://api.geckoterminal.com/api/v2/networks/solana/tokens/{CA}/pools?page=1"
    
    attempts = 0

    while True:
        try:
            response = requests.get(url, timeout=10)  # Include a timeout to avoid hanging
            response.raise_for_status()              # Raise an HTTPError if the response was unsuccessful
            
            data = response.json()
            pool = data["data"][0]["attributes"]["address"]
            fdv = data["data"][0]["attributes"]["fdv_usd"]
            
            # delaying for 2 seconds to avoid rate limit
            time.sleep(2)
                    
            return [pool, fdv]  # If successful, return the data immediately
                    
        except requests.exceptions.HTTPError as http_err:
            # error handling for when the token is not found (return empty data)
            if response.status_code == 404:
                print(f"Token not found: {CA} — Returning empty data.")
                return ['',0]
            
            # Handle HTTP-specific errors (e.g., 429 rate limit)
            attempts += 1
            print(f"HTTP error: {http_err} — Attempt {attempts}/{max_retries}. Retrying in {sleep_on_error}s.")
            if attempts >= max_retries:
                print("Max retries reached. Returning empty data.")
                return []
            time.sleep(sleep_on_error)

        except requests.exceptions.ConnectionError as conn_err:
            # This will catch your "RemoteDisconnected" error among other connection issues
            attempts += 1
            print(f"Connection error: {conn_err} — Attempt {attempts}/{max_retries}. Retrying in {sleep_on_error}s.")
            if attempts >= max_retries:
                print("Max retries reached. Returning empty data.")
                return []
            time.sleep(sleep_on_error)
            
        except requests.exceptions.RequestException as req_err:
            # Catches all other requests-related errors (including timeouts, too)
            attempts += 1
            print(f"Request error: {req_err} — Attempt {attempts}/{max_retries}. Retrying in {sleep_on_error}s.")
            if attempts >= max_retries:
                print("Max retries reached. Returning empty data.")
                return []
            time.sleep(sleep_on_error)

# takes a csv of all tokens and gets the pools and fdv data for each token and exports the updated csv
def pools_on_CG(all_tokens_csv):
    # extract data from the csv
    
    df = pd.read_csv(all_tokens_csv)
    print ("Starting to get LP data")
    
    # add a new column called lp
    df['index'] = df.index
    df['LP'] = ''
    df['fdv'] = 0.0
    
    # add timestamp to the csv
    df['timestamp'] = df['block_time'].apply(lambda x: getUnixTimestamp(x))
    
    # loop over the tokens and get the pool data
    for index, row in df.iterrows():
        print (f"Getting LP data for token {index}")
        data = get_pool_data_CG(row['token_address'])
        df.at[index, 'LP'] = data[0]
        df.at[index, 'fdv'] = float(data[1])
        
    # export the updated csv
    fileName = all_tokens_csv.split(".")[0]
    df.to_csv(f'{fileName}_LP.csv', index=False)

# get the OHLCV data for a given pair for "hours" after the pair was created (max 10 hrs)
def ohlcv_for_pool(chain, CA, pool, timestamp, hours, max_retries=20, sleep_on_error=5):
    """
    Fetch the OHLCV data. Retries on errors (network issues, rate-limits, etc.).
    
    :param chain: Network chain name
    :param CA: Contract address
    :param hours: Hours back from 'date' to fetch
    :param max_retries: Maximum number of retries before giving up 
    :param sleep_on_error: Seconds to wait between retries
    :return: List of OHLCV data or empty list if pool is invalid
    """
    before_timestamp = add_hours_to_unix_time(timestamp, hours)
    
    if pool == '':
        return []

    url = f"https://api.geckoterminal.com/api/v2/networks/{chain}/pools/{pool}/ohlcv/minute?aggregate=1&before_timestamp={before_timestamp}&limit=600&currency=usd&token={CA}"

    attempts = 0
    while True:
        try:
            response = requests.get(url, timeout=10)  # Include a timeout to avoid hanging
            response.raise_for_status()              # Raise an HTTPError if the response was unsuccessful
            
            data = response.json()
            data = data["data"]["attributes"]["ohlcv_list"]
            return data  # If successful, return the data immediately

        except requests.exceptions.HTTPError as http_err:
            # Handle HTTP-specific errors (e.g., 429 rate limit)
            attempts += 1
            print(f"HTTP error: {http_err} — Attempt {attempts}/{max_retries}. Retrying in {sleep_on_error}s.")
            if attempts >= max_retries:
                print("Max retries reached. Returning empty data.")
                return []
            time.sleep(sleep_on_error)

        except requests.exceptions.ConnectionError as conn_err:
            # This will catch your "RemoteDisconnected" error among other connection issues
            attempts += 1
            print(f"Connection error: {conn_err} — Attempt {attempts}/{max_retries}. Retrying in {sleep_on_error}s.")
            if attempts >= max_retries:
                print("Max retries reached. Returning empty data.")
                return []
            time.sleep(sleep_on_error)
            
        except requests.exceptions.RequestException as req_err:
            # Catches all other requests-related errors (including timeouts, too)
            attempts += 1
            print(f"Request error: {req_err} — Attempt {attempts}/{max_retries}. Retrying in {sleep_on_error}s.")
            if attempts >= max_retries:
                print("Max retries reached. Returning empty data.")
                return []
            time.sleep(sleep_on_error)

# takes a csv of all tokens and returns the ohlcv data for each token (as a csv for each token)
def get_full_ohlcv_from_allTokens(all_tokens_csv, output_folder):
    # takes the list of valid tokens and gets the ohlcv data for each token
    df = pd.read_csv(all_tokens_csv)
    
    # import the list of all tokens, add the pools and timestamp to the csv, export the updated csv
    pools_on_CG(all_tokens_csv)
    
    # import the updated csv, get the ohlcv data for each token
    fileName = all_tokens_csv.split(".")[0]
    df = pd.read_csv(f'{fileName}_LP.csv')
    
    # loop over the tokens and get the ohlcv data
    print ("Starting to get OHLCV data")
    for index, row in df.iterrows():
        data = ohlcv_for_pool(chain, row['token_address'], row['LP'], row['timestamp'], 10)
        
        if len(data) == 0:
            print (f"Data not found for token {index}")
            continue
        # export data to csv with title is the index of the token
        file_path = f'{output_folder}/{index}.csv'
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
        
        if index % 100 == 0:
            print (f"Done getting data for {index} tokens")
            
        time.sleep(2)
    
    print (f"Done getting OHLCV data!!!!!! for {all_tokens_csv}")



# takes a solscan csv file and gets the pools and fdv data for each token
def pools_for_solscan(all_tokens_csv):
    # extract data from the csv
    
    df = pd.read_csv(all_tokens_csv)
    print ("Starting to get LP data")
    
    # add a new column called lp
    df['index'] = df.index
    df['LP'] = ''
    df['fdv'] = 0.0
    
    # add timestamp to the csv
        
    # loop over the tokens and get the pool data
    for index, row in df.iterrows():
        print (f"Getting LP data for token {index}")
        data = get_pool_data_CG(row['TokenAddress'])
        df.at[index, 'LP'] = data[0]
        df.at[index, 'fdv'] = float(data[1])
        
    # export the updated csv
    fileName = all_tokens_csv.split(".")[0]
    df.to_csv(f'{fileName}_LP.csv', index=False)
def get_full_ohlcv_from_allTokens_solscan(all_tokens_csv, output_folder):
    # takes the list of valid tokens and gets the ohlcv data for each token
    df = pd.read_csv(all_tokens_csv)
    
    # import the list of all tokens, add the pools and timestamp to the csv, export the updated csv
    pools_for_solscan(all_tokens_csv)
    
    # import the updated csv, get the ohlcv data for each token
    fileName = all_tokens_csv.split(".")[0]
    df = pd.read_csv(f'{fileName}_LP.csv')
    
    # loop over the tokens and get the ohlcv data
    print ("Starting to get OHLCV data")
    for index, row in df.iterrows():
        data = ohlcv_for_pool(chain, row['TokenAddress'], row['LP'], row['Time'], 10)
        
        # export data to csv with title is the index of the token
        file_path = f'{output_folder}/{index}.csv'
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
        
        if index % 100 == 0:
            print (f"Done getting data for {index} tokens")
            
        time.sleep(2)
    
    print (f"Done getting OHLCV data!!!!!! for {all_tokens_csv}")


###################################### End of CoinGecko API function ####################################


december_days = [
    ['dec01Coins', 'dec01_ohlcv'], 
    ['dec02Coins', 'dec02_ohlcv'], 
    ['dec03Coins', 'dec03_ohlcv'], 
    ['dec04Coins', 'dec04_ohlcv'], 
    ['dec05Coins', 'dec05_ohlcv'], 
    ['dec06Coins', 'dec06_ohlcv'], 
    ['dec07Coins', 'dec07_ohlcv'], 
    ['dec08Coins', 'dec08_ohlcv'], 
    ['dec09Coins', 'dec09_ohlcv'], 
    ['dec10Coins', 'dec10_ohlcv'], 
    ['dec11Coins', 'dec11_ohlcv'], 
    ['dec12Coins', 'dec12_ohlcv'], 
    ['dec13Coins', 'dec13_ohlcv'], 
    ['dec14Coins', 'dec14_ohlcv'], 
    ['dec15Coins', 'dec15_ohlcv'], 
    ['dec16Coins', 'dec16_ohlcv'], 
    ['dec17Coins', 'dec17_ohlcv'], 
    ['dec18Coins', 'dec18_ohlcv'], 
    ['dec19Coins', 'dec19_ohlcv'], 
    ['dec20Coins', 'dec20_ohlcv'], 
    ['dec21Coins', 'dec21_ohlcv'], 
    ['dec22Coins', 'dec22_ohlcv'], 
    ['dec23Coins', 'dec23_ohlcv'], 
    ['dec24Coins', 'dec24_ohlcv'], 
    ['dec25Coins', 'dec25_ohlcv'], 
    ['dec26Coins', 'dec26_ohlcv'], 
    ['dec27Coins', 'dec27_ohlcv'], 
    ['dec28Coins', 'dec28_ohlcv'], 
    ['dec29Coins', 'dec29_ohlcv'], 
    ['dec30Coins', 'dec30_ohlcv'], 
    ['dec31Coins', 'dec31_ohlcv']
]

list_of_days_19_26 = [['jan20Coins', 'jan20_ohlcv'],
                      ['jan21Coins', 'jan21_ohlcv'],
                      ['jan22Coins', 'jan22_ohlcv'],
                      ['jan23Coins', 'jan23_ohlcv'],
                      ['jan24Coins', 'jan24_ohlcv'],
                      ['jan25Coins', 'jan25_ohlcv'],
                      ['jan26Coins', 'jan26_ohlcv']
    ]

# for day in list_of_days:
#     get_full_ohlcv_from_allTokens(f'{day[0]}.csv',  day[1])

# for day in list_of_days_19_26:
#     get_full_ohlcv_from_allTokens_solscan(f'{day[0]}.csv',  day[1])