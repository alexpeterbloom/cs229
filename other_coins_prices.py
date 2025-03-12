import csv
import bisect
import json

def load_candles(csv_filename):
    """
    Loads candle data from a CSV file into a list of dictionaries.
    Each dictionary contains: start, open, and volume.
    The 'start' field is converted to an integer and 'open' and 'volume' to floats.
    The returned list is sorted by the 'start' timestamp.
    """
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
    """
    Returns the candle in 'candles' whose 'start' is closest to the given timestamp.
    Uses binary search (via bisect) for efficiency.
    """
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
    """
    Returns the sum of volumes for candles whose start time is in the interval:
    [start_interval, end_interval)
    """
    total = sum(candle["volume"] for candle in candles if start_interval < candle["timestamp"] <= end_interval + 10)
    return total

def get_market_data(migration_ts, candles, sol = True):
    """
    Given a migration timestamp and a sorted list of candles, returns a dictionary with:
      - price_migration: SOL price at migration time (from the closest candle's open)
      - vol_24h: Sum of volume for the 24 hours preceding the migration timestamp
      - price_change_24h: Price difference between the migration candle's open and 
                          the open from the candle closest to 24 hours before
      - price_change_7d: Price difference between the migration candle's open and 
                         the open from the candle closest to 7 days before
      - volume_change_7d: Volume difference between the migration candle and the candle 
                          closest to 7 days before
    """
    # Find the candle closest to the migration timestamp.
    migration_candle = find_closest_candle(migration_ts, candles)
    print (migration_candle)
    
    # Calculate 24h metrics:
    ts_24h = migration_ts - 24 * 3600
    candle_24h = find_closest_candle(ts_24h, candles)
    print (candle_24h)
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

# Example usage:
if __name__ == "__main__":
    
    # example use case
    
    sol_candles = load_candles("SOL-USD.csv")
    btc_candles = load_candles("BTC-USD.csv")
    
    json_file = "sep_feb_data/feb01_0.json"
    with open(json_file, "r") as f:
        data = json.load(f)
        migration_ts = data["token_data"]["migration_timestamp"]
        print (migration_ts)
        sol_info = get_market_data(migration_ts, sol_candles, sol = True)
        btc_info = get_market_data(migration_ts, btc_candles, sol = False)

        # Update the JSON data with the new market info.
        data["processed_data"]["sol_market_data"] = sol_info
        data["processed_data"]["btc_market_data"] = btc_info
        
    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)
        
    
                
