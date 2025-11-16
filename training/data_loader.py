# training/data_loader.py
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
import os
from dotenv import load_dotenv
import time
import traceback

def _get_interval_ms(interval_str):
    """Converts interval (e.g. '15m') to milliseconds."""
    if interval_str == Client.KLINE_INTERVAL_15MINUTE: return 15 * 60 * 1000
    if interval_str == Client.KLINE_INTERVAL_1HOUR: return 60 * 60 * 1000
    if interval_str == Client.KLINE_INTERVAL_4HOUR: return 4 * 60 * 60 * 1000
    # (Add others as needed)
    return 15 * 60 * 1000 # Default 15m

def download_data(pair, interval, data_points, api_key, api_secret, use_testnet, cache_file=None):
    """Downloads Klines AND Funding Rates, merges them and caches."""
    if cache_file and os.path.exists(cache_file):
        try:
            print(f"Loading data from cache: {cache_file}")
            df = pd.read_parquet(cache_file)
            if len(df) >= data_points and 'funding_rate' in df.columns:
                 print("Data (Klines+Funding) successfully loaded from cache.")
                 if not isinstance(df.index, pd.DatetimeIndex):
                      # ... (code to restore index) ...
                      pass
                 df.sort_index(inplace=True)
                 return df.iloc[-data_points:]
            else: 
                 print(f"Cache incomplete (Data: {len(df)} < {data_points} or no 'funding_rate'). Reloading...")
        except Exception as e: print(f"Cache read error {cache_file}: {e}. Reloading...")
        try: os.remove(cache_file)
        except OSError: pass

    # 1. Download Klines
    df_klines = download_fresh_klines(pair, interval, data_points, api_key, api_secret, use_testnet)
    if df_klines.empty: return pd.DataFrame()
    
    # 2. Download Funding Rate
    # We need funding history covering the same period
    start_time_ms = int(df_klines.index[0].timestamp() * 1000)
    df_funding = download_fresh_funding_rate(pair, start_time_ms, api_key, api_secret, use_testnet)
    if df_funding.empty:
        print("⚠️ Failed to load Funding Rate. 'funding_rate' feature will be NaN.")
        df_klines['funding_rate'] = np.nan
    else:
        # 3. Merge
        # Klines (15m) + Funding (4h/8h). Apply ffill() to fill gaps.
        df_merged = pd.merge_asof(
            df_klines.sort_index(), 
            df_funding.sort_index(), 
            left_index=True, 
            right_index=True, 
            direction='backward' # Take last known funding rate value
        )
        # ffill() in case first klines were before first funding rate
        df_merged['funding_rate'] = df_merged['funding_rate'].ffill() 
        df_klines = df_merged

    if cache_file:
        try:
            print(f"Saving data (Klines+Funding) to cache: {cache_file}")
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            df_klines.to_parquet(cache_file)
        except Exception as e: print(f"Cache save error: {e}")
            
    return df_klines.iloc[-data_points:] # Return last N

def download_fresh_klines(pair, interval, data_points, api_key, api_secret, use_testnet):
    """(Auxiliary) Downloads ONLY klines from Binance."""
    print(f"Downloading {data_points} candles for {pair} {interval} from Binance...")
    client = Client(api_key, api_secret, testnet=use_testnet)

    klines_all = []
    limit_per_req = 1000 
    end_ts_str = None 
    interval_ms = _get_interval_ms(interval)

    while len(klines_all) < data_points:
        fetch_limit = min(limit_per_req, data_points - len(klines_all))
        print(f"Requesting {fetch_limit} candles...", end="")
        try:
            klines = client.get_historical_klines(pair, interval, start_str=None, end_str=end_ts_str, limit=fetch_limit)
            print(f" Received {len(klines)}.")
            if not klines: print("No more data."); break
            
            if klines_all and klines_all[0][0] == klines[-1][0]:
                 klines = klines[:-1] 
                 if not klines: continue 

            klines_all = klines + klines_all 
            
            oldest_kline_ts = klines[0][0] 
            # end_ts_str = pd.to_datetime(oldest_kline_ts - 1, unit='ms').strftime('%Y-%m-%d %H:%M:%S')
            # More reliable klines pagination method - use (timestamp - 1 interval)
            end_ts = oldest_kline_ts - interval_ms
            end_ts_str = pd.to_datetime(end_ts, unit='ms').strftime('%Y-%m-%d %H:%M:%S')
            
            time.sleep(0.5) 
            
        except BinanceAPIException as e: print(f"\n❌ Binance API error: {e}. Retry in 5 sec..."); time.sleep(5)
        except Exception as e: print(f"\n❌ Unknown error: {e}. Retry in 5 sec..."); traceback.print_exc(limit=2); time.sleep(5)
            
    print(f"Total downloaded {len(klines_all)} klines.")
    if not klines_all: return pd.DataFrame()

    df = pd.DataFrame(klines_all, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
        'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
        'taker_buy_quote_asset_volume', 'ignore'
    ])
    
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)
    df = df[~df.index.duplicated(keep='first')]; df.sort_index(inplace=True)
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 'number_of_trades', 
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
    for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def download_fresh_funding_rate(pair, start_time_ms, api_key, api_secret, use_testnet):
    """(Auxiliary) Downloads funding rate history from Binance."""
    print(f"Downloading Funding Rate for {pair}...")
    client = Client(api_key, api_secret, testnet=use_testnet)
    
    all_funding_rates = []
    limit = 1000 # Max limit
    
    try:
        # Get data in parts
        while True:
            print(f"Requesting 1000 Funding Rates (starting from {start_time_ms})...")
            funding_rates = client.futures_funding_rate(
                symbol=pair, 
                startTime=start_time_ms, 
                limit=limit
            )
            
            if not funding_rates:
                print("No more Funding Rate data.")
                break
                
            all_funding_rates.extend(funding_rates)
            
            # Set new startTime for next request
            # +1 ms to last received time
            start_time_ms = funding_rates[-1]['fundingTime'] + 1
            
            if len(funding_rates) < limit:
                break # Reached end of history
            
            time.sleep(0.5)

        if not all_funding_rates:
             print("⚠️ Funding Rate not found.")
             return pd.DataFrame()
        
        print(f"Total downloaded {len(all_funding_rates)} Funding Rate records.")
        
        df_funding = pd.DataFrame(all_funding_rates)
        df_funding['fundingTime'] = pd.to_datetime(df_funding['fundingTime'], unit='ms')
        df_funding.set_index('fundingTime', inplace=True)
        df_funding['funding_rate'] = pd.to_numeric(df_funding['fundingRate'], errors='coerce')
        
        # Keep only needed column
        return df_funding[['funding_rate']]
        
    except BinanceAPIException as e:
        print(f"❌ Binance API error (Funding Rate): {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ Unknown error (Funding Rate): {e}")
        traceback.print_exc(limit=2)
        return pd.DataFrame()

# ... (if __name__ == '__main__': block for test) ...
if __name__ == '__main__':
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    load_dotenv(dotenv_path=dotenv_path) 
    
    API_KEY = os.getenv("BINANCE_API_KEY") 
    API_SECRET = os.getenv("BINANCE_API_SECRET")
    USE_TESTNET = os.getenv("USE_TESTNET", "False").lower() in ('true', '1', 't')

    PAIR = "SOLUSDT"
    INTERVAL = Client.KLINE_INTERVAL_15MINUTE
    DATA_POINTS = 5000 
    CACHE_DIR = "../data"
    CACHE_FILE = os.path.join(CACHE_DIR, f"{PAIR}_{INTERVAL}_patchtst_cache.parquet") # Unique name
    
    # Test COMBINED download
    df_data = download_data(PAIR, INTERVAL, DATA_POINTS, API_KEY, API_SECRET, USE_TESTNET, CACHE_FILE)
    
    if not df_data.empty:
        print("\nSample downloaded data (Klines + Funding):")
        print(df_data.head())
        print(df_data.tail())
        print(f"\nDataFrame size: {df_data.shape}")
        print("\nFunding rate check (should have NaN and numbers):")
        print(df_data['funding_rate'].describe())
    else:
        print("Failed to download data for test.")