# training/feature_engineering.py
import pandas as pd
import numpy as np
import pandas_ta as ta
import traceback

def get_session(hour):
    """Determines trading session by UTC hour."""
    if 0 <= hour < 8: # (00:00 - 07:59 UTC)
        return 'asia'
    elif 8 <= hour < 13: # (08:00 - 12:59 UTC) - London Open
        return 'europe'
    elif 13 <= hour < 21: # (13:00 - 20:59 UTC) - NY Open
        return 'usa'
    else: # (21:00 - 23:59 UTC) - Cooldown
        return 'off'

def generate_features(df_input: pd.DataFrame, use_rolling_norm: bool = False, rolling_window: int = 200) -> pd.DataFrame:
    """
    Generates advanced feature set for PatchTST / Quantile Regression model.
    """
    print("Generating features (PatchTST)...")
    df = df_input.copy()

    # --- 1. Basic + VWAP + Taker Ratio ---
    try:
        print("- Calculating VWAP, Taker Ratio, Log Return...")
        # VWAP (requires H, L, C, V)
        df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
        
        # Taker Buy Ratio (Proxy for CVD)
        volume_safe = df['volume'].replace(0, np.nan)
        if 'taker_buy_base_asset_volume' in df.columns:
             df['taker_buy_ratio'] = df['taker_buy_base_asset_volume'] / volume_safe
        else:
             print("⚠️ 'taker_buy_base_asset_volume' missing! (Needed for Binance)")
             df['taker_buy_ratio'] = np.nan # Placeholder
             
        # Log Return (for normalization and as feature)
        df['log_return'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
        
    except Exception as e: print(f"⚠️ Basic feature error: {e}"); traceback.print_exc(limit=1)

    # --- 2. Technical features (EMA, RSI, ATR, OBV, MACD, Stoch, BB) ---
    try:
        print("- Calculating EMA/MA (9, 21, 50, 200)...")
        for length in [9, 21, 50, 200]: df[f'EMA_{length}'] = ta.ema(df['close'], length=length)
        
        print("- Calculating RSI, ATR, OBV, MACD...")
        df.ta.rsi(length=7, append=True, col_names=('RSI_7',))
        df.ta.rsi(length=14, append=True, col_names=('RSI_14',))
        
        # ATR and ATR%
        atr_series = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['ATR_14'] = atr_series
        df['ATR_14_pct'] = (atr_series / df['close']) * 100 # ATR as % of price
        
        obv = ta.obv(df['close'], df['volume']); df = df.join(obv.rename('OBV'))
        df.ta.macd(append=True, col_names=('MACD', 'MACD_h', 'MACD_s'))
        
        print("- Calculating StochRSI, Bollinger Bands...")
        if 'RSI_14' in df.columns and df['RSI_14'].notna().sum() >= 20: # 14+3+3
             stoch_rsi = ta.stochrsi(df['RSI_14'].dropna(), length=14, rsi_length=14, k=3, d=3)
             if isinstance(stoch_rsi, pd.DataFrame) and not stoch_rsi.empty:
                  stoch_k_col=next((c for c in stoch_rsi.columns if c.startswith('STOCHRSIk')),None)
                  stoch_d_col=next((c for c in stoch_rsi.columns if c.startswith('STOCHRSId')),None)
                  if stoch_k_col: df = df.join(stoch_rsi[stoch_k_col].rename('STOCHRSIk'))
                  if stoch_d_col: df = df.join(stoch_rsi[stoch_d_col].rename('STOCHRSId'))
             else: df['STOCHRSIk'] = df['STOCHRSId'] = np.nan
        else: print("⚠️ Skipping StochRSI: RSI_14 not calculated."); df['STOCHRSIk'] = df['STOCHRSId'] = np.nan

        # Bollinger Bands Width
        bbands = ta.bbands(df['close'], length=20, std=2)
        if bbands is not None and isinstance(bbands, pd.DataFrame) and not bbands.empty:
            bbu_col = next((c for c in bbands.columns if c.startswith('BBU_')), None)
            bbl_col = next((c for c in bbands.columns if c.startswith('BBL_')), None)
            if bbu_col and bbl_col:
                # (Upper - Lower) / Middle
                df['BB_width'] = (bbands[bbu_col] - bbands[bbl_col]) / ta.sma(df['close'], length=20)
            else: df['BB_width'] = np.nan
        else: df['BB_width'] = np.nan
        
    except Exception as e: print(f"⚠️ Technical feature error: {e}"); traceback.print_exc(limit=1)

    # --- 3. Market structure (Funding Rate + Sessions/Weekends) ---
    try:
        print("- Calculating Time features (Sessions, Weekends)...")
        # Funding Rate should already be in df from data_loader, just ffill
        if 'funding_rate' in df.columns:
            df['funding_rate'] = df['funding_rate'].ffill()
        else:
            print("⚠️ 'funding_rate' missing, filling with 0.")
            df['funding_rate'] = 0.0 # Placeholder if download failed
            
        df['is_weekend'] = (df.index.weekday >= 5).astype(int) # 5=Saturday, 6=Sunday
        
        # Hourly features
        df['hour_of_day'] = df.index.hour
        df['day_of_week'] = df.index.weekday
        df['month_of_year'] = df.index.month
        df['minute_of_day'] = df.index.hour * 60 + df.index.minute

        # Session flags (One-Hot Encoding)
        sessions = df['hour_of_day'].apply(get_session)
        df['session_asia'] = (sessions == 'asia').astype(int)
        df['session_europe'] = (sessions == 'europe').astype(int)
        df['session_usa'] = (sessions == 'usa').astype(int)
        
    except Exception as e: print(f"⚠️ Time feature error: {e}"); traceback.print_exc(limit=1)

    # --- 4. Rolling Normalization (Optional) ---
    # This step is better done in train_model.py and model.py,
    # because rolling mean/std can "leak" (look-ahead) with simple calculation
    # But if needed as feature:
    # if use_rolling_norm:
    #     print(f"- Calculating Rolling Normalization (Window: {rolling_window})...")
    #     rolling_mean = df['close'].rolling(window=rolling_window, min_periods=30).mean()
    #     rolling_std = df['close'].rolling(window=rolling_window, min_periods=30).std()
    #     df['close_norm_rolling'] = (df['close'] - rolling_mean) / rolling_std
    
    print("✅ Feature generation (PatchTST) completed.")
    return df

# ... (if __name__ == '__main__': block for test) ...