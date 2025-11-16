# bot/model.py
import numpy as np
import pandas as pd
import pandas_ta as ta
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import traceback
from . import config 
from .binance_client import BinanceClient
from training.feature_engineering import generate_features 

# --- SETTINGS ---
SEQUENCE_LENGTH = config.SEQUENCE_LENGTH 
MODEL_FILE = config.MODEL_FILE 
SCALER_FILE = config.SCALER_FILE 

# --- GLOBAL VARIABLES ---
MODEL = None
SCALER = None

# --- ❗️ FIX: "Wrapper" Quantile Loss (should be here) ---
def create_quantile_loss(q):
    """
    This function *creates* and *returns* a loss function,
    which "remembers" the q value.
    """
    def quantile_loss(y_true, y_pred):
        """Pinball loss function for Quantile Regression."""
        e = y_true - y_pred
        return tf.keras.backend.mean(
            tf.keras.backend.maximum(q * e, (q - 1) * e), 
            axis=-1
        )
    # ❗️ IMPORTANT: Give function UNIQUE NAME (must match train_model.py)
    quantile_loss.__name__ = f'quantile_loss_q{int(q*100)}'
    return quantile_loss
# --- END OF FIX ---


# --- MODEL LOADING ---
def load_model_and_scaler():
    """Loads model and scaler into global variables."""
    global MODEL, SCALER
    if MODEL and SCALER: return True
    try:
        # --- ❗️ FIX: Create custom_objects ---
        QUANTILES = config.QUANTILES # [0.10, 0.50, 0.90]
        losses = [create_quantile_loss(q=q) for q in QUANTILES]
        
        # Create dictionary { 'function_name': function }
        custom_objects = {
            losses[0].__name__: losses[0], # {'quantile_loss_q10': ...}
            losses[1].__name__: losses[1], # {'quantile_loss_q50': ...}
            losses[2].__name__: losses[2]  # {'quantile_loss_q90': ...}
        }
        print(f"Loading model with custom_objects: {custom_objects.keys()}")
        # --- END OF FIX ---

        MODEL = load_model(MODEL_FILE, custom_objects=custom_objects) # <-- Pass dictionary
        SCALER = joblib.load(SCALER_FILE)
        
        if not hasattr(SCALER, 'feature_names_in_'): print("⚠️ Scaler does not contain feature list.")
        print(f"✅ 15M 'PatchTST' CNN+Transformer model ({config.PAIR}) and scaler loaded.")
        return True
    except FileNotFoundError:
        print(f"❌ ERROR: Model file ('{MODEL_FILE}') or scaler file ('{SCALER_FILE}') not found.")
        print(f"--- MAKE SURE YOU RETRAINED THE MODEL (train_model.py) FOR {config.PAIR} 15M (PatchTST) ---")
        return False
    except Exception as e: 
        print(f"❌ Error loading 'PatchTST' 15M model/scaler: {e}")
        traceback.print_exc(limit=3)
        return False

# --- PREDICTION FUNCTION ---
def get_prediction_with_indicators(klines: list, binance: BinanceClient) -> tuple:
    # (Code of this function remains UNCHANGED, as in previous answer)
    if MODEL is None or SCALER is None:
        print("❌ Model/scaler not loaded (error in __init__).")
        return ([0, 0, 0], None)
    
    required_length = SEQUENCE_LENGTH + 250 
    if len(klines) < required_length: 
        print(f"⚠️ Insufficient data ({len(klines)} < {required_length}) for PatchTST indicators.")
        return ([0, 0, 0], None)

    # 1. DataFrame Klines
    df_klines = pd.DataFrame(klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
        'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    df_klines['open_time'] = pd.to_datetime(df_klines['open_time'], unit='ms'); df_klines.set_index('open_time', inplace=True)
    numeric_cols = ['open','high','low','close','volume','quote_asset_volume','number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume']
    for col in numeric_cols: df_klines[col] = pd.to_numeric(df_klines[col], errors='coerce')
    df_klines.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)
    if len(df_klines) < required_length - SEQUENCE_LENGTH: 
        print(f"⚠️ Insufficient data ({len(df_klines)}) after OHLCV cleaning."); return ([0, 0, 0], None)
    
    df = df_klines.copy()

    # 2. Get and merge Funding Rate
    try:
        print("Loading Funding Rate for prediction...")
        df_funding = binance.get_funding_rate_history(limit=100) 
        if df_funding.empty:
            print("⚠️ Failed to load Funding Rate. 'funding_rate' will be NaN.")
            df['funding_rate'] = np.nan
        else:
            df = pd.merge_asof(
                df.sort_index(), df_funding.sort_index(), 
                left_index=True, right_index=True, direction='backward'
            )
            df['funding_rate'] = df['funding_rate'].ffill()
    except Exception as e_fund:
        print(f"❌ Error merging Funding Rate: {e_fund}"); df['funding_rate'] = np.nan

    # 3. Generate "PatchTST" features
    print("Generating PatchTST features (prediction)...")
    try:
        df_features = generate_features(df, use_rolling_norm=False)
        print("✅ PatchTST features generated.")
    except Exception as feature_err:
        print(f"❌ PATCHTST FEATURE GENERATION ERROR: {feature_err}"); traceback.print_exc(limit=3)
        return ([0, 0, 0], None)

    # 4. Process NaN
    print(f"Rows BEFORE ffill/bfill: {len(df_features)}")
    df_processed = df_features.copy()
    df_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_processed.ffill(inplace=True); df_processed.bfill(inplace=True)
    print(f"Rows AFTER ffill + bfill: {len(df_processed)}")

    # 5. Check for NaN
    try: feature_list_from_scaler = list(SCALER.feature_names_in_)
    except AttributeError: print("❌ No feature list in scaler."); return ([0, 0, 0], None)
    
    key_filter_indicators = ['EMA_9', 'EMA_21', 'RSI_14', 'ATR_14'] 
    all_cols_needed = list(set(feature_list_from_scaler + key_filter_indicators))
    all_cols_needed = [col for col in all_cols_needed if col in df_processed.columns] 
    
    if not all_cols_needed: print("❌ No columns to check for NaN."); return ([0, 0, 0], None)

    initial_rows = len(df_processed)
    df_processed.dropna(subset=all_cols_needed, inplace=True)
    if len(df_processed) < initial_rows: print(f"Removed {initial_rows - len(df_processed)} rows with NaN (Full cleaning).")
    if len(df_processed) == 0: print("⚠️ No data after full NaN cleaning."); return ([0, 0, 0], None)

    if len(df_processed) < SEQUENCE_LENGTH:
         print(f"⚠️ Insufficient rows ({len(df_processed)}) for {SEQUENCE_LENGTH} window after NaN.")
         return ([0, 0, 0], None)

    # 6. Prepare data for model
    last_seq = df_processed.tail(SEQUENCE_LENGTH)
    missing = [f for f in feature_list_from_scaler if f not in last_seq.columns]
    if missing: 
        print(f"❌ Missing features for scaler: {missing}"); return ([0, 0, 0], None)

    try:
        data_to_scale = last_seq[feature_list_from_scaler].reindex(columns=feature_list_from_scaler)
        
        if data_to_scale.isnull().values.any():
             print("\n" + "="*50); print("❌ ERROR: NaN DETECTED BEFORE SCALING!");
             nan_cols_report = data_to_scale.isnull().sum(); print(nan_cols_report[nan_cols_report > 0])
             print(data_to_scale[nan_cols_report[nan_cols_report > 0].index].tail(5)); print("="*50 + "\n")
             return ([0, 0, 0], None)
        
        if not np.isfinite(data_to_scale).all().all():
             print("\n" + "="*50); print("❌ ERROR: INFINITE (inf) VALUES DETECTED!"); print("="*50 + "\n")
             return ([0, 0, 0], None)

        scaled = SCALER.transform(data_to_scale)
        
    except Exception as scale_err:
        print(f"❌ SCALING ERROR: {scale_err}"); traceback.print_exc(limit=2)
        return ([0, 0, 0], None)

    X_input = np.expand_dims(scaled, axis=0)

    # 7. Model prediction (QUANTILES)
    try:
        prediction_quantiles = MODEL.predict(X_input, verbose=0)[0] 
    except Exception as predict_err: 
        print(f"❌ Prediction error (Quantile): {predict_err}"); return ([0, 0, 0], None)

    # 8. Collect indicators for filter
    last_row_for_indicators = df_processed.iloc[-1]
    indicators = {
        "ema_fast": last_row_for_indicators.get('EMA_9'),
        "ema_slow": last_row_for_indicators.get('EMA_200'), 
        "rsi": last_row_for_indicators.get('RSI_14'), 
        "atr": last_row_for_indicators.get('ATR_14'), 
        "volume": last_row_for_indicators.get('volume'), 
        "close": last_row_for_indicators.get('close'), 
        "ema_mid": last_row_for_indicators.get('EMA_21'),
        "rsi_fast": last_row_for_indicators.get('RSI_7'), 
        "avg_atr": df_processed['ATR_14'].tail(20).median() if len(df_processed) >= 20 else np.nan,
        "avg_volume": df_processed['volume'].tail(20).median() if len(df_processed) >= 20 else np.nan,
    }

    if any(v is None or pd.isna(v) for v in indicators.values()):
        print("⚠️ NaN in filter indicators (AFTER get).")
        return (prediction_quantiles, None) 

    # 9. Return result
    print("✅ Prediction (Quantile) and indicators (PatchTST) successfully obtained.")
    return prediction_quantiles, indicators

# --- ❗️❗️❗️ LOAD MODEL ON IMPORT ---
load_model_and_scaler()