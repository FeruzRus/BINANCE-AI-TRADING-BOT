# training/train_model.py

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import tensorflow as tf # <-- Make sure tf is imported
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, LayerNormalization, Dense, Dropout, Add,
    MultiHeadAttention, GlobalAveragePooling1D, Concatenate
)
from tensorflow.keras.optimizers import Adam
from binance.client import Client # Needed for INTERVAL
import os
from dotenv import load_dotenv
import traceback

# Import our modules
from .data_loader import download_data
from .feature_engineering import generate_features

# --- SETTINGS ---
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path) 

API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
USE_TESTNET = os.getenv("USE_TESTNET", "False").lower() in ('true', '1', 't')
PAIR = os.getenv("PAIR", "SOLUSDT")
INTERVAL = Client.KLINE_INTERVAL_15MINUTE # 15M
DATA_POINTS = 100000 # ~2 years of 15-minute data

CACHE_DIR = "../data"
CACHE_FILE = os.path.join(CACHE_DIR, f"{PAIR}_{INTERVAL}_patchtst_cache.parquet") # Klines+Funding cache

# --- Data labeling parameters ---
LOOK_FORWARD_CANDLES = 4 # Forecast 1 hour ahead (4 * 15 min)

# --- Model parameters ---
SEQUENCE_LENGTH = 96 # Window 96 * 15m = 24 hours
# --- NEW FILE NAMES ---
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_FILE = os.path.join(ROOT_DIR, 'trading_model_binance_15m_patchtst.keras') 
SCALER_FILE = os.path.join(ROOT_DIR, 'scaler_binance_15m_patchtst.pkl')
PROCESSED_DATA_FILE = os.path.join(ROOT_DIR, 'processed_data_binance_15m_patchtst.parquet')

# --- Model Parameters (TCN + Attention) ---
NUM_HEADS = 4; FF_DIM = 64; NUM_TRANSFORMER_BLOCKS = 2
TCN_FILTERS = 32; TCN_KERNEL_SIZE = 3; TCN_DILATIONS = [1, 2, 4, 8]

# --- ❗️ NEW PARAMETERS: QUANTILES ---
QUANTILES = [0.10, 0.50, 0.90] # Q10 (risk), Q50 (median), Q90 (potential)
OUTPUT_SIZE = len(QUANTILES)

# --- 1. LOAD DATA ---
df_raw = download_data(PAIR, INTERVAL, DATA_POINTS, API_KEY, API_SECRET, USE_TESTNET, CACHE_FILE)
if df_raw.empty: print("❌ Failed to load data."); exit()

# --- 2. GENERATE FEATURES (NEW) ---
df = generate_features(df_raw, use_rolling_norm=False)

# --- ❗️ NEW "PatchTST" FEATURE LIST (from config.py) ---
feature_list_config = [
    'open', 'high', 'low', 'close', 'volume', 
    'taker_buy_ratio', 'vwap', 'log_return', 'EMA_9', 'EMA_21', 'EMA_50', 'EMA_200',
    'RSI_7', 'RSI_14', 'ATR_14', 'ATR_14_pct', 'OBV',
    'MACD', 'MACD_h', 'MACD_s',
    'STOCHRSIk', 'STOCHRSId', 'BB_width', 'funding_rate',
    'is_weekend', 'session_asia', 'session_europe', 'session_usa',
    'minute_of_day', 'hour_of_day', 'day_of_week', 'month_of_year',
]

# Check if all columns from list exist in DataFrame
actual_cols = [f for f in feature_list_config if f in df.columns]
missing_cols = list(set(feature_list_config) - set(actual_cols))
if missing_cols: 
    print(f"⚠️ WARNING: Could not create/find columns: {missing_cols}. They will be excluded.")
feature_list = actual_cols 
print(f"\nFinal PatchTST feature list ({len(feature_list)}): {feature_list}")

# --- 3. CREATE TARGET VARIABLE (Log Return) ---
print(f"Creating target variable (Log Return {LOOK_FORWARD_CANDLES} lags)...")
df['target_log_return'] = np.log(df['close'].shift(-LOOK_FORWARD_CANDLES) / df['close'])
df.dropna(subset=['target_log_return'], inplace=True) 

# --- 4. Clean NaN in features (ffill/bfill) ---
print(f"Rows before final NaN/inf cleaning: {len(df)}")
feature_list = [f for f in feature_list if f in df.columns] 
df_features_only = df[feature_list].copy()
df_features_only.replace([np.inf, -np.inf], np.nan, inplace=True)
df_features_only.ffill(inplace=True); df_features_only.bfill(inplace=True)
remaining_nan_cols = df_features_only.isnull().sum(); remaining_nan_cols = remaining_nan_cols[remaining_nan_cols > 0]
if not remaining_nan_cols.empty: 
    print(f"❌ CRITICAL ERROR: NaN remained AFTER ffill+bfill:"); print(remaining_nan_cols); exit()
else: print("✅ NaN check: No gaps found.")
df = df.loc[df_features_only.index]

# --- 5. SAVE PREPARED DATA ---
print(f"Saving prepared data to {PROCESSED_DATA_FILE}...")
cols_to_save = ['open', 'high', 'low', 'close', 'volume'] + feature_list + ['target_log_return']
cols_to_save = sorted(list(set(cols_to_save)))
df_to_save = df[cols_to_save].copy()
try:
    os.makedirs(os.path.dirname(PROCESSED_DATA_FILE), exist_ok=True)
    df_to_save.to_parquet(PROCESSED_DATA_FILE)
    print("Data saved.")
except Exception as e_save: print(f"❌ Parquet save error: {e_save}"); exit()


# --- 6. PREPARE DATA FOR MODEL TRAINING ---
print("Executing StandardScaler...")
scaler = StandardScaler()
scaled_features_for_scaler_fit = scaler.fit_transform(df_features_only) 
print("StandardScaler trained.")

print("Creating windows...")
y_data = df['target_log_return'].values 
X, y = [], []
if len(scaled_features_for_scaler_fit) != len(y_data): print("❌ ERROR: X and y length mismatch!"); exit()
current_sequence_length = min(SEQUENCE_LENGTH, len(scaled_features_for_scaler_fit) - 1)
if current_sequence_length < 30: print("❌ Too little data for windows!"); exit()
elif current_sequence_length != SEQUENCE_LENGTH: print(f"⚠️ Window length reduced to {current_sequence_length}.")

print("Creating windows (Standard Scaler)...")
for i in range(current_sequence_length, len(scaled_features_for_scaler_fit)):
    X.append(scaled_features_for_scaler_fit[i-current_sequence_length:i])
    y.append(y_data[i])
if not X: print("❌ ERROR: Failed to create data windows!"); exit()
X, y = np.array(X), np.array(y)
print(f"Training data size: X={X.shape}, y={y.shape}")

# --- 7. BUILD AND TRAIN FINAL MODEL ---
print("Building TCN-Transformer model (Quantile)...")
# --- ❗️ FIX: "Wrapper" for Quantile Loss ---
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
    # Return the function itself (not its result)
    return quantile_loss
# --- END OF FIX ---

def transformer_encoder(inputs, num_heads, ff_dim, dropout=0.1):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    key_dim = max(1, inputs.shape[-1] // num_heads) 
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    attention_output = Dropout(dropout)(attention_output)
    x = Add()([inputs, attention_output]) 
    x_ff = LayerNormalization(epsilon=1e-6)(x)
    x_ff = Dense(ff_dim, activation="relu")(x_ff)
    x_ff = Dropout(dropout)(x_ff)
    x_ff = Dense(inputs.shape[-1])(x_ff)
    x = Add()([x, x_ff]) 
    return x

def build_tcn_transformer_model(input_shape, num_heads, ff_dim, tcn_filters, tcn_kernel_size, tcn_dilations, num_transformer_blocks=1, dropout=0.1, output_size=3):
    """Builds hybrid TCN + Transformer model with N outputs."""
    inputs = Input(shape=input_shape)
    
    # --- TCN Block ---
    x = inputs
    for dilation_rate in tcn_dilations:
        x_conv = Conv1D(filters=tcn_filters, kernel_size=tcn_kernel_size, 
                        dilation_rate=dilation_rate, activation='relu', padding='causal')(x)
        x_conv = Dropout(dropout)(x_conv)
        if x.shape[-1] == tcn_filters:
             x = Add()([x, x_conv])
        else:
             x_res = Conv1D(filters=tcn_filters, kernel_size=1, padding='same')(x)
             x = Add()([x_res, x_conv])
        x = LayerNormalization(epsilon=1e-6)(x)
    
    # --- Transformer Block ---
    transformer_input = x
    for _ in range(num_transformer_blocks):
        transformer_input = transformer_encoder(transformer_input, num_heads, ff_dim, dropout)
    
    # --- Final layers ---
    x = GlobalAveragePooling1D()(transformer_input)
    x = Dropout(0.3)(x); x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x); x = Dense(64, activation='relu')(x)
    outputs = Dense(output_size, activation='linear')(x)
    
    return Model(inputs=inputs, outputs=outputs)

# --- Model compilation ---
input_shape_model = (current_sequence_length, len(feature_list))
if X.shape[1:] != input_shape_model: print(f"❌ ERROR X shape: {X.shape}, expected ... "); exit()

model = build_tcn_transformer_model(
    input_shape=input_shape_model, num_heads=NUM_HEADS, ff_dim=FF_DIM,
    tcn_filters=TCN_FILTERS, tcn_kernel_size=TCN_KERNEL_SIZE, tcn_dilations=TCN_DILATIONS,
    num_transformer_blocks=NUM_TRANSFORMER_BLOCKS,
    output_size=OUTPUT_SIZE
)

# --- FIX: Call "wrapper" ---
losses = [create_quantile_loss(q=q) for q in QUANTILES]
# --- END OF FIX ---

model.compile(optimizer=Adam(learning_rate=0.0001), loss=losses)
model.summary()

print(f"Training Quantile Regression model (15M, {PAIR}) on {X.shape[0]} examples...")
y_train_quantile = np.tile(y, (OUTPUT_SIZE, 1)).T 
print(f"y_train_quantile size: {y_train_quantile.shape}") # (N_samples, 3)

model.fit(X, y_train_quantile, epochs=30, batch_size=32, verbose=1, validation_split=0.1)

# --- 8. SAVE FINAL MODEL AND Scaler ---
print("Saving final model and scaler...")
try:
    model.save(MODEL_FILE, save_format='keras')
    scaler_final = StandardScaler().fit(df_features_only) 
    joblib.dump(scaler_final, SCALER_FILE)
    print(f"Final model saved as {MODEL_FILE}")
    print(f"Final scaler saved as {SCALER_FILE}")
except Exception as e_save_model: print(f"❌ Save error: {e_save_model}")