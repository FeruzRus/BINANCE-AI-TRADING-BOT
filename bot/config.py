# bot/config.py

import os
import json
from dotenv import load_dotenv
from binance.client import Client # For INTERVAL

# Load environment variables from .env in project root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# --- File paths (NEW NAMES) ---
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_FILE = os.path.join(ROOT_DIR, 'trading_model_binance_15m_patchtst.keras') # <-- NEW NAME (PatchTST)
SCALER_FILE = os.path.join(ROOT_DIR, 'scaler_binance_15m_patchtst.pkl')         # <-- NEW NAME (PatchTST)
PROCESSED_DATA_FILE = os.path.join(ROOT_DIR, 'processed_data_binance_15m_patchtst.parquet') # <-- NEW NAME (PatchTST)
SETTINGS_FILE = os.path.join(ROOT_DIR, 'settings.json')

# --- Binance API ---
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

# --- Telegram Bot ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# --- Trading Settings ---
PAIR = os.getenv("PAIR", "SOLUSDT") # <-- Set to SOLUSDT
# --- 15 MINUTES ---
INTERVAL = Client.KLINE_INTERVAL_15MINUTE
# --- END OF CHANGE ---
USE_TESTNET = os.getenv("USE_TESTNET", "False").lower() in ('true', '1', 't')

# --- Position size and precision settings (for Binance SOL) ---
POSITION_SIZE_PERCENT = float(os.getenv("POSITION_SIZE_PERCENT", "5"))
QUANTITY_PRECISION = int(os.getenv("QUANTITY_PRECISION", "2")) # 2 for SOL
PRICE_PRECISION = int(os.getenv("PRICE_PRECISION", "3")) # 3 for SOL

# --- Model and filter settings (PatchTST) ---
SEQUENCE_LENGTH = 96 # Window 96 * 15m = 24 hours (PatchTST needs more data)
LOOK_FORWARD_CANDLES = 4 # Forecast 1 hour ahead (4 * 15 min)

# --- ❗️ FIX: Added QUANTILES ---
# Quantiles that the model predicts
QUANTILES = [0.10, 0.50, 0.90] 
OUTPUT_SIZE = len(QUANTILES)
# --- END OF FIX ---

# --- Thresholds for QUANTILE (Quantile) signal ---
QUANTILE_LONG_THRESHOLD = 0.0005  # (0.05%)
QUANTILE_SHORT_THRESHOLD = -0.0005 # (-0.05%)

# --- Thresholds for SECONDARY filter (EMA/RSI) ---
RSI_MAX_LONG = 80     # (Weakened, because model is main)
RSI_MIN_SHORT = 20    # (Weakened)
ATR_PERIOD = 14       # (Standard ATR)
ATR_TP_MULTIPLIER = 1.5 
ATR_SL_MULTIPLIER = 0.7 

# --- Local LLM settings (unchanged) ---
OLLAMA_API_URL = "http://localhost:11434/api/generate"; LLM_MODEL_NAME = "mistral"; LLM_CONFIDENCE_THRESHOLD = 0.7

# --- Dynamic settings (unchanged) ---
def load_settings():
    defaults = {"TRADE_AMOUNT": 25.0, "TAKE_PROFIT": 2.5, "STOP_LOSS": 1.5}
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f: settings = json.load(f)
            for key, default_val in defaults.items():
                 if key not in settings: settings[key] = default_val
            return settings
        except Exception as e:
            print(f"⚠️ settings.json error: {e}. Using defaults.")
            return defaults
    else:
        print("settings.json not found. Using defaults."); save_settings(defaults); return defaults

def save_settings(settings):
    try:
        with open(SETTINGS_FILE, 'w') as f: json.dump(settings, f, indent=2)
    except Exception as e: print(f"⚠️ Error saving settings.json: {e}")

settings = load_settings(); TRADE_AMOUNT = settings.get("TRADE_AMOUNT")
TAKE_PROFIT = settings.get("TAKE_PROFIT"); STOP_LOSS = settings.get("STOP_LOSS")

# --- ❗️ NEW "PatchTST" FEATURE LIST (for 15M) ---
FEATURE_LIST = [
    # 1. Basic + VWAP + Taker (CVD proxy)
    'open', 'high', 'low', 'close', 'volume', 
    'taker_buy_ratio', # (Proxy for CVD)
    'vwap',            # (Calculated)
    'log_return',      # (For normalization)
    
    # 2. Technical features
    'EMA_9', 'EMA_21', 'EMA_50', 'EMA_200',
    'RSI_7', 'RSI_14',
    'ATR_14', 'ATR_14_pct', # ATR and ATR%
    'OBV',
    'MACD', 'MACD_h', 'MACD_s',
    'STOCHRSIk', 'STOCHRSId', 
    'BB_width', # Bollinger width
    
    # 3. Market structure
    'funding_rate',    # (API data)
    'is_weekend',      # (Time)
    'session_asia',    # (Time)
    'session_europe',  # (Time)
    'session_usa',     # (Time)
    
    # 4. Time features
    'minute_of_day',
    'hour_of_day',
    'day_of_week',
    'month_of_year',
]