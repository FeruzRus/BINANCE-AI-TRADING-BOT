# bot/strategy_logic.py
import pandas as pd
import numpy as np
from . import config # Import config for ATR multipliers

def calculate_atr_stops(indicators: dict, entry_price: float, is_long: bool) -> tuple:
    """
    Calculates Take Profit and Stop Loss prices based on READY ATR
    from INDICATORS dictionary.
    """
    
    # --- USE ATR (from config) ---
    ATR_COLUMN_NAME = f'ATR_{config.ATR_PERIOD}' # 'ATR_14'
    
    # --- CHANGE: Get ATR from dictionary ---
    atr_val = indicators.get('atr', np.nan) # 'atr' is ATR_14 from model.py
    
    if pd.isna(atr_val):
        print(f"❌ ATR Stops error: Could not find 'atr' (ATR_14) in indicator dictionary.")
        return None, None
    # --- END OF CHANGE ---

    if atr_val <= 0:
        print(f"❌ ATR Stops error: Invalid {ATR_COLUMN_NAME} value ({atr_val}).")
        return None, None

    print(f"TP/SL calculation: Entry={entry_price:.3f}, {ATR_COLUMN_NAME}={atr_val:.3f}") # 3 digits for SOL

    if is_long:
        tp_price = entry_price + atr_val * config.ATR_TP_MULTIPLIER
        sl_price = entry_price - atr_val * config.ATR_SL_MULTIPLIER
    else: # Short
        tp_price = entry_price - atr_val * config.ATR_TP_MULTIPLIER
        sl_price = entry_price + atr_val * config.ATR_SL_MULTIPLIER

    # Round to correct precision (from config)
    price_precision = config.PRICE_PRECISION # 3 for SOL
    tp_price = round(tp_price, price_precision)
    sl_price = round(sl_price, price_precision)

    print(f"Calculated stops: TP={tp_price}, SL={sl_price}")
    return tp_price, sl_price