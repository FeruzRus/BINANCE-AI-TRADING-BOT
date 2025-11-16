# bot/filter_logic.py
import pandas as pd
import numpy as np
from . import config # Import config for RSI thresholds
import traceback

def apply_signal_filter(indicators: dict) -> dict:
    """
    Applies TECHNICAL FILTER (PatchTST)
    (Adapted for EMA 9/200, RSI 14, ATR 14)
    RECEIVES READY INDICATORS from model.py.
    """
    if not indicators:
        return {"long_confirmed": False, "short_confirmed": False, "error": "No indicator data for filter"}

    try:
        # --- Get NEW indicator values FROM DICTIONARY ---
        ema9 = indicators.get('ema_fast', np.nan)    # EMA_9
        ema_slow = indicators.get('ema_slow', np.nan)   # EMA_200
        rsi14 = indicators.get('rsi', np.nan)        # RSI_14
        atr14 = indicators.get('atr', np.nan)         # ATR_14
        volume = indicators.get('volume', np.nan)
        avg_atr = indicators.get('avg_atr', np.nan)    # Median(ATR_14, 20)
        avg_volume = indicators.get('avg_volume', np.nan) # Median(Volume, 20)

        # --- Check for NaN ---
        required_values = [ema9, ema_slow, rsi14, atr14, volume, avg_atr, avg_volume]
        if any(pd.isna(v) for v in required_values):
            print("⚠️ NaN in indicators received by filter (PatchTST):")
            return {"long_confirmed": False, "short_confirmed": False, "error": "NaN in filter data"}

        # --- Filter logic (EMA, RSI, ATR, Vol) ---
        
        # Weakened thresholds (0.8 / 0.2) - can be adjusted
        atr_threshold = avg_atr * 0.8
        volume_threshold = avg_volume * 0.2
        
        atr_condition = atr14 > atr_threshold
        volume_condition = volume > volume_threshold

        # LONG filter: Uptrend (EMA9 > EMA200) + RSI not overbought
        long_filter_passed = (
            ema9 > ema_slow and
            rsi14 < config.RSI_MAX_LONG and  # (e.g., 80)
            atr_condition and
            volume_condition
        )

        # SHORT filter: Downtrend (EMA9 < EMA200) + RSI not oversold
        short_filter_passed = (
            ema9 < ema_slow and
            rsi14 > config.RSI_MIN_SHORT and # (e.g., 20)
            atr_condition and
            volume_condition
        )
        
        # --- Log results (to console) ---
        price_precision = config.PRICE_PRECISION # 3
        atr_precision = price_precision + 1 # 4

        print("--- Filter Results (PatchTST) ---")
        print(f"Long Filter -> EMA9>200:{ema9>ema_slow}({ema9:.{price_precision}f}>{ema_slow:.{price_precision}f}), "
              f"RSI14<{config.RSI_MAX_LONG}:{rsi14<config.RSI_MAX_LONG}({rsi14:.2f}), "
              f"ATR>0.8*avg:{atr_condition}({atr14:.{atr_precision}f}>{atr_threshold:.{atr_precision}f}), "
              f"Vol>0.2*avg:{volume_condition}({volume:.0f}>{volume_threshold:.0f}) "
              f"=> {long_filter_passed}")
        
        print(f"Short Filter -> EMA9<200:{ema9<ema_slow}({ema9:.{price_precision}f}<{ema_slow:.{price_precision}f}), "
              f"RSI14>{config.RSI_MIN_SHORT}:{rsi14>config.RSI_MIN_SHORT}({rsi14:.2f}), "
              f"ATR>0.8*avg:{atr_condition}({atr14:.{atr_precision}f}>{atr_threshold:.{atr_precision}f}), "
              f"Vol>0.2*avg:{volume_condition}({volume:.0f}>{volume_threshold:.0f}) "
              f"=> {short_filter_passed}")
        print("------------------------------------------")

        # Also return indicators so main.py can use them for logs/LLM
        return {
            "long_confirmed": long_filter_passed, 
            "short_confirmed": short_filter_passed,
            "indicators": indicators # Return dictionary with numbers
        }

    except Exception as e:
        print(f"❌ Error applying filter: {e}")
        traceback.print_exc(limit=2)
        return {"long_confirmed": False, "short_confirmed": False, "error": str(e)}