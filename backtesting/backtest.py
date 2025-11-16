# backtesting/backtest.py

import backtrader as bt
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from sklearn.preprocessing import StandardScaler # Needed for Scaler
import os
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import traceback

# Import modules for feature generation
from training.feature_engineering import generate_features
from training.data_loader import _get_interval_ms # For calculating close_time (if needed)

# --- SETTINGS ---
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

PAIR = os.getenv("PAIR", "SOLUSDT")
# --- CHANGE: File paths "PatchTST" ---
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
PROCESSED_DATA_FILE = os.path.join(ROOT_DIR, 'processed_data_binance_15m_patchtst.parquet')
MODEL_FILE = os.path.join(ROOT_DIR, 'trading_model_binance_15m_patchtst.keras')
SCALER_FILE = os.path.join(ROOT_DIR, 'scaler_binance_15m_patchtst.pkl')
SEQUENCE_LENGTH = 96 # Must match train_model.py (PatchTST)
ATR_PERIOD = 14      # Use standard ATR 14

# --- Strategy parameters (PatchTST) ---
QUANTILE_LONG_THRESHOLD = float(os.getenv("QUANTILE_LONG_THRESHOLD", "0.0005")) # 0.05%
QUANTILE_SHORT_THRESHOLD = float(os.getenv("QUANTILE_SHORT_THRESHOLD", "-0.0005")) # -0.05%
RSI_MAX_LONG = 80
RSI_MIN_SHORT = 20
ATR_TP_MULTIPLIER = 2.5
ATR_SL_MULTIPLIER = 1.5
PRICE_PRECISION = 3    # Precision for SOL

# --- Backtest parameters ---
INITIAL_CASH = 10000.0
POSITION_SIZE_PERCENT = 90
COMMISSION_RATE = 0.0004   # 0.04% (Binance Taker)
SLIPPAGE_PERCENT = 0.0002  # 0.02%

# --- 1. Load model and scaler ---
print("Loading PatchTST model and scaler (SOL)...")
model = None; scaler = None; feature_list = []
try:
    model = load_model(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    if hasattr(scaler, 'feature_names_in_'):
        feature_list = list(scaler.feature_names_in_)
        print(f"Model expects {len(feature_list)} features.")
    else: print("❌ ERROR: Scaler does not contain feature list."); exit()
except FileNotFoundError: print(f"❌ ERROR: PatchTST model/scaler files not found. Run train_model.py!"); exit()
except Exception as e: print(f"❌ Loading error: {e}"); traceback.print_exc(limit=2); exit()

# --- 2. Define Backtrader strategy ---
class CnnTransformerStrategy(bt.Strategy):
    params = (
        ('seq_len', SEQUENCE_LENGTH),
        ('feature_list', feature_list),
        ('atr_period', ATR_PERIOD),
        ('atr_tp_mult', ATR_TP_MULTIPLIER),
        ('atr_sl_mult', ATR_SL_MULTIPLIER),
        ('rsi_max_long', RSI_MAX_LONG),
        ('rsi_min_short', RSI_MIN_SHORT),
        ('price_precision', PRICE_PRECISION),
        # New thresholds
        ('q_long_thresh', QUANTILE_LONG_THRESHOLD),
        ('q_short_thresh', QUANTILE_SHORT_THRESHOLD),
    )

    def __init__(self):
        self.model = model
        self.scaler = scaler
        # --- Indicators (PatchTST) ---
        self.ema_fast = bt.indicators.EMA(self.data.close, period=9)
        self.ema_slow = bt.indicators.EMA(self.data.close, period=200) # Slow
        self.rsi_slow = bt.indicators.RSI(self.data.close, period=14) # RSI 14
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period) # ATR 14
        # --- Use MEDIAN for Avg ---
        self.volume_avg = bt.indicators.Percentile(self.data.volume, period=20, perc=50.0)
        self.atr_avg = bt.indicators.Percentile(self.atr, period=20, perc=50.0)
        
        self.order = None; self.sl_order = None; self.tp_order = None

        # --- Create DataFrame to store ALL data ---
        # Backtrader cannot store 40+ features as lines,
        # so we will store them in pandas DataFrame
        self.df_history = pd.DataFrame()
        # Get names of all lines that ARE in data feed
        self.data_lines_names = self.data.lines.getlinealiases()
        print(f"Backtrader Data Feed contains {len(self.data_lines_names)} lines.")

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.datetime(0)
        print(f'{dt.strftime("%Y-%m-%d %H:%M:%S")}, {txt}')

    def notify_order(self, order): # ... (code notify_order from "SOL 15M Points (Simple)" - unchanged) ...
    def notify_trade(self, trade): # ... (code notify_trade from "SOL 15M Points (Simple)" - unchanged) ...
    def set_atr_stops(self, entry_price, is_long, size): # ... (code set_atr_stops from "SOL 15M Points (Simple)" - unchanged) ...
        
    def next(self):
        # --- 1. Update our DataFrame history ---
        current_bar_data = {}
        for line_name in self.data_lines_names:
            try:
                # Get value for current bar
                current_bar_data[line_name] = self.data.lines.getlinealias(line_name)[0]
            except KeyError:
                continue # Skip if line not found
        
        # Add to DataFrame
        # Use to_datetime64() for compatibility with pandas 2.x+
        current_dt = pd.to_datetime(self.data.datetime.datetime(0))
        self.df_history.loc[current_dt] = current_bar_data
        
        # --- Anti-memory (as you requested) ---
        if len(self.df_history) > 1000: # Store ~1000 recent candles
             self.df_history = self.df_history.iloc[-1000:]
             
        # Check bars (longest EMA(200) + window)
        required_bars = max(200, self.p.seq_len) + 1 
        if len(self) < required_bars or len(self.df_history) < required_bars:
             return # Wait for data to accumulate
        if self.position or self.order: return

        # --- 2. Prepare data for model ---
        # Take FULL df_history (with all features)
        # Ensure it's sorted by index
        self.df_history.sort_index(inplace=True)
        
        # --- CHANGE: We DON'T use generate_features() here ---
        # We assume PROCESSED_DATA_FILE already contains ALL features
        # (including ffill/bfill from train_model.py)
        
        # Check df_history for NaN in LAST N rows
        df_window = self.df_history.iloc[-self.p.seq_len:]
        
        # Check for features (from scaler)
        missing_features = [f for f in self.p.feature_list if f not in df_window.columns]
        if missing_features:
            self.log(f"❌ ERROR: Missing features in backtest data: {missing_features}")
            return
            
        data_to_scale = df_window[self.p.feature_list].reindex(columns=self.p.feature_list)
        
        # Check for NaN
        if data_to_scale.isnull().values.any():
            # self.log("Skip (NaN in window)") # Too many logs
            return
        if not np.isfinite(data_to_scale).all().all():
            # self.log("Skip (Inf in window)")
            return
        
        # Scale
        try:
            X_input_scaled = self.scaler.transform(data_to_scale)
            X_input_final = np.expand_dims(X_input_scaled, axis=0)
        except Exception as scale_err: self.log(f"Data preparation error: {scale_err}"); return

        # --- 3. Get model prediction (QUANTILES) ---
        try: 
             q10, q50, q90 = self.model.predict(X_input_final, verbose=0)[0]
        except Exception as pred_err: self.log(f"Prediction error: {pred_err}"); return
        self.log(f'Prediction (Quantile): Q10={q10:.4f}, Q50={q50:.4f}, Q90={q90:.4f}')

        # --- 4. Check FILTER (PatchTST) ---
        ema9 = self.ema_fast[0]; ema_slow = self.ema_slow[0]; # EMA 9/200
        rsi14 = self.rsi_slow[0]; # RSI 14
        atr14 = self.atr[0]; vol = self.data.volume[0]; 
        vol_avg_median = self.volume_avg[0] # Median
        atr_avg_median = self.atr_avg[0]     # Median
        if any(np.isnan(v) for v in [ema9, ema_slow, rsi14, atr14, vol, vol_avg_median, atr_avg_median]): return

        atr_threshold = atr_avg_median * 0.8
        volume_threshold = vol_avg_median * 0.5
        atr_condition = atr14 > atr_threshold
        volume_condition = vol > volume_threshold
        
        long_filter = (ema9 > ema_slow and rsi14 < self.p.rsi_max_long and atr_condition and volume_condition)
        short_filter = (ema9 < ema_slow and rsi14 > self.p.rsi_min_short and atr_condition and volume_condition)

        self.log(f'LONG Filter: {long_filter} (E9>200:{ema9>ema_slow}, RSI<{self.p.rsi_max_long}:{rsi14<self.p.rsi_max_long}, ATR:{atr_condition}, Vol:{volume_condition})')
        self.log(f'SHORT Filter: {short_filter} (E9<200:{ema9<ema_slow}, RSI>{self.p.rsi_min_short}:{rsi14>self.p.rsi_min_short}, ATR:{atr_condition}, Vol:{volume_condition})')

        # --- 5. Decision and entry (NEW LOGIC) ---
        signal = "NONE"
        # 1. Check QUANTILES
        if q90 > self.p.q_long_thresh and q10 > 0: # (Both > 0, 90% chance of > N% growth)
            signal = "BUY"
            self.log("Signal (Quantile): LONG")
            # 2. Check FILTER
            if not long_filter:
                 self.log("...LONG signal cancelled by filter (EMA/RSI).")
                 signal = "NONE"
                 
        elif q10 < self.p.q_short_thresh and q90 < 0: # (Both < 0, 90% chance of < N% drop)
            signal = "SELL"
            self.log("Signal (Quantile): SHORT")
            # 2. Check FILTER
            if not short_filter:
                 self.log("...SHORT signal cancelled by filter (EMA/RSI).")
                 signal = "NONE"

        if signal != "NONE":
            self.log(f'>>> FINAL SIGNAL: {signal} <<<')
            if signal == "BUY": self.log(f'BUY CREATE, Price: {self.data.close[0]:.{self.p.price_precision}f}'); self.order = self.buy()
            elif signal == "SELL": self.log(f'SELL CREATE, Price: {self.data.close[0]:.{self.p.price_precision}f}'); self.order = self.sell()

# --- 3. Load data for Backtrader ---
print(f"Loading data from {PROCESSED_DATA_FILE}...")
data = None 
try:
    df_bt = pd.read_parquet(PROCESSED_DATA_FILE)
    if not isinstance(df_bt.index, pd.DatetimeIndex):
         df_bt.index = pd.to_datetime(df_bt.index)
         if not isinstance(df_bt.index, pd.DatetimeIndex): raise ValueError("Not DatetimeIndex")
    df_bt.rename(columns={'close': 'close', 'open': 'open', 'high': 'high', 'low': 'low', 'volume': 'volume'}, inplace=True, errors='ignore')
    df_bt['openinterest'] = 0
    print(f"Loaded {len(df_bt)} data rows.")

    # --- CREATE DATA CLASS WITH ADDITIONAL LINES ---
    if not feature_list: raise ValueError("Feature list (feature_list) is empty!")
    missing_data_cols = [f for f in feature_list if f not in df_bt.columns]
    if missing_data_cols: 
        print(f"❌ ERROR: Data missing columns: {missing_data_cols}")
        print("--- (This is normal if 'close' was in feature_list but not in df_bt) ---")
        # Ignore 'close' if it's in feature_list but not in df (because it's already in data feed)
        feature_list = [f for f in feature_list if f in df_bt.columns]
        
    print(f"Will actually load {len(feature_list)} feature lines into Backtrader.")

    class PandasDataWithFeatures(bt.feeds.PandasData):
         lines = tuple(feature_list)
         params = tuple((f, -1) for f in feature_list) + \
                  (('open', -1), ('high', -1), ('low', -1), ('close', -1), ('volume', -1), ('openinterest', -1),)

    data = PandasDataWithFeatures(dataname=df_bt)
    print("✅ Data feed for Backtrader created.")

except FileNotFoundError: print(f"❌ File {PROCESSED_DATA_FILE} not found. Run train_model.py!"); exit()
except Exception as e_data: print(f"❌ Backtrader data error: {e_data}"); traceback.print_exc(limit=2); exit()


# --- 4. Setup and run Cerebro ---
print("Setting up Backtrader Cerebro...")
if data is None: print("❌ CRITICAL ERROR: 'data' was not created."); exit()

cerebro = bt.Cerebro(stdstats=False)
cerebro.adddata(data)
cerebro.addstrategy(CnnTransformerStrategy)
cerebro.broker.set_cash(INITIAL_CASH)
cerebro.addsizer(bt.sizers.PercentSizer, percents=POSITION_SIZE_PERCENT)
cerebro.broker.setcommission(commission=COMMISSION_RATE)
cerebro.broker.set_slippage_perc(perc=SLIPPAGE_PERCENT / 100.0)

# Analyzers
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='tradeanalyzer')
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, factor=365, annualize=True, riskfreerate=0.0)
cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')

# --- 5. Run backtest ---
print("Running backtest...")
try:
    results = cerebro.run()
    strat = results[0]
except Exception as run_err: print(f"❌ Backtest error: {run_err}"); traceback.print_exc(); exit()

# --- 6. Output results ---
print(f"\n--- Backtest Results ({config.PAIR} {config.INTERVAL} 'PatchTST') ---")
if not df_bt.empty: print(f"Period: {df_bt.index[0]} - {df_bt.index[-1]}")
print(f"Initial capital: {cerebro.broker.startingcash:.2f}")
print(f"Final capital: {cerebro.broker.getvalue():.2f}")
total_return = (cerebro.broker.getvalue() / cerebro.broker.startingcash - 1) * 100
print(f"Total return: {total_return:.2f}%")

# Use safe .get() access
trade_analysis = strat.analyzers.tradeanalyzer.get_analysis() if hasattr(strat.analyzers, 'tradeanalyzer') else {}
sharpe_analysis = strat.analyzers.sharpe.get_analysis() if hasattr(strat.analyzers, 'sharpe') else {}
drawdown_analysis = strat.analyzers.drawdown.get_analysis() if hasattr(strat.analyzers, 'drawdown') else {}
returns_analysis = strat.analyzers.returns.get_analysis() if hasattr(strat.analyzers, 'returns') else {}
sqn_analysis = strat.analyzers.sqn.get_analysis() if hasattr(strat.analyzers, 'sqn') else {}

print("\n--- Trade Analysis ---")
closed_trades = trade_analysis.get('total', {}).get('closed', 0)
if closed_trades > 0:
    won_trades = trade_analysis.get('won', {}).get('total', 0); lost_trades = trade_analysis.get('lost', {}).get('total', 0)
    print(f"Total closed trades: {closed_trades}"); print(f"Winning trades: {won_trades}"); print(f"Losing trades: {lost_trades}")
    win_rate = won_trades / closed_trades * 100 if closed_trades > 0 else 0.0; print(f"Win Rate: {win_rate:.2f}%")
    avg_pnl_net = trade_analysis.get('pnl', {}).get('net', {}).get('average', 0.0); print(f"Average PnL per trade (Net): {avg_pnl_net:.3f}")
    avg_win = trade_analysis.get('won', {}).get('pnl', {}).get('average', 0.0); avg_loss = trade_analysis.get('lost', {}).get('pnl', {}).get('average', 0.0)
    if won_trades > 0: print(f"Average win: {avg_win:.3f}");
    if lost_trades > 0: print(f"Average loss: {avg_loss:.3f}")
    gross_profit = trade_analysis.get('pnl', {}).get('gross', {}).get('total', 0.0); gross_loss = abs(trade_analysis.get('pnl', {}).get('gross', {}).get('lost', 0.0))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf'); print(f"Profit Factor (Gross): {profit_factor:.2f}")
else: print("No trades or insufficient data for analysis.")

print("\n--- Risk and Return Analysis ---")
sharpe_ratio = sharpe_analysis.get('sharperatio')
if sharpe_ratio is not None: print(f"Sharpe Ratio (annual): {sharpe_ratio:.2f}"); else: print("Failed to calculate Sharpe Ratio.")
max_dd = drawdown_analysis.get('max', {}).get('drawdown', 0.0); max_dd_money = drawdown_analysis.get('max', {}).get('moneydown', 0.0)
print(f"Maximum drawdown: {max_dd:.2f}%"); print(f"Maximum $ drawdown: {max_dd_money:.2f}")
rtot = returns_analysis.get('rtot');
if rtot is not None: print(f"Return over period (Total return): {rtot * 100:.2f}%"); else: print("Return over period (Total return): 0.00% (no data)")
sqn_val = sqn_analysis.get('sqn');
if sqn_val is not None: print(f"System Quality Number (SQN): {sqn_val:.2f}")

# --- 7. Chart ---
try:
    print("\nPlotting chart...")
    figure = cerebro.plot(style='candlestick', barup='green', bardown='red', volup='#43A047', voldown='#E53935', dpi=100)[0][0]
    figure.savefig('backtest_chart_15m_patchtst.png') # <-- New name
    print("Chart saved to 'backtest_chart_15m_patchtst.png'")
except IndexError: print("Failed to plot chart (possibly no data).")
except Exception as e_plot: print(f"Failed to plot chart: {e_plot}")