# bot/binance_client.py

import pandas as pd
import pandas_ta as ta
import time
from binance.client import Client
from binance.exceptions import BinanceAPIException
from . import config # Import config to access settings
import traceback # For error output
import math # For rounding

class BinanceClient:
    def __init__(self, activation_manager=None):
        
        self._activation_manager = activation_manager
        
        requests_params = {"timeout": 15}
        self.client = Client(
            config.BINANCE_API_KEY,
            config.BINANCE_API_SECRET,
            testnet=config.USE_TESTNET,
            requests_params=requests_params
        )
        print("✅ Binance Client initialized.")
        self._set_initial_settings()
    
    def set_activation_manager(self, activation_manager):
        
        self._activation_manager = activation_manager
    
    def _check_activation(self):
        
        if self._activation_manager and not self._activation_manager.is_activated():
            raise RuntimeError(
                "❌ CRITICAL ERROR: Activation required to perform operation. "
                "Use /start to activate."
            )

    def _set_initial_settings(self):
        
        try:
            print(f"Attempting to set ISOLATED margin type for {config.PAIR}...")
            self.client.futures_change_margin_type(symbol=config.PAIR, marginType='ISOLATED')
            print("✅ ISOLATED margin type set (or was already set).")
        except BinanceAPIException as e:
            if "No need to change margin type" in str(e):
                print("Margin type is already ISOLATED.")
            else:
                print(f"⚠️ API error when setting margin type: {e}")
        except Exception as e_margin:
            print(f"❌ Unknown error when setting margin: {e_margin}")

    def get_klines(self, limit=300):
        """Gets historical klines (candles) from Binance Futures."""
        self._check_activation() 
        try:
            return self.client.futures_klines(
                symbol=config.PAIR,
                interval=config.INTERVAL, # 15M
                limit=limit
            )
        except BinanceAPIException as e:
            print(f"❌ Binance API error when getting klines: {e}")
            return []
        except Exception as e_klines:
            print(f"❌ Unknown error when getting klines: {e_klines}")
            return []

    # --- NEW FUNCTION: Getting Funding Rate ---
    def get_funding_rate_history(self, limit=100):
        """Gets the LAST N funding rate records."""
        try:
            # Request last 100 records (usually 3 times a day)
            funding_rates = self.client.futures_funding_rate(
                symbol=config.PAIR, 
                limit=limit
            )
            
            if not funding_rates:
                 print("⚠️ Funding Rate not found (get_funding_rate_history).")
                 return pd.DataFrame()
                 
            df_funding = pd.DataFrame(funding_rates)
            df_funding['fundingTime'] = pd.to_datetime(df_funding['fundingTime'], unit='ms')
            df_funding.set_index('fundingTime', inplace=True)
            df_funding['funding_rate'] = pd.to_numeric(df_funding['fundingRate'], errors='coerce')
            
            # Return DataFrame with time index and 'funding_rate' column
            return df_funding[['funding_rate']]
            
        except BinanceAPIException as e:
            print(f"❌ Binance API error (Funding Rate): {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"❌ Unknown error (Funding Rate): {e}")
            return pd.DataFrame()
    # --- END OF NEW FUNCTION ---

    def calculate_leverage(self, indicators: dict) -> int:
        """Calculates leverage (20x-50x) based on ATR from DICTIONARY."""
        
        ATR_COLUMN_NAME = 'atr' # Key 'atr' (ATR_14)
        
        atr_val = indicators.get(ATR_COLUMN_NAME)
        close_price = indicators.get('close') # 'close' should be in the dictionary
        
        if atr_val is None or close_price is None or pd.isna(atr_val) or pd.isna(close_price):
            print(f"⚠️ Insufficient data to calculate leverage (ATR/Close from dictionary), using 20x.")
            return 20 

        if close_price <= 0 or atr_val <= 0:
             print(f"⚠️ Invalid data to calculate leverage (ATR={atr_val}, Close={close_price}), using 20x.")
             return 20

        volatility_percent = (atr_val / close_price) * 100
        print(f"Leverage calculation: ATR={atr_val:.3f}, Close={close_price:.3f}, Volatility={volatility_percent:.2f}%") # 3 digits for SOL

        if volatility_percent > 1.0: leverage = 20
        elif volatility_percent > 0.5: leverage = 35
        else: leverage = 50

        print(f"Selected leverage: {leverage}x")
        return leverage

    def get_balance(self, asset='USDT') -> float:
        """Gets total balance on futures account."""
        self._check_activation()  # CRITICAL CHECK
        try:
            balance_info = self.client.futures_account_balance()
            for balance in balance_info:
                if balance.get('asset') == asset:
                     bal = float(balance.get('balance', '0'))
                     print(f"Balance {asset}: {bal:.2f}")
                     return bal
            print(f"⚠️ Asset {asset} not found in balance."); return 0.0
        except BinanceAPIException as e: print(f"⚠️ API error when getting balance: {e}"); return 0.0
        except Exception as e_bal: print(f"❌ Unknown error getting balance: {e_bal}"); return 0.0

    def get_quantity(self, balance: float, leverage: int, price: float) -> float:
        """Calculates asset quantity for order."""
        if price <= 0 or balance <= 0 or leverage <= 0:
            print("❌ Quantity calculation error: invalid input data."); return 0.0

        notional = balance * leverage * config.POSITION_SIZE_PERCENT / 100
        quantity = notional / price

        rounded_quantity = round(quantity, config.QUANTITY_PRECISION) # 2 for SOL
        print(f"Quantity calculation: Notional={notional:.2f}, Price={price:.3f}, Quantity={quantity:.8f}, Rounded={rounded_quantity}")
        return rounded_quantity

    def open_order(self, side: str, leverage: int, tp_price: float = None, sl_price: float = None):
        """Opens a market order and sets TP/SL."""
        self._check_activation()  # CRITICAL CHECK BEFORE ORDER
        current_step = "Start of open_order"
        try:
            current_step = "Getting Mark Price"
            mark_price_info = self.client.futures_mark_price(symbol=config.PAIR)
            entry_price_approx = float(mark_price_info['markPrice'])

            current_step = "Setting Margin/Leverage"
            print(f"⚙️ Setting leverage {leverage}x for {config.PAIR} (Isolated)...")
            try: self.client.futures_change_margin_type(symbol=config.PAIR, marginType='ISOLATED')
            except BinanceAPIException as e_mt:
                 if "No need to change margin type" in str(e_mt): pass
                 else: print(f"⚠️ Margin type setting error: {e_mt}")
            self.client.futures_change_leverage(symbol=config.PAIR, leverage=leverage)
            print(f"✅ Leverage {leverage}x set.")

            current_step = "Calculating Quantity"
            balance = self.get_balance()
            if balance <= 0: print(f"❌ Error: USDT balance ({balance}) <= 0."); return None, None, None
            quantity = self.get_quantity(balance, leverage, entry_price_approx)
            if quantity <= 0: print(f"❌ Error: Calculated quantity <= 0 ({quantity})."); return None, None, None

            current_step = "Creating Market Order"
            print(f"📊 Opening {side} | Price ~{entry_price_approx:.3f} | Quantity: {quantity} | Leverage: {leverage}x")
            order = self.client.futures_create_order(
                symbol=config.PAIR, side=side, type='MARKET', quantity=quantity
            )

            print("Waiting for order execution..."); time.sleep(2)

            current_step = "Getting Position and Setting TP/SL"
            position_info = self.get_open_positions()
            if position_info:
                actual_entry_price = float(position_info.get('entryPrice', entry_price_approx))
                print(f"✅ Order executed (ID: {order.get('orderId', 'N/A')}). Actual entry price: {actual_entry_price:.3f}")
                if tp_price is not None and sl_price is not None:
                    self.set_tp_sl(side, actual_entry_price, tp_price, sl_price)
                return order, actual_entry_price, quantity
            else:
                print("⚠️ Could not get position information immediately after opening.")
                if tp_price is not None and sl_price is not None:
                     print("Attempting to set TP/SL using mark price.")
                     self.set_tp_sl(side, entry_price_approx, tp_price, sl_price)
                return order, entry_price_approx, quantity

        except BinanceAPIException as e: print(f"❌ Binance API Error at step '{current_step}': {e}"); traceback.print_exc(limit=2); return None, None, None
        except Exception as e: print(f"❌ Unknown error at step '{current_step}': {e}"); traceback.print_exc(limit=2); return None, None, None

    def set_tp_sl(self, original_side: str, entry_price: float, tp_price: float, sl_price: float):
        """Sets Take Profit and Stop Loss orders."""
        is_long = original_side == 'BUY'
        close_side = 'SELL' if is_long else 'BUY'
        price_precision = config.PRICE_PRECISION # 3 for SOL

        try:
            print("Canceling previous TP/SL orders...")
            self.client.futures_cancel_all_open_orders(symbol=config.PAIR)
            time.sleep(0.5)

            # Price validation
            if tp_price <= 0 or sl_price <= 0: raise ValueError("TP/SL prices must be > 0")
            if tp_price == sl_price: raise ValueError("TP and SL prices cannot be the same")
            if is_long and (tp_price <= entry_price or sl_price >= entry_price):
                 print(f"⚠️ Warning: Illogical TP/SL for LONG: TP={tp_price}, SL={sl_price}, Entry={entry_price}")
            if not is_long and (tp_price >= entry_price or sl_price <= entry_price):
                 print(f"⚠️ Warning: Illogical TP/SL for SHORT: TP={tp_price}, SL={sl_price}, Entry={entry_price}")

            tp_price_str = f"{tp_price:.{price_precision}f}"
            sl_price_str = f"{sl_price:.{price_precision}f}"

            # Setting Take Profit
            print(f"Setting TP ({close_side}) @ {tp_price_str}...")
            tp_order = self.client.futures_create_order(
                symbol=config.PAIR, side=close_side, type='TAKE_PROFIT_MARKET',
                stopPrice=tp_price_str, closePosition=True, timeInForce='GTC'
            )
            print(f"✅ TP order set (ID: {tp_order.get('orderId', 'N/A')}).")

            # Setting Stop Loss
            print(f"Setting SL ({close_side}) @ {sl_price_str}...")
            sl_order = self.client.futures_create_order(
                symbol=config.PAIR, side=close_side, type='STOP_MARKET',
                stopPrice=sl_price_str, closePosition=True, timeInForce='GTC'
            )
            print(f"✅ SL order set (ID: {sl_order.get('orderId', 'N/A')}).")
            print(f"🎯 Final stops set: TP={tp_price_str} | 🛑 SL={sl_price_str}")

        except BinanceAPIException as e: print(f"❌ Binance API error when setting TP/SL: {e}"); traceback.print_exc(limit=2)
        except ValueError as ve: print(f"❌ TP/SL validation error: {ve}")
        except Exception as e_tp_sl: print(f"❌ Unknown error when setting TP/SL: {e_tp_sl}"); traceback.print_exc(limit=2)

    def get_open_positions(self):
        """Returns information about open position for pair or None."""
        self._check_activation()  # CRITICAL CHECK
        try:
            positions = self.client.futures_position_information(symbol=config.PAIR, recvWindow=5000)
            if isinstance(positions, list):
                 for pos in positions:
                      if 'positionAmt' in pos and float(pos.get('positionAmt', '0')) != 0:
                           return pos
            elif isinstance(positions, dict) and 'positionAmt' in positions and float(positions.get('positionAmt','0')) != 0:
                 return positions
            return None
        except BinanceAPIException as e:
            print(f"⚠️ API error when getting position: {e}")
            raise e # Pass error so main.py can catch it
        except Exception as e_pos:
            print(f"❌ Unknown error when getting position: {e_pos}")
            raise e_pos # Pass error

    def get_tp_sl_orders(self):
        """Gets OPEN TP and SL orders for current pair."""
        try:
            orders = self.client.futures_get_open_orders(symbol=config.PAIR)
            tp_order = next((o for o in orders if o.get('type') == 'TAKE_PROFIT_MARKET' and o.get('status') == 'NEW'), None)
            sl_order = next((o for o in orders if o.get('type') == 'STOP_MARKET' and o.get('status') == 'NEW'), None)
            return tp_order, sl_order
        except BinanceAPIException as e: print(f"⚠️ Error getting TP/SL orders: {e}"); return None, None
        except Exception as e_get_ord: print(f"❌ Unknown error getting TP/SL orders: {e_get_ord}"); return None, None

    def close_open_position(self):
        """Closes current open position at market price."""
        self._check_activation()  # CRITICAL CHECK
        position = self.get_open_positions()
        if not position: return False, "No open positions."
        try:
            pos_amt = float(position.get('positionAmt','0'))
            if pos_amt == 0: return False, "Position already closed (Amt=0)."
            side = 'BUY' if pos_amt < 0 else 'SELL'; quantity = abs(pos_amt)
            print(f"Closing position {side} | Quantity: {quantity}"); print("Canceling TP/SL before closing...")
            self.client.futures_cancel_all_open_orders(symbol=config.PAIR); time.sleep(1) 
            close_order = self.client.futures_create_order(symbol=config.PAIR, side=side, type='MARKET', quantity=quantity, reduceOnly=True)
            print(f"Close order sent (ID: {close_order.get('orderId', 'N/A')})."); time.sleep(2)
            final_pos = self.get_open_positions()
            if final_pos is None: return True, "✅ Position successfully closed."
            else:
                 print("⚠️ Position still exists... Re-cancelling..."); self.client.futures_cancel_all_open_orders(symbol=config.PAIR); time.sleep(1)
                 final_pos_after_cancel = self.get_open_positions()
                 if final_pos_after_cancel is None: return True, "✅ Position closed after re-cancellation."
                 else: return False, f"⚠️ Unable to confirm closure. Current Amt: {final_pos_after_cancel.get('positionAmt')}"
        except BinanceAPIException as e: print(f"❌ API error when closing: {e}"); return False, f"Binance error: {e}"
        except Exception as e_close: print(f"❌ Unknown error when closing: {e_close}"); return False, f"Unknown error: {e_close}"

    def get_account_stats(self):
        """Collects statistics on balance and trades (PnL)."""
        stats = {"balance": 0.0, "total": 0, "successful": 0, "unsuccessful": 0}
        try:
            balance_info = self.client.futures_account_balance()
            usdt_balance = next((b['balance'] for b in balance_info if b.get('asset') == 'USDT'), '0')
            stats['balance'] = float(usdt_balance)
        except Exception as e_bal: print(f"⚠️ Balance retrieval error for statistics: {e_bal}")
        try:
            trades = self.client.futures_income_history(incomeType='REALIZED_PNL', limit=1000)
            stats['total'] = len(trades)
            stats['successful'] = sum(1 for t in trades if float(t.get('income', '0')) > 0)
            stats['unsuccessful'] = stats['total'] - stats['successful']
            return stats
        except Exception as e_hist:
             print(f"⚠️ PnL history retrieval error for statistics: {e_hist}")
             return stats