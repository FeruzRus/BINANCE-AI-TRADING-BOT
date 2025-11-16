# bot/main.py

import asyncio
import requests
import pandas as pd
import numpy as np
import traceback
from datetime import datetime, timezone, timedelta
from binance.client import Client 

# Import bot modules
from . import config # Common settings
from .binance_client import BinanceClient
from .telegram_handler import TelegramHandler
from .activation import ActivationManager
from . import model as model_pipeline 
from . import telegram_handler 

from . import filter_logic # Module for filter
from . import strategy_logic # Module for TP/SL
from . import local_llm # Module for LLM

QUANTILE_LONG_THRESHOLD = config.QUANTILE_LONG_THRESHOLD
QUANTILE_SHORT_THRESHOLD = config.QUANTILE_SHORT_THRESHOLD

class TradingBot:
    def __init__(self):
        self.is_running = False
        self.in_position = False
        self.main_loop = asyncio.get_event_loop()
        self.current_trade_open_time = None # Timestamp in ms
        self.activation_manager = None  # Will be set on first activation
        self.binance = BinanceClient() 
        self.telegram = TelegramHandler(
            main_logic_callback=self.handle_command,
            binance_client=self.binance, 
            get_bot_state_callback=lambda: self.is_running,
            trading_bot=self  # Pass reference to self for setting activation_manager
        )
        # model_pipeline (model.py) loads itself on import
        if model_pipeline.MODEL is None or model_pipeline.SCALER is None:
             print(f"❌ Critical error: Failed to load model/scaler for {config.PAIR} (PatchTST).")
             # (Can add emergency stop)
    
    def set_activation_manager(self, activation_manager):
        
        self.activation_manager = activation_manager
        self.binance.set_activation_manager(activation_manager)

    def handle_command(self, command):
        if command == 'start_robot':
            if not self.is_running:
                self.is_running = True
                asyncio.run_coroutine_threadsafe(self.telegram.send_log("🟢 Robot started! Starting analysis..."), self.main_loop)
        elif command == 'stop_robot':
            if self.is_running:
                self.is_running = False
                asyncio.run_coroutine_threadsafe(self.telegram.send_log("🔴 Robot stopped by command."), self.main_loop)

    async def run_trading_cycle(self):
        if not self.is_running: return
        
        
        if not self.activation_manager or not self.activation_manager.is_activated():
            print("⚠️ Trading cycle skipped: activation required")
            await self.telegram.send_log("⚠️ Activation required for bot operation. Use /start")
            return

        print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] Entering run_trading_cycle...")
        current_step = "Start of cycle" # For debugging
        try:
            # 1. Check current position
            current_step = "Checking position"
            print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] Calling get_open_positions...")
            open_position_info = None
            network_error_getting_position = False
            try:
                open_position_info = self.binance.get_open_positions()
            except Exception as pos_err:
                print(f"❌ Error when calling get_open_positions: {pos_err}")
                network_error_getting_position = True
            
            currently_in_position = open_position_info is not None
            print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] get_open_positions completed. (API: {'Position exists' if currently_in_position else 'No position'}, Bot 'thinks': {'In position' if self.in_position else 'Out of position'})")

            # 2. Log PnL (Strict check)
            current_step = "Logging PnL"
            
            if network_error_getting_position:
                print("Skipping PnL check due to network failure. 'in_position' status not changed.")
            
            elif self.in_position and not currently_in_position:
                print("[Closing check] API reported no position. Trying to confirm FRESH PnL...")
                await self.telegram.send_log("🔄 Position was closed? Checking result...")
                
                start_check_time = self.current_trade_open_time if self.current_trade_open_time else 0 
                
                try:
                    income_list = self.binance.client.futures_income_history(
                        symbol=config.PAIR, 
                        incomeType='REALIZED_PNL', 
                        startTime=int(start_check_time),
                        limit=5
                    ) 
                    
                    new_pnl_found = False
                    if income_list and len(income_list) > 0:
                        for income in income_list:
                            if income['time'] >= start_check_time:
                                pnl = float(income['income'])
                                await self.telegram.send_log(f"✅ CONFIRMED. PROFIT: +{pnl:.2f} USDT" if pnl > 0 else f"❌ CONFIRMED. LOSS: {pnl:.2f} USDT")
                                print("CONFIRMED: Fresh PnL found. Considering position closed.")
                                self.in_position = False 
                                self.current_trade_open_time = None 
                                new_pnl_found = True
                                break
                    
                    if not new_pnl_found:
                        await self.telegram.send_log("⚠️ Could not find FRESH PnL. Likely API glitch.")
                        print("ERROR: API said 'no position', but FRESH PnL not found. 'in_position' flag NOT reset.")
                except Exception as pnl_err:
                    await self.telegram.send_log(f"⚠️ PnL retrieval error: {pnl_err}. 'in_position' flag NOT reset.")
                    print(f"PnL ERROR: {pnl_err}.")
            
            elif not self.in_position and currently_in_position:
                 print("Detected open position (synchronization)."); self.in_position = True
                 if self.current_trade_open_time is None:
                      self.current_trade_open_time = int(open_position_info.get('updateTime', datetime.now(timezone.utc).timestamp() * 1000 - 10000))
                      print(f"Set (synchronized) trade open time: {self.current_trade_open_time}")
            
            # 3. If IN POSITION -> exit
            if self.in_position:
                 print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] In position, skipping entry cycle.")
                 return

            # --- Logic for OPENING NEW TRADE ---
            
            # 4. Get Klines
            current_step = "Loading Klines"
            # (SEQUENCE_LENGTH=96) + (EMA200) = ~300. Taking 350.
            required_candles = 1000 
            await self.telegram.send_log(f"Getting quotes ({config.INTERVAL}, {required_candles} candles)...")
            klines = self.binance.get_klines(limit=required_candles)
            if len(klines) < config.SEQUENCE_LENGTH + 200: 
                 await self.telegram.send_log(f"⚠️ Insufficient Klines ({len(klines)}) for 15M analysis (PatchTST)."); return

            # 5. GENERATION, CLEANING, PREDICTION AND INDICATORS (ALL IN ONE PLACE)
            current_step = "Model Pipeline"
            # Pass "raw" klines and CLIENT (for funding rate) to bot/model.py
            prediction, indicators = model_pipeline.get_prediction_with_indicators(klines, self.binance)
            
            if indicators is None:
                 print("❌ Error in 'get_prediction_with_indicators' (see log above).")
                 await self.telegram.send_log("⚠️ Data preparation error (NaN/Scale). Prediction skipped.")
                 return
            
            # 6. Decode QUANTILES
            q10, q50, q90 = prediction[0], prediction[1], prediction[2]
            
            # 7. Apply TECHNICAL FILTER (EMA/RSI)
            current_step = "Technical filter"
            filter_result = filter_logic.apply_signal_filter(indicators) 
            if filter_result.get("error"): await self.telegram.send_log(f"⚠️ Filter error: {filter_result['error']}"); return
            long_confirmed = filter_result.get("long_confirmed", False)
            short_confirmed = filter_result.get("short_confirmed", False)
            
            # --- Send detailed filter log ---
            log_indicators = filter_result.get("indicators", {})
            ema9=log_indicators.get('ema_fast',np.nan); ema_slow=log_indicators.get('ema_slow',np.nan); # EMA_200
            rsi14=log_indicators.get('rsi',np.nan); atr14=log_indicators.get('atr',np.nan);
            volume=log_indicators.get('volume',np.nan); avg_atr=log_indicators.get('avg_atr',np.nan); avg_volume=log_indicators.get('avg_volume',np.nan)
            
            atr_threshold = avg_atr * 0.8; volume_threshold = avg_volume * 0.2
            price_precision = config.PRICE_PRECISION # 3
            atr_precision = price_precision + 1 # 4
            
            detailed_log_msg = f"🔎 Detailed filter (PatchTST):\n"
            detailed_log_msg += f"  Long: {'✅' if long_confirmed else '❌'}\n"
            detailed_log_msg += f"    - EMA9 > EMA200: {'✅' if ema9 > ema_slow else '❌'} ({ema9:.{price_precision}f} > {ema_slow:.{price_precision}f})\n"
            detailed_log_msg += f"    - RSI14 < {config.RSI_MAX_LONG}: {'✅' if rsi14 < config.RSI_MAX_LONG else '❌'} ({rsi14:.2f})\n"
            detailed_log_msg += f"    - ATR > {0.8:.1f}*Avg: {'✅' if atr14 > atr_threshold else '❌'} ({atr14:.{atr_precision}f} > {atr_threshold:.{atr_precision}f})\n"
            detailed_log_msg += f"    - Vol > {0.2:.1f}*Avg: {'✅' if volume > volume_threshold else '❌'} ({volume:.0f} > {volume_threshold:.0f})\n"
            detailed_log_msg += f"  Short: {'✅' if short_confirmed else '❌'}\n"
            detailed_log_msg += f"    - EMA9 < EMA200: {'✅' if ema9 < ema_slow else '❌'} ({ema9:.{price_precision}f} < {ema_slow:.{price_precision}f})\n"
            detailed_log_msg += f"    - RSI14 > {config.RSI_MIN_SHORT}: {'✅' if rsi14 > config.RSI_MIN_SHORT else '❌'} ({rsi14:.2f})\n"
            detailed_log_msg += f"    - ATR > {0.8:.1f}*Avg: {'✅' if atr14 > atr_threshold else '❌'} ({atr14:.{atr_precision}f} > {atr_threshold:.{atr_precision}f})\n"
            detailed_log_msg += f"    - Vol > {0.2:.1f}*Avg: {'✅' if volume > volume_threshold else '❌'} ({volume:.0f} > {volume_threshold:.0f})"
            
            # --- CHANGE: Send understandable forecast and filter ---
            
            # 8. Decision mechanism (NEW LOGIC + LOGS)
            current_step = "Decision mechanism"
            signal = "NONE"
            ai_signal = "HOLD" # Signal from Quantiles
            
            # 1. Interpret QUANTILES (Main signal)
            if q90 > QUANTILE_LONG_THRESHOLD and q10 > 0:
                ai_signal = "LONG"
                print("Signal (Quantile): LONG")
            elif q10 < QUANTILE_SHORT_THRESHOLD and q90 < 0:
                ai_signal = "SHORT"
                print("Signal (Quantile): SHORT")
            else:
                 print("Signal (Quantile): HOLD (Neutral)")
            
            # Send INTERPRETATION of forecast
            await self.telegram.send_log(f"🧠 *AI Forecast (Quantile):* {ai_signal}\n(Q10: {q10:.4f}, Q90: {q90:.4f})")
            
            # Send filter log
            try: await self.telegram.send_log(detailed_log_msg)
            except Exception as log_err: print(f"Failed to send detailed filter log: {log_err}")
            
            # 2. Check FILTER
            if ai_signal == "LONG":
                if long_confirmed:
                    print("...Filter (EMA/RSI) CONFIRMED LONG.")
                    signal = "BUY"
                else:
                    print("...LONG signal CANCELLED by filter (EMA/RSI).")
                    await self.telegram.send_log("🛡️ *Filter:* CANCELLED LONG signal (trend/RSI mismatch).")
                    
            elif ai_signal == "SHORT":
                if short_confirmed:
                    print("...Filter (EMA/RSI) CONFIRMED SHORT.")
                    signal = "SELL"
                else:
                    print("...SHORT signal CANCELLED by filter (EMA/RSI).")
                    await self.telegram.send_log("🛡️ *Filter:* CANCELLED SHORT signal (trend/RSI mismatch).")

            # 3. Final decision
            if signal == "NONE":
                 await self.telegram.send_log("...result: no signal or filter cancelled."); return
            
            await self.telegram.send_log(f"✅ *RESULT:* SIGNAL {signal} CONFIRMED (AI + Filter)")
            # --- END OF CHANGE ---

            # 9. (Optional) Request to LOCAL LLM
            current_step = "Request to LLM"
            use_llm = False
            if use_llm:
                safe_filter_result = {"long_confirmed": bool(long_confirmed), "short_confirmed": bool(short_confirmed)}
                indicators_dict_llm = {k: (float(v) if pd.notna(v) else None) for k, v in indicators.items()}
                model_pred_dict = {"Q10": float(q10), "Q50": float(q50), "Q90": float(q90)}

                llm_market_data = { "pair": config.PAIR, "timeframe": config.INTERVAL, "model_prediction": model_pred_dict, "filter_result": safe_filter_result, "indicators": indicators_dict_llm, "current_signal": "LONG" if signal == "BUY" else "SHORT", "open_position": False }
                
                llm_response = local_llm.get_llm_decision(llm_market_data)
                if llm_response:
                     llm_action = llm_response.get("suggested_action", "HOLD").upper(); llm_confidence = llm_response.get("confidence", 0.0); llm_reason = llm_response.get("reason", "N/A")
                     await self.telegram.send_log(f"🤖 LLM ({config.LLM_MODEL_NAME}): {llm_action} (Conf: {llm_confidence:.0%}). Reason: {llm_reason}")
                     if (signal == "BUY" and llm_action == "LONG" and llm_confidence >= config.LLM_CONFIDENCE_THRESHOLD) or \
                        (signal == "SELL" and llm_action == "SHORT" and llm_confidence >= config.LLM_CONFIDENCE_THRESHOLD):
                         await self.telegram.send_log("👍 LLM confirms signal.")
                     else: await self.telegram.send_log("⚠️ LLM does NOT confirm signal. Trade cancelled."); signal = "NONE"
                else: await self.telegram.send_log("⚠️ Failed to get response from LLM. Using Model+Filter signal.")

            # 10. Open trade
            current_step = "Opening trade"
            if signal != "NONE":
                leverage = self.binance.calculate_leverage(indicators)
                entry_price_approx = indicators.get('close', klines[-1][4]) # 'close' from indicators
                tp_price, sl_price = strategy_logic.calculate_atr_stops(indicators, entry_price_approx, signal == "BUY")
                if tp_price is None or sl_price is None: await self.telegram.send_log("❌ ERROR: Failed to calculate ATR TP/SL."); return

                price_precision = config.PRICE_PRECISION # 3 for SOL
                await self.telegram.send_log(f"Opening {signal} trade (based on % balance) with leverage {leverage}× | TP: ${tp_price:.{price_precision}f}, SL: ${sl_price:.{price_precision}f}")
                order, actual_entry_price, quantity = self.binance.open_order(signal, leverage, tp_price, sl_price)

                if order and actual_entry_price is not None:
                    asset_symbol = config.PAIR.replace("USDT", "")
                    await self.telegram.send_log(f"✅ Trade opened! {'LONG' if signal == 'BUY' else 'SHORT'} | {config.PAIR} | Entry: ${actual_entry_price:.{price_precision}f} | Quantity: {quantity} {asset_symbol}")
                    self.in_position = True
                    self.current_trade_open_time = datetime.now(timezone.utc).timestamp() * 1000
                    print(f"NEW TRADE: Open time recorded: {self.current_trade_open_time}")
                else: await self.telegram.send_log("❌ ERROR: Failed to open trade.")

        # --- RELIABLE ERROR HANDLING ---
        except requests.exceptions.ReadTimeout:
            error_message = "⏳ Binance timeout."; print(error_message);
            try: await self.telegram.send_log(error_message)
            except Exception as tg_e: print(f"TG ERR: {tg_e}")
        except requests.exceptions.ConnectionError as e:
            error_message = f"🔌 Binance network failure: {e}."; print(error_message);
            try: await self.telegram.send_log(error_message)
            except Exception as tg_e: print(f"TG ERR: {tg_e}")
        except Exception as e:
            error_message = f"🔥 Critical error at step '{current_step}': {e}"
            print(f"{error_message}\n--- TRACEBACK ---")
            traceback.print_exc(); print("--- END TRACEBACK ---")
            try: await self.telegram.send_log(f"🔥 Critical error (see console): {type(e).__name__} at '{current_step}'")
            except Exception as tg_err: print(f"TG ERR (critical): {tg_err}")

        print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] End of run_trading_cycle.")


    async def main(self):
        """Main bot loop with wait for next CANDLE (15M)"""
        while True:
            try:
                if self.is_running:
                    now_utc = datetime.now(timezone.utc)
                    
                    # --- Logic to wait for next 15-MINUTE candle ---
                    if config.INTERVAL == Client.KLINE_INTERVAL_15MINUTE:
                        next_minute_block = (now_utc.minute // 15 + 1) * 15
                        if next_minute_block >= 60:
                             next_run = (now_utc + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
                        else:
                             next_run = now_utc.replace(minute=next_minute_block, second=0, microsecond=0)
                        sleep_check_interval = 60 # 1 minute
                        print_interval_name = "15M"
                    else: # Default (1H)
                         next_run = (now_utc + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
                         sleep_check_interval = 60; print_interval_name = "1H"
                    
                    wait_seconds = (next_run - now_utc).total_seconds()
                    wait_seconds += 5 # Small delay (5 sec)
                    
                    print(f"Next {print_interval_name} analysis at {next_run.strftime('%Y-%m-%d %H:%M:%S UTC')}. Waiting {wait_seconds:.0f} sec (status check every {sleep_check_interval} sec)...")
                    
                    while wait_seconds > 0 and self.is_running:
                         sleep_interval = min(sleep_check_interval, wait_seconds) 
                         await asyncio.sleep(sleep_interval)
                         wait_seconds -= sleep_interval 
                         
                    if self.is_running:
                         print(f"{datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}: Starting {print_interval_name} candle analysis...")
                         await self.run_trading_cycle()
                    else:
                         print("Robot stopped during wait."); await asyncio.sleep(5)
                         
                else: # If robot is stopped
                    await asyncio.sleep(5) # Check status every 5 seconds

            except Exception as loop_err:
                 print(f"Critical error in main asyncio loop: {loop_err}")
                 traceback.print_exc()
                 await asyncio.sleep(60) # Pause before restarting loop

async def start_bot():
    bot = TradingBot()
    await asyncio.gather(bot.telegram.start_bot(), bot.main())

if __name__ == "__main__":
    try: asyncio.run(start_bot())
    except KeyboardInterrupt: print("\nBot manually stopped.")
    except Exception as main_e: print(f"Critical startup error: {main_e}"); traceback.print_exc()