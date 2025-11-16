# bot/telegram_handler.py

from . import config
from .activation import ActivationManager
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler, MessageHandler,
    ContextTypes, ConversationHandler, filters
)
import pandas as pd
import re
import traceback
import numpy as np # Needed for np.nan

# --- MarkdownV2 escaping function ---
def escape_markdown_v2(text: str) -> str:
    """Escapes all reserved characters for MarkdownV2."""
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', str(text))

# --- Constants for ConversationHandler states ---
SET_TRADE_AMOUNT, SET_FALLBACK_TP, SET_FALLBACK_SL, ACTIVATION_KEY_INPUT = range(4)

class TelegramHandler:
    def __init__(self, main_logic_callback, binance_client, get_bot_state_callback, trading_bot=None):
        self.app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()
        self.main_logic_callback = main_logic_callback
        self.binance_client = binance_client # Name binance_client
        self.get_bot_state_callback = get_bot_state_callback
        self.trading_bot = trading_bot  # Reference to TradingBot for setting activation_manager
        self.activation_manager = None  # Will be initialized on first use
        self.setup_handlers()
    
    def _get_activation_manager(self, telegram_user_id: int) -> ActivationManager:
        
        if self.activation_manager is None or self.activation_manager.telegram_user_id != telegram_user_id:
            self.activation_manager = ActivationManager(telegram_user_id=telegram_user_id)
            
            if self.trading_bot:
                self.trading_bot.set_activation_manager(self.activation_manager)
        return self.activation_manager

    def get_main_menu_keyboard(self):
        
        is_running = self.get_bot_state_callback()
        
        if is_running:
            run_stop_button = [InlineKeyboardButton("🔴 Stop robot", callback_data='stop_robot')]
        else:
            run_stop_button = [InlineKeyboardButton("🟢 Start robot", callback_data='start_robot')]

        keyboard = [
            run_stop_button,
            [InlineKeyboardButton("📊 Active trade", callback_data='active_trade')],
            [InlineKeyboardButton("📈 Status", callback_data='status'), InlineKeyboardButton("⚙️ Settings", callback_data='settings')],
            [InlineKeyboardButton("📜 Trade history", callback_data='history_page_0')],
        ]
        return InlineKeyboardMarkup(keyboard)

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        
        telegram_user_id = update.effective_user.id
        telegram_username = update.effective_user.username
        
        # Get activation manager
        activation_manager = self._get_activation_manager(telegram_user_id)
        
        
        if activation_manager.is_activated():
            
            await update.message.reply_text(
                f"Welcome to the bot control panel! (Binance, {config.PAIR}, {config.INTERVAL}, PatchTST)",
                reply_markup=self.get_main_menu_keyboard()
            )
            return ConversationHandler.END
        else:
            # User not activated - request key
            await update.message.reply_text(
                "🔐 *Bot activation required*\n\n"
                "Please enter your activation key that you received when purchasing:\n\n"
                "If you don't have a key, buy it here - https://t.me/binance_ai_trading_script\n\n"
                "Or enter /cancel to cancel",
                parse_mode='Markdown'
            )
            return ACTIVATION_KEY_INPUT

    async def main_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        
        query = update.callback_query
        telegram_user_id = query.from_user.id
        
        # Check activation before showing menu
        activation_manager = self._get_activation_manager(telegram_user_id)
        if not activation_manager.is_activated():
            await query.answer("❌ Activation required. Use /start", show_alert=True)
            return
        
        await query.answer()
        await query.edit_message_text(
            text="Main menu:",
            reply_markup=self.get_main_menu_keyboard()
        )
    
    async def handle_activation_key(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handles activation key input"""
        key_code = update.message.text.strip()
        telegram_user_id = update.effective_user.id
        telegram_username = update.effective_user.username
        
        # Get activation manager
        activation_manager = self._get_activation_manager(telegram_user_id)
        
        # Try to activate key
        result = activation_manager.activate_key(key_code, telegram_username)
        
        if result.get("success", False):
            await update.message.reply_text(
                "✅ *Key successfully activated! Welcome!*\n\n"
                f"Welcome to the bot control panel! (Binance, {config.PAIR}, {config.INTERVAL}, PatchTST)",
                parse_mode='Markdown',
                reply_markup=self.get_main_menu_keyboard()
            )
            return ConversationHandler.END
        else:
            error_msg = result.get("message", "Unknown error")
            await update.message.reply_text(
                f"❌ *Activation error:* {error_msg}\n\n"
                "Please check the key and try again.\n"
                "Or enter /cancel to cancel.",
                parse_mode='Markdown'
            )
            return ACTIVATION_KEY_INPUT
    
    async def cancel_activation(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Cancels activation process"""
        await update.message.reply_text(
            "Activation cancelled. Use /start to try again."
        )
        return ConversationHandler.END

    async def toggle_robot(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Toggles robot state (Start/Stop)."""
        query = update.callback_query
        telegram_user_id = query.from_user.id
        
        # CRITICAL CHECK: Robot doesn't start without activation
        activation_manager = self._get_activation_manager(telegram_user_id)
        if not activation_manager.is_activated():
            await query.answer("❌ Activation required to start robot. Use /start", show_alert=True)
            return
        
        action = query.data # 'start_robot' or 'stop_robot'
        self.main_logic_callback(action)
        
        await query.answer(f"Robot {'started' if action == 'start_robot' else 'stopped'}.")
        try:
            await query.edit_message_reply_markup(reply_markup=self.get_main_menu_keyboard())
        except Exception as e:
             print(f"Button update error: {e}") # Debug

    # --- ADAPTED FOR BINANCE (KeyError) and PRECISION (Precision) ---
    async def show_active_trade(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Shows detailed information about active trade."""
        query = update.callback_query
        telegram_user_id = query.from_user.id
        
        # CRITICAL CHECK: Data not available without activation
        activation_manager = self._get_activation_manager(telegram_user_id)
        if not activation_manager.is_activated():
            await query.answer("❌ Activation required. Use /start", show_alert=True)
            return
        
        await query.answer("Requesting data (Binance)...")
        try:
            position = self.binance_client.get_open_positions()

            if not position:
                await query.edit_message_text("No active trades.", reply_markup=self.get_back_button())
                return

            tp_order, sl_order = self.binance_client.get_tp_sl_orders()

            # --- USE SAFE .get() method for ALL fields ---
            price_precision = config.PRICE_PRECISION # 3 for SOL
            qty_precision = config.QUANTITY_PRECISION # 2 for SOL

            tp_price_val = float(tp_order.get('stopPrice', '0')) if tp_order else 0
            sl_price_val = float(sl_order.get('stopPrice', '0')) if sl_order else 0
            tp_price_str = f"${tp_price_val:.{price_precision}f}" if tp_order else "N/A"
            sl_price_str = f"${sl_price_val:.{price_precision}f}" if sl_order else "N/A"

            entry_price = float(position.get('entryPrice', '0'))
            mark_price = float(position.get('markPrice', '0'))
            pnl = float(position.get('unRealizedProfit', '0'))
            margin = float(position.get('isolatedWallet', '0'))
            pos_size_asset = float(position.get('positionAmt', '0'))
            liquidation_price = float(position.get('liquidationPrice', '0'))
            
            pos_size_usdt = abs(pos_size_asset * mark_price) # Nominal value
            side = "LONG" if pos_size_asset > 0 else "SHORT"
            roi = (pnl / margin * 100) if margin != 0 else 0
            
            # --- Manual leverage calculation (more reliable) ---
            leverage = round(pos_size_usdt / margin) if margin != 0 else 0
            # --- END ---

            asset_symbol = config.PAIR.replace("USDT", "")

            text = (
                f"*Active position: {config.PAIR} ({side})*\n\n"
                f"*Size:* {abs(pos_size_asset):.{qty_precision}f} {asset_symbol} ({pos_size_usdt:.2f} USDT)\n"
                f"*Margin:* {margin:.2f} USDT\n"
                f"*Leverage:* {leverage}x\n\n"
                f"*PNL (ROI %):* `{pnl:+.2f} USDT` (`{roi:+.2f}%`)\n\n"
                f"*Entry price:* ${entry_price:.{price_precision}f}\n"
                f"*Mark price:* ${mark_price:.{price_precision}f}\n"
                f"*Liquidation price:* ${liquidation_price:.{price_precision}f}\n\n"
                f"*Take Profit:* {tp_price_str}\n"
                f"*Stop Loss:* {sl_price_str}"
            )
            keyboard = [ [InlineKeyboardButton("🔄 Refresh", callback_data='active_trade')], [InlineKeyboardButton("✖️ Close position", callback_data='close_trade_confirm')], [InlineKeyboardButton("⬅️ Back", callback_data='main_menu')]]
            await query.edit_message_text(escape_markdown_v2(text), parse_mode='MarkdownV2', reply_markup=InlineKeyboardMarkup(keyboard))

        except Exception as e:
            print(f"❌ Error in show_active_trade (Binance): {e}"); traceback.print_exc(limit=3)
            try: await query.edit_message_text(f"Error: {e}", reply_markup=self.get_back_button())
            except Exception as send_err: print(f"Failed to send error: {send_err}")

    async def close_trade_confirm(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        keyboard = [[InlineKeyboardButton("Yes, close", callback_data='close_trade_execute')], [InlineKeyboardButton("No, cancel", callback_data='active_trade')]]
        await query.edit_message_text("Are you sure you want to close the position at market price?", reply_markup=InlineKeyboardMarkup(keyboard))

    async def close_trade_execute(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.edit_message_text("<i>Closing position (Binance)...</i>", parse_mode='HTML')
        await query.answer()
        success, message = self.binance_client.close_open_position()
        await query.edit_message_text(escape_markdown_v2(message), parse_mode='MarkdownV2', reply_markup=self.get_back_button())

    async def show_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        telegram_user_id = query.from_user.id
        
        # CRITICAL CHECK: Statistics not available without activation
        activation_manager = self._get_activation_manager(telegram_user_id)
        if not activation_manager.is_activated():
            await query.answer("❌ Activation required. Use /start", show_alert=True)
            return
        
        await query.answer("Collecting statistics (Binance)...")
        stats = self.binance_client.get_account_stats()
        text = "Failed to load statistics."
        if stats:
            text = (
                f"*Account statistics (Binance)*\n\n"
                f"Balance: {stats['balance']:.2f} USDT\n"
                f"Total trades (PnL): {stats['total']}\n"
                f"Successful: {stats['successful']}\n"
                f"Unsuccessful: {stats['unsuccessful']}"
            )
        await query.edit_message_text(escape_markdown_v2(text), parse_mode='MarkdownV2', reply_markup=self.get_back_button())

    async def settings_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        settings = config.load_settings()
        text = (
            f"*Current settings:*\n\n"
            f"Trade amount (TRADE_AMOUNT): {settings['TRADE_AMOUNT']} USDT (Info)\n" # Info, because % of balance is used
            f"Fallback Take Profit: {settings['TAKE_PROFIT']}%\n"
            f"Fallback Stop Loss: {settings['STOP_LOSS']}%\n\n"
            f"_(TP/SL are now calculated by ATR)_"
        )
        keyboard = [
            [InlineKeyboardButton("Trade amount (Info)", callback_data='set_amount')],
            [InlineKeyboardButton("Fallback TP %", callback_data='set_fallback_tp')],
            [InlineKeyboardButton("Fallback SL %", callback_data='set_fallback_sl')],
            [InlineKeyboardButton("⬅️ Back", callback_data='main_menu')]
        ]
        await query.edit_message_text(escape_markdown_v2(text), parse_mode='MarkdownV2', reply_markup=InlineKeyboardMarkup(keyboard))
        return ConversationHandler.END

    async def ask_for_new_value(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        context.user_data['setting_type'] = query.data
        
        state_map = {
            'set_amount': SET_TRADE_AMOUNT,
            'set_fallback_tp': SET_FALLBACK_TP,
            'set_fallback_sl': SET_FALLBACK_SL
        }
        text_map = {
            'set_amount': "Enter new Trade amount (TRADE_AMOUNT, for example *25*): \n_(Note: % of balance is currently used, this value is informational)_",
            'set_fallback_tp': "Enter Fallback Take Profit in % (in case of ATR error, for example *2.5*):",
            'set_fallback_sl': "Enter Fallback Stop Loss in % (in case of ATR error, for example *1.5*):"
        }
        
        text = text_map[query.data]
        state = state_map[query.data]
        
        await query.edit_message_text(escape_markdown_v2(text), parse_mode='MarkdownV2')
        return state

    async def save_new_value(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        new_value_str = update.message.text
        setting_type = context.user_data.get('setting_type')
        if not setting_type: return ConversationHandler.END

        try:
            value = float(new_value_str)
            if value <= 0:
                 await update.message.reply_text("Error. Value must be positive.")
                 state_map = {'set_amount': SET_TRADE_AMOUNT, 'set_fallback_tp': SET_FALLBACK_TP, 'set_fallback_sl': SET_FALLBACK_SL}
                 return state_map[setting_type] # Stay in same state

            settings = config.load_settings()
            key_map = {'set_amount': 'TRADE_AMOUNT', 'set_fallback_tp': 'TAKE_PROFIT', 'set_fallback_sl': 'STOP_LOSS'}
            key = key_map[setting_type]

            settings[key] = value
            config.save_settings(settings)
            # Update global config variables
            if key == 'TRADE_AMOUNT': setattr(config, 'TRADE_AMOUNT', value)
            if key == 'TAKE_PROFIT': setattr(config, 'TAKE_PROFIT', value)
            if key == 'STOP_LOSS': setattr(config, 'STOP_LOSS', value)

            await update.message.reply_text(escape_markdown_v2(f"Setting '{key}' updated to *{value}*."), parse_mode='MarkdownV2')
        except ValueError:
            await update.message.reply_text("Error. Enter a number.")
            state_map = {'set_amount': SET_TRADE_AMOUNT, 'set_fallback_tp': SET_FALLBACK_TP, 'set_fallback_sl': SET_FALLBACK_SL}
            return state_map[setting_type]
        except Exception as e:
             print(f"Setting save error: {e}")
             await update.message.reply_text("An error occurred while saving.")

        await self.settings_menu_after_save(update.message.chat_id, context)
        return ConversationHandler.END

    async def settings_menu_after_save(self, chat_id, context):
        settings = config.load_settings()
        text = (f"*Current settings:*\n\nTrade amount (TRADE_AMOUNT): {settings['TRADE_AMOUNT']} USDT (Info)\n"
                f"Fallback TP: {settings['TAKE_PROFIT']}%\nFallback SL: {settings['STOP_LOSS']}%\n"
                f"_(TP/SL are now by ATR)_")
        keyboard = [ [InlineKeyboardButton("Trade amount (Info)", callback_data='set_amount')], [InlineKeyboardButton("Fallback TP %", callback_data='set_fallback_tp')], [InlineKeyboardButton("Fallback SL %", callback_data='set_fallback_sl')], [InlineKeyboardButton("⬅️ Back", callback_data='main_menu')]]
        await context.bot.send_message(chat_id, escape_markdown_v2(text), parse_mode='MarkdownV2', reply_markup=InlineKeyboardMarkup(keyboard))

    async def show_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        page = int(query.data.split('_')[-1]); items_per_page = 10
        await query.answer("Loading history (Binance PnL)...")
        try:
            history = self.binance_client.client.futures_income_history(incomeType='REALIZED_PNL', limit=1000) 
            history.reverse() # Show recent trades first
            
            start_index = page * items_per_page
            end_index = start_index + items_per_page
            page_items = history[start_index:end_index] 
            
            if not page_items and page == 0: await query.edit_message_text("Trade history (PnL) is empty.", reply_markup=self.get_back_button()); return
            elif not page_items: await query.answer("No more records."); return

            text = f"*Trade history (PnL - Page {page + 1}):*\n\n"
            for trade in page_items:
                dt = pd.to_datetime(trade['time'], unit='ms').strftime('%d.%m %H:%M')
                pnl = float(trade.get('income', '0'))
                symbol = trade.get('symbol') # Get symbol from history
                emoji = "📈" if pnl > 0 else "📉"
                text += f"`{dt}`|`{symbol}`|{emoji} *{pnl:+.2f} USDT*\n"

            keyboard = []; nav_buttons = []
            if page > 0: nav_buttons.append(InlineKeyboardButton("<< Back", callback_data=f'history_page_{page-1}'))
            if end_index < len(history): nav_buttons.append(InlineKeyboardButton("Forward >>", callback_data=f'history_page_{page+1}'))
            if nav_buttons: keyboard.append(nav_buttons)
            keyboard.append([InlineKeyboardButton("⬅️ Main menu", callback_data='main_menu')])

            await query.edit_message_text(escape_markdown_v2(text), parse_mode='MarkdownV2', reply_markup=InlineKeyboardMarkup(keyboard))
        except Exception as e:
             print(f"History loading error (Binance): {e}"); traceback.print_exc(limit=2)
             await query.edit_message_text("Failed to load history.", reply_markup=self.get_back_button())

    def get_back_button(self):
        return InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data='main_menu')]])

    async def send_log(self, message):
        try:
             # Send without formatting
             await self.app.bot.send_message(chat_id=config.TELEGRAM_CHAT_ID, text=message)
        except Exception as e:
             print(f"Failed to send log to Telegram: {e}")

    def setup_handlers(self):
        # Activation handler (must be first)
        activation_handler = ConversationHandler(
            entry_points=[CommandHandler("start", self.start)],
            states={
                ACTIVATION_KEY_INPUT: [
                    MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_activation_key)
                ],
            },
            fallbacks=[
                CommandHandler('cancel', self.cancel_activation),
            ],
            conversation_timeout=300  # 5 minutes to enter key
        )
        
        conv_handler = ConversationHandler(
            entry_points=[CallbackQueryHandler(self.ask_for_new_value, pattern='^set_amount$|^set_fallback_tp$|^set_fallback_sl$')],
            states={
                SET_TRADE_AMOUNT: [MessageHandler(filters.TEXT & ~filters.COMMAND, self.save_new_value)],
                SET_FALLBACK_TP: [MessageHandler(filters.TEXT & ~filters.COMMAND, self.save_new_value)],
                SET_FALLBACK_SL: [MessageHandler(filters.TEXT & ~filters.COMMAND, self.save_new_value)],
            },
            fallbacks=[
                 CommandHandler('cancel', self.cancel_settings),
                 CallbackQueryHandler(self.main_menu, pattern='^main_menu$'),
                 CallbackQueryHandler(self.settings_menu, pattern='^settings$')
            ],
            conversation_timeout=120
        )
        
        self.app.add_handler(activation_handler)
        self.app.add_handler(conv_handler)
        self.app.add_handler(CallbackQueryHandler(self.toggle_robot, pattern='^start_robot$|^stop_robot$'))
        self.app.add_handler(CallbackQueryHandler(self.show_active_trade, pattern='^active_trade$'))
        self.app.add_handler(CallbackQueryHandler(self.close_trade_confirm, pattern='^close_trade_confirm$'))
        self.app.add_handler(CallbackQueryHandler(self.close_trade_execute, pattern='^close_trade_execute$'))
        self.app.add_handler(CallbackQueryHandler(self.show_status, pattern='^status$'))
        self.app.add_handler(CallbackQueryHandler(self.settings_menu, pattern='^settings$'))
        self.app.add_handler(CallbackQueryHandler(self.show_history, pattern='^history_page_'))
        self.app.add_handler(CallbackQueryHandler(self.main_menu, pattern='^main_menu$'))
        
    async def cancel_settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Cancels current settings dialog."""
        print("Settings dialog cancelled.")
        if update.message:
            await update.message.reply_text("Settings change cancelled.")
            await self.start(update, context) # Show main menu
        elif update.callback_query:
            await update.callback_query.answer("Cancelled")
            await self.main_menu(update, context) # Return to main menu
        return ConversationHandler.END

    async def start_bot(self):
        print("Telegram bot started...")
        try:
            await self.app.initialize()
            await self.app.start()
            await self.app.updater.start_polling()
        except Exception as e:
             print(f"Critical Telegram bot startup error: {e}")
