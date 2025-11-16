import requests
import json
import time
import base64
from typing import Optional, Dict

class ActivationManager:
    
    def __init__(self, telegram_user_id: Optional[int] = None):
        
        _encrypted_domain = self._decrypt_domain()
        self.api_base_url = f"https://{_encrypted_domain}/api"
        self.telegram_user_id = telegram_user_id
        self._cached_status = None
        self._last_check = 0
        self._check_interval = 3600  
        self._activation_required = True
        self.platform = "binance"  # Platform identifier
        
    
    def _decrypt_domain(self) -> str:
        encrypted = b'YWN0LnByb2RtLnh5eg=='
        try:
            decoded = base64.b64decode(encrypted)
            return decoded.decode('utf-8')
        except Exception:
            return "act.prodm.xyz"        
    
    def is_activated(self, force_check: bool = False) -> bool:
        
        if not self._activation_required:
            return True
        
        if not self.telegram_user_id:
            return False
        
        
        current_time = time.time()
        if force_check or (current_time - self._last_check > self._check_interval):
            try:
                result = self._check_server()
                if result is not None:
                    self._cached_status = result
                    self._last_check = current_time
                
                return self._cached_status if self._cached_status is not None else False
            except Exception as e:
                print(f"⚠️ Activation check error: {e}")
                
                return self._cached_status if self._cached_status is not None else False
        
        return self._cached_status if self._cached_status is not None else False
    
    def _check_server(self) -> Optional[bool]:
        
        try:
            url = f"{self.api_base_url}/check_activation.php"
            payload = {
                "telegram_user_id": self.telegram_user_id,
                "platform": self.platform
            }
            response = requests.post(url, json=payload, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            if data.get("success") and data.get("activated"):
                return True
            return False
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Activation server connection error: {e}")
            return None
        except Exception as e:
            print(f"⚠️ Unexpected activation check error: {e}")
            return None
    
    def activate_key(self, key_code: str, telegram_username: Optional[str] = None) -> Dict:
        
        if not self.telegram_user_id:
            return {"success": False, "message": "Telegram user ID not set"}
        
        try:
            url = f"{self.api_base_url}/activate_key.php"
            payload = {
                "telegram_user_id": self.telegram_user_id,
                "key_code": key_code.strip().upper(),
                "telegram_username": telegram_username,
                "platform": self.platform
            }
            response = requests.post(url, json=payload, timeout=5)
            response.raise_for_status()
            result = response.json()
            
            
            if result.get("success"):
                self._cached_status = True
                self._last_check = time.time()
            
            return result
        except requests.exceptions.RequestException as e:
            error_msg = f"Connection error: {str(e)}"
            print(f"⚠️ {error_msg}")
            return {"success": False, "message": error_msg}
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(f"⚠️ {error_msg}")
            return {"success": False, "message": error_msg}
    
    def require_activation(self):
        
        def decorator(func):
            def wrapper(*args, **kwargs):
                if not self.is_activated():
                    raise RuntimeError(
                        f"❌ CRITICAL ERROR: Function '{func.__name__}' requires activation. "
                        "Use /start to activate."
                    )
                return func(*args, **kwargs)
            return wrapper
        return decorator