# bot/model_predictor.py
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import traceback
from . import config # Import config for file paths

MODEL = None
SCALER = None
FEATURE_LIST_FROM_SCALER = []

def load_model_and_scaler():
    """Loads model and scaler at startup."""
    global MODEL, SCALER, FEATURE_LIST_FROM_SCALER
    if MODEL and SCALER:
        print("Model and scaler already loaded.")
        return True

    print(f"Loading model from: {config.MODEL_FILE}")
    print(f"Loading scaler from: {config.SCALER_FILE}")
    try:
        MODEL = load_model(config.MODEL_FILE)
        SCALER = joblib.load(config.SCALER_FILE)
        if hasattr(SCALER, 'feature_names_in_'):
            FEATURE_LIST_FROM_SCALER = list(SCALER.feature_names_in_)
            print(f"✅ Model and scaler loaded. Expecting {len(FEATURE_LIST_FROM_SCALER)} features.")
            return True
        else:
            print("⚠️ Scaler does not contain feature list. Unable to verify!")
            # Try to use list from config
            if config.FEATURE_LIST:
                 FEATURE_LIST_FROM_SCALER = config.FEATURE_LIST
                 print("Using feature list from config.py.")
                 return True
            else:
                 print("❌ ERROR: No feature list in scaler or config.py.")
                 return False
    except FileNotFoundError:
        print(f"❌ ERROR: Model file ('{config.MODEL_FILE}') or scaler file ('{config.SCALER_FILE}') not found.")
        return False
    except Exception as e:
        print(f"❌ Error loading model/scaler: {e}")
        traceback.print_exc(limit=2)
        return False

def get_model_prediction(df_features_scaled: pd.DataFrame) -> list:
    """
    Takes DataFrame with LAST **SEQUENCE_LENGTH** rows of ALREADY SCALED features.
    Returns list [p_short, p_long].
    """
    if MODEL is None or SCALER is None or not FEATURE_LIST_FROM_SCALER:
        print("❌ Prediction impossible: model/scaler not loaded or no feature list.")
        return [0.5, 0.5] # Neutral prediction

    if len(df_features_scaled) != config.SEQUENCE_LENGTH:
        print(f"❌ Error: Expected {config.SEQUENCE_LENGTH} rows for prediction, got {len(df_features_scaled)}")
        return [0.5, 0.5]

    # Check for required columns (just in case)
    missing = [f for f in FEATURE_LIST_FROM_SCALER if f not in df_features_scaled.columns]
    if missing:
        print(f"❌ Error: Missing features for prediction: {missing}")
        return [0.5, 0.5]

    # Ensure data is in correct order
    data_for_prediction = df_features_scaled[FEATURE_LIST_FROM_SCALER].values

    # Check for NaN before prediction
    if np.isnan(data_for_prediction).any():
        print("❌ Error: NaN detected in data BEFORE prediction!")
        return [0.5, 0.5]

    X_input = np.expand_dims(data_for_prediction, axis=0)

    # Model prediction
    try:
        p_long = MODEL.predict(X_input, verbose=0)[0][0]
        p_short = 1.0 - p_long
        return [p_short, p_long]
    except Exception as predict_err:
        print(f"❌ Error during model prediction: {predict_err}")
        traceback.print_exc(limit=2)
        return [0.5, 0.5]

# Load model on module import
load_model_and_scaler()