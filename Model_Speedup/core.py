from pathlib import Path
import pandas as pd
from tensorflow import keras
import joblib
import tensorflow as tf

#Helper function to identify project directory
#Useful for .ipynb files whose working directory differs from project directory by default
def find_project_root(marker="README.md"):
    """Helper function to identify project directory. 
    Useful for .ipynb files whose directory differs from project directory by default. 
    Uses, README file as a marker of the root directory."""
    path = Path.cwd()
    while path != path.parent:
        if (path / marker).exists():
            return path
        path = path.parent
    path = Path.cwd()
    print(f"Project root not found, using default: {path}")
    return path


def load_data():
    """Load multiple parquet files (entire directory) into pandas dataframe"""
    root = find_project_root()
    data_dir = root / "Data" / "Data_Cleaned"
    df = pd.read_parquet(data_dir)  
    # Remove unneeded columns
    df = df[['passenger_count', 'trip_distance', 'payment_type', 
                'pickup_hour', 'pickup_dayofweek', 
                'pickup_month', 'is_weekend', 'trip_duration_seconds',
    ]]
    #Remove nans
    df = df.dropna()
    return df

def load_model():
    """Load the model"""
    root = find_project_root()
    model_dir = root / "Models" / "taxi_model_NN.keras"
    model = keras.models.load_model(model_dir)
    return model

def load_scaler():
    """Load the scaler mapping that data was trained on"""
    root = find_project_root()
    scaler_dir = root / "Models" / "scaler.pkl"
    scaler = joblib.load(scaler_dir)
    return scaler

def load_TF_model(): 
    "Loads TFLite version of the Model"
    root = find_project_root()
    model_dir = root / "Model_Speedup" / "taxi_model.tflite"
    model = tf.lite.Interpreter(model_path=model_dir)
    return model