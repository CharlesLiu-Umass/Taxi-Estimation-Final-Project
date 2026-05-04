from pathlib import Path
import pandas as pd

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
    df = df.drop(columns=['speed_mph_proxy', 'PULocationID', 'DOLocationID'])
    #Remove nans
    df = df.dropna()
    return df