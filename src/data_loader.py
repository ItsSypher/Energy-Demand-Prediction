import pandas as pd
import os

DATA_PATH = "Dataset/20251111_JUNCTION_training.xlsx"

def load_data(data_path=DATA_PATH):
    """
    Loads the training data from the Excel file.
    Returns a dictionary containing the dataframes for 'groups', 'training_consumption', and 'training_prices'.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    print(f"Loading data from {data_path}...")
    xls = pd.ExcelFile(data_path)
    
    data = {}
    for sheet_name in ['groups', 'training_consumption', 'training_prices']:
        if sheet_name in xls.sheet_names:
            print(f"Loading sheet: {sheet_name}")
            data[sheet_name] = pd.read_excel(xls, sheet_name=sheet_name)
        else:
            print(f"Warning: Sheet {sheet_name} not found in Excel file.")
            
    return data

def inspect_data(data):
    """
    Prints basic info about the loaded dataframes.
    """
    for name, df in data.items():
        print(f"\n--- {name} ---")
        print(df.info())
        print(df.head())

if __name__ == "__main__":
    # Test loading
    try:
        data = load_data()
        inspect_data(data)
    except Exception as e:
        print(f"Error: {e}")
