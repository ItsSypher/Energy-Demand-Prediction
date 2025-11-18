import os
import pandas as pd
from dotenv import load_dotenv
from external_data import get_fingrid_data_real, get_fmi_data_real
import datetime

load_dotenv()

def fetch_and_save_data():
    print("Starting external data fetch...")
    
    # Define range: From start of training data (approx 2021) to end of forecast (late 2025)
    # Training data starts 2021-01-01. Forecast ends 2025-09-30.
    start_date = pd.Timestamp("2021-01-01", tz="UTC")
    end_date = pd.Timestamp("2025-10-01", tz="UTC") # Buffer
    
    output_dir = "data/external"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Fingrid Data
    print("\n--- Fetching Fingrid Data ---")
    fingrid_file = os.path.join(output_dir, "fingrid_wind.csv")
    try:
        df_fingrid = get_fingrid_data_real(start_date, end_date)
        if df_fingrid is not None and not df_fingrid.empty:
            df_fingrid.to_csv(fingrid_file, index=False)
            print(f"Saved Fingrid data to {fingrid_file} ({len(df_fingrid)} rows)")
        else:
            print("Failed to fetch Fingrid data.")
    except Exception as e:
        print(f"Error fetching Fingrid data: {e}")

    # 2. FMI Data
    print("\n--- Fetching FMI Data ---")
    fmi_file = os.path.join(output_dir, "fmi_weather.csv")
    if os.path.exists(fmi_file):
        print(f"FMI data already exists at {fmi_file}. Skipping.")
    else:
        try:
            df_fmi = get_fmi_data_real(start_date, end_date)
            if df_fmi is not None and not df_fmi.empty:
                df_fmi.to_csv(fmi_file, index=False)
                print(f"Saved FMI data to {fmi_file} ({len(df_fmi)} rows)")
            else:
                print("Failed to fetch FMI data.")
        except Exception as e:
            print(f"Error fetching FMI data: {e}")

if __name__ == "__main__":
    fetch_and_save_data()
