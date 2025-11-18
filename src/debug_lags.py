import pandas as pd
import numpy as np
from data_loader import load_data
from long_term_model import prepare_monthly_data

def debug_lags():
    print("Loading data...")
    data = load_data()
    df_cons = data['training_consumption']
    df_groups = data['groups']
    
    print("Preparing monthly data...")
    df_monthly = prepare_monthly_data(df_cons, df_groups)
    
    # Ensure datetime and UTC
    df_monthly['measured_at'] = pd.to_datetime(df_monthly['measured_at'], utc=True)
    
    # Pick a group
    gid = df_monthly['group_id'].unique()[0]
    print(f"Debugging Group ID: {gid}")
    
    g_hist = df_monthly[df_monthly['group_id'] == gid].copy()
    g_hist = g_hist.sort_values('measured_at')
    
    print(f"History range: {g_hist['measured_at'].min()} to {g_hist['measured_at'].max()}")
    print(f"History rows: {len(g_hist)}")
    
    # Check for gaps
    expected_dates = pd.date_range(start=g_hist['measured_at'].min(), end=g_hist['measured_at'].max(), freq='MS', tz='UTC')
    print(f"Expected rows: {len(expected_dates)}")
    
    missing_dates = expected_dates.difference(g_hist['measured_at'])
    if len(missing_dates) > 0:
        print(f"Missing dates in history:\n{missing_dates}")
    else:
        print("No missing dates in history.")
        
    # Create future
    future_dates = pd.date_range(start='2024-10-01', end='2025-09-01', freq='MS', tz='UTC')
    df_future = pd.DataFrame({'measured_at': future_dates, 'group_id': gid})
    
    # Combine
    df_combined = pd.concat([g_hist, df_future], ignore_index=True)
    df_combined = df_combined.sort_values('measured_at')
    
    # Calculate Lag 12
    df_combined['lag_12'] = df_combined['consumption'].shift(12)
    
    # Check Oct 2024
    oct24 = df_combined[df_combined['measured_at'] == '2024-10-01 00:00:00+00:00']
    print("\nOct 2024 Row:")
    print(oct24)
    
    # Check Source (Oct 2023)
    oct23 = df_combined[df_combined['measured_at'] == '2023-10-01 00:00:00+00:00']
    print("\nOct 2023 Row (Source for Lag 12):")
    print(oct23)
    
    if oct23.empty:
        print("ERROR: Oct 2023 is missing!")
    elif pd.isna(oct23['consumption'].values[0]):
        print("ERROR: Oct 2023 consumption is NaN!")
    else:
        print(f"Oct 2023 Consumption: {oct23['consumption'].values[0]}")
        print(f"Oct 2024 Lag 12: {oct24['lag_12'].values[0]}")

if __name__ == "__main__":
    debug_lags()
