import pandas as pd
import numpy as np

def format_submission(df, filename):
    """
    Formats the dataframe to the required submission format:
    - Semicolon delimiter
    - Comma decimal separator
    - ISO 8601 timestamps
    - Correct headers
    """
    print(f"Formatting {filename}...")
    
    # Ensure index is measured_at
    if 'measured_at' in df.columns:
        df = df.set_index('measured_at')
    
    # Fill NaNs with 0
    df = df.fillna(0)
    
    # Format timestamps to ISO 8601 with Z suffix
    # The index is DatetimeIndex.
    # We need to convert to string with specific format.
    # e.g. 2024-10-01T00:00:00.000Z
    
    # Function to format datetime
    def iso_format(dt):
        return dt.strftime('%Y-%m-%dT%H:%M:%S.000Z')
    
    # Reset index to format it
    df_reset = df.reset_index()
    df_reset['measured_at'] = df_reset['measured_at'].apply(iso_format)
    
    # Format values: replace dot with comma
    # We need to convert all float columns to strings with comma
    # But we must keep the structure.
    
    # Get group columns (all except measured_at)
    group_cols = [c for c in df_reset.columns if c != 'measured_at']
    
    for col in group_cols:
        # Convert to string with comma decimal
        # Use map to handle potentially non-float types if any
        df_reset[col] = df_reset[col].apply(lambda x: f"{x:.9f}".replace('.', ','))
    
    # Save to CSV
    # sep=';', index=False
    # Ensure directory exists
    import os
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df_reset.to_csv(filename, sep=';', index=False)
    print(f"Saved to {filename}")

if __name__ == "__main__":
    # Load raw forecasts
    try:
        df_hourly = pd.read_csv("predictions/hourly/raw_forecast.csv", index_col='measured_at', parse_dates=True)
        df_monthly = pd.read_csv("predictions/monthly/raw_forecast.csv", index_col='measured_at', parse_dates=True)
        
        format_submission(df_hourly, "predictions/hourly/submission.csv")
        format_submission(df_monthly, "predictions/monthly/submission.csv")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
