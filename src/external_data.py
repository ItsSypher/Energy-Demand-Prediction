import pandas as pd
import numpy as np
import requests
import os
from dotenv import load_dotenv
from fmiopendata.wfs import download_stored_query
import datetime

load_dotenv()

def get_fingrid_data_real(start_date, end_date):
    """
    Fetches real data from Fingrid API.
    Variables:
    - 75: Wind Power Production (MW)
    """
    api_key = os.getenv("FINGRID_API_KEY")
    if not api_key:
        print("FINGRID_API_KEY not found in .env. Using simulation.")
        return None

    print("Fetching data from Fingrid API...")
    
    # Correct URL for Fingrid Open Data API (New structure)
    url = "https://data.fingrid.fi/api/datasets/75/data"
    
    headers = {'x-api-key': api_key}
    
    # Convert dates to ISO 8601
    start_str = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_str = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # Parameters for new endpoint might use camelCase
    params = {
        'startTime': start_str,
        'endTime': end_str,
        'page': 1,
        'pageSize': 5000 # Reduced page size
    }
    
    all_data = []
    import time

    # Initialize last_page
    last_page = None
    
    while True:
        print(f"Fetching Fingrid page {params['page']}...")
        try:
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code == 429:
                print("Rate limit hit (429). Waiting 10 seconds...")
                time.sleep(10)
                continue # Retry same page
                
            response.raise_for_status()
            json_resp = response.json()
            
            if 'data' in json_resp:
                page_data = json_resp['data']
                print(f"  Page {params['page']} items: {len(page_data)}")
                all_data.extend(page_data)
                
                if not page_data:
                    print("No data in this page. Stopping.")
                    break
            else:
                print("Unexpected Fingrid response structure.")
                break
                
            pagination = json_resp.get('pagination', {})
            # print(f"  Pagination: {pagination}") # Reduce noise
            
            if 'lastPage' in pagination:
                last_page = pagination['lastPage']
            
            next_page = pagination.get('nextPage')
            current_page = params['page']
            
            if next_page:
                params['page'] = next_page
            elif last_page and current_page < last_page:
                params['page'] = current_page + 1
            else:
                print("No next page and reached last page (or unknown). Stopping.")
                break
                
            time.sleep(2.5) # Rate limit safe
                
        except Exception as e:
            print(f"Error fetching Fingrid data: {e}")
            return None
            
    if not all_data:
        print("Fingrid API returned empty data.")
        return None
        
    df = pd.DataFrame(all_data)
            
    # Column names might be different in new API. Usually 'startTime', 'value'.
    # Let's check columns if possible, or assume standard.
    # Based on test output keys, it's likely list of dicts.
    
    # Map columns
    if 'startTime' in df.columns:
        df['measured_at'] = pd.to_datetime(df['startTime']).dt.tz_convert('UTC')
    elif 'start_time' in df.columns:
            df['measured_at'] = pd.to_datetime(df['start_time']).dt.tz_convert('UTC')
            
    df['wind_power_production'] = df['value']
    
    # Resample to hourly
    df = df.set_index('measured_at').resample('h')['wind_power_production'].mean().reset_index()
    
    return df[['measured_at', 'wind_power_production']]

def get_fmi_data_real(start_date, end_date):
    """
    Fetches weather data from FMI WFS API using fmiopendata library.
    Location: Helsinki (proxy)
    """
    print("Fetching data from FMI API (via fmiopendata)...")
    
    import time
    from datetime import timedelta
    max_retries = 3
    
    # FMI limit is 168 hours (7 days). We use 6 days to be safe.
    chunk_size = timedelta(days=6)
    current_start = start_date.to_pydatetime()
    final_end = end_date.to_pydatetime()
    
    all_data_list = []
    
    while current_start < final_end:
        current_end = min(current_start + chunk_size, final_end)
        print(f"Fetching FMI data chunk: {current_start} to {current_end}")
        
        chunk_success = False
        for attempt in range(max_retries):
            try:
                # FMI expects strict ISO format, sometimes issues with +00:00
                # Let's use Z format
                start_str = current_start.strftime('%Y-%m-%dT%H:%M:%SZ')
                end_str = current_end.strftime('%Y-%m-%dT%H:%M:%SZ')
                
                # FMI WFS Query
                # stored query: fmi::observations::weather::multipointcoverage
                # place: Helsinki
                obs = download_stored_query("fmi::observations::weather::multipointcoverage",
                                            args=["place=Helsinki",
                                                  f"starttime={start_str}",
                                                  f"endtime={end_str}"])
                
                if not obs.data:
                    print(f"No data for chunk {start_str} - {end_str}")
                    chunk_success = True # Treat as success (empty) to continue
                    break
        
                # Parse the result
                # Structure: timestamp -> {station_name: {variable: value}}
                
                for ts, stations_data in obs.data.items():
                    for station_name, variables in stations_data.items():
                        # variables is dict: variable_name -> value
                        # Variable names can be 'Air temperature', 't2m', etc. depending on library version/mapping
                        
                        # Helper to find value case-insensitively or by partial match
                        def get_val(vars_dict, keys):
                            for k in keys:
                                if k in vars_dict:
                                    return vars_dict[k].get('value', np.nan)
                            # Try searching keys
                            for k in vars_dict:
                                for key in keys:
                                    if key.lower() in k.lower():
                                        return vars_dict[k].get('value', np.nan)
                            return np.nan
        
                        temp = get_val(variables, ['t2m', 'Air temperature', 'Temperature'])
                        wind = get_val(variables, ['ws_10min', 'Wind speed', 'Wind'])
                        
                        all_data_list.append({
                            'measured_at': ts,
                            'temperature': temp,
                            'wind_speed': wind
                        })
                        # Take first station (e.g. Kaisaniemi)
                        break
                
                chunk_success = True
                break # Success, move to next chunk
                
            except Exception as e:
                print(f"Error fetching FMI data chunk (Attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2) # Wait before retry
        
        if not chunk_success:
            print("Failed to fetch chunk after retries. Continuing with partial data.")
            
        # Move to next chunk
        current_start = current_end
        
    if not all_data_list:
        return None
        
    df = pd.DataFrame(all_data_list)
    if df.empty:
        return None
        
    df['measured_at'] = pd.to_datetime(df['measured_at'], utc=True)
    
    # Resample to hourly mean
    df = df.set_index('measured_at').resample('h').mean().reset_index()
    
    return df

def get_weather_data(date_range):
    """
    Returns hourly weather data (Temperature, Wind Speed, Solar Radiation).
    Columns: measured_at, temperature, wind_speed, solar_radiation
    """
    print("Fetching/Generating external data...")
    
    # Check for cached file
    cache_file = "data/external/fmi_weather.csv"
    df_real = None
    
    if os.path.exists(cache_file):
        print(f"Loading FMI data from cache: {cache_file}")
        try:
            df_real = pd.read_csv(cache_file)
            df_real['measured_at'] = pd.to_datetime(df_real['measured_at'], utc=True)
            
            # Filter to date_range
            min_date = date_range.min()
            max_date = date_range.max()
            mask = (df_real['measured_at'] >= min_date) & (df_real['measured_at'] <= max_date)
            df_real = df_real[mask]
            
            if df_real.empty:
                print("Cached FMI data is empty for the requested range. Falling back to API/Sim.")
                df_real = None
        except Exception as e:
            print(f"Error reading FMI cache: {e}")
            df_real = None
            
    if df_real is None:
        # Fallback to API if cache missing/invalid
        # Only try API if we have a key (proxy check using Fingrid key for now, or just try)
        # Actually get_fmi_data_real doesn't need a key, but it's slow.
        # Let's try API if cache fails.
        start_date = date_range.min()
        end_date = date_range.max()
        df_real = get_fmi_data_real(start_date, end_date)

    if df_real is not None:
        # Merge with date_range to ensure completeness
        df = pd.DataFrame({'measured_at': date_range})
        df['measured_at'] = pd.to_datetime(df['measured_at'], utc=True)
        df = df.merge(df_real, on='measured_at', how='left')
        # Fill missing
        df = df.ffill().bfill()
        
        # Add solar radiation (simulated for now as FMI simple query didn't include it easily)
        # We can simulate it based on time of day/year
        if 'solar_radiation' not in df.columns:
             # Simple solar proxy
            df['hour'] = df['measured_at'].dt.hour
            df['month'] = df['measured_at'].dt.month
            # Peak in summer (6-8), day (10-14)
            df['solar_radiation'] = 0.0
            day_mask = (df['hour'] >= 6) & (df['hour'] <= 20)
            # Simple curve
            df.loc[day_mask, 'solar_radiation'] = np.sin((df.loc[day_mask, 'hour'] - 6) * np.pi / 14) * 500
            # Seasonality
            df['solar_radiation'] *= (1 - np.abs(df['month'] - 6.5) / 6.5)
            df['solar_radiation'] = df['solar_radiation'].clip(lower=0)
            df = df.drop(columns=['hour', 'month'])
            
        return df
        
    print("Generating synthetic Finnish weather data (Fallback)...")
    # ... (existing simulation code) ...
    df = pd.DataFrame({'measured_at': date_range})
    df['measured_at'] = pd.to_datetime(df['measured_at'], utc=True)
    
    np.random.seed(42)
    n = len(df)
    
    # Temperature
    df['month'] = df['measured_at'].dt.month
    df['hour'] = df['measured_at'].dt.hour
    
    # Base temp by month (approx Finland)
    monthly_temps = {1:-5, 2:-6, 3:-2, 4:4, 5:10, 6:15, 7:17, 8:15, 9:10, 10:5, 11:0, 12:-3}
    df['base_temp'] = df['month'].map(monthly_temps)
    
    # Daily variation
    df['daily_var'] = -3 * np.cos(2 * np.pi * df['hour'] / 24)
    
    df['temperature'] = df['base_temp'] + df['daily_var'] + np.random.normal(0, 3, n)
    
    # Wind Speed (Weibull-ish)
    df['wind_speed'] = np.random.weibull(2, n) * 5
    
    # Solar Radiation
    df['solar_radiation'] = 0.0
    day_mask = (df['hour'] >= 6) & (df['hour'] <= 20)
    df.loc[day_mask, 'solar_radiation'] = np.sin((df.loc[day_mask, 'hour'] - 6) * np.pi / 14) * 500
    df['solar_radiation'] *= (1 - np.abs(df['month'] - 6.5) / 6.5)
    df['solar_radiation'] = df['solar_radiation'].clip(lower=0)
    
    return df[['measured_at', 'temperature', 'wind_speed', 'solar_radiation']]

def get_grid_data(date_range):
    """
    Returns hourly grid data (Wind Power Production).
    Columns: measured_at, wind_power_production
    """
    # Check for cached file
    cache_file = "data/external/fingrid_wind.csv"
    df_real = None
    
    if os.path.exists(cache_file):
        print(f"Loading Fingrid data from cache: {cache_file}")
        try:
            df_real = pd.read_csv(cache_file)
            df_real['measured_at'] = pd.to_datetime(df_real['measured_at'], utc=True)
            
            # Filter
            min_date = date_range.min()
            max_date = date_range.max()
            mask = (df_real['measured_at'] >= min_date) & (df_real['measured_at'] <= max_date)
            df_real = df_real[mask]
            
            if df_real.empty:
                print("Cached Fingrid data is empty for range. Falling back.")
                df_real = None
        except Exception as e:
            print(f"Error reading Fingrid cache: {e}")
            df_real = None

    if df_real is None:
        start_date = date_range.min()
        end_date = date_range.max()
        df_real = get_fingrid_data_real(start_date, end_date)

    if df_real is not None:
        df = pd.DataFrame({'measured_at': date_range})
        df['measured_at'] = pd.to_datetime(df['measured_at'], utc=True)
        df = df.merge(df_real, on='measured_at', how='left')
        df = df.ffill().bfill()
        return df

    print("Generating synthetic Grid data (Fallback)...")
    df = pd.DataFrame({'measured_at': date_range})
    df['measured_at'] = pd.to_datetime(df['measured_at'], utc=True)
    
    np.random.seed(42)
    # Wind power correlates with wind speed (cubed roughly)
    # We don't have wind speed here easily unless we pass it.
    # Just random walk + noise
    
    df['wind_power_production'] = np.random.normal(2000, 1000, len(df))
    df['wind_power_production'] = df['wind_power_production'].clip(lower=0)
    
    return df[['measured_at', 'wind_power_production']]

def get_external_data(start_date, end_date):
    """
    Main function to get all external data merged.
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='h', tz='UTC')
    
    df_weather = get_weather_data(dates)
    df_grid = get_grid_data(dates)
    
    df_external = pd.merge(df_weather, df_grid, on='measured_at')
    
    return df_external

if __name__ == "__main__":
    # Test
    df = get_external_data('2024-01-01', '2024-01-05')
    print(df.head())
    print(df.describe())
