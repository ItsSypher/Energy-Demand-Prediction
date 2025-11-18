import os
import pandas as pd
import requests
from dotenv import load_dotenv
from external_data import get_fingrid_data_real, get_fmi_data_real
import datetime

load_dotenv()

def test_fingrid():
    print("\n--- Testing Fingrid API ---")
    api_key = os.getenv("FINGRID_API_KEY")
    if not api_key:
        print("SKIPPING: FINGRID_API_KEY not found.")
        return

    start_date = pd.Timestamp("2024-09-01", tz="UTC")
    end_date = pd.Timestamp("2024-09-02", tz="UTC")
    
    start_str = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_str = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    headers = {'x-api-key': api_key}
    
    # List of potential endpoints to try
    endpoints = [
        f"https://data.fingrid.fi/api/v1/variable/75/events/json",
        f"https://data.fingrid.fi/api/datasets/75/data", # Some APIs use datasets/ID/data
        f"https://api.fingrid.fi/v1/variable/75/events/json", # Original (DNS error likely)
    ]
    
    for url in endpoints:
        print(f"Trying URL: {url}")
        params = {'start_time': start_str, 'end_time': end_str}
        if "datasets" in url:
             params = {'startTime': start_str, 'endTime': end_str} # CamelCase check
             
        try:
            response = requests.get(url, headers=headers, params=params)
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                print("SUCCESS!")
                data = response.json()
                # Print first item to see structure
                if isinstance(data, list) and len(data) > 0:
                    print(f"Sample: {data[0]}")
                elif isinstance(data, dict):
                    print(f"Keys: {data.keys()}")
                break
            else:
                print(f"Response: {response.text[:100]}")
        except Exception as e:
            print(f"Error: {e}")

def test_fmi():
    print("\n--- Testing FMI API ---")
    # We haven't implemented the library usage yet in external_data.py, 
    # so this tests the current (broken/stubbed) implementation.
    
    start_date = pd.Timestamp("2024-09-01", tz="UTC")
    end_date = pd.Timestamp("2024-09-02", tz="UTC")
    
    try:
        # Import here to ensure we use the latest code if we updated it (we haven't updated external_data.py yet for FMI fix)
        from external_data import get_fmi_data_real
        df = get_fmi_data_real(start_date, end_date)
        if df is not None and not df.empty:
            print("SUCCESS: FMI data fetched.")
            print(df.head())
        else:
            print("FAILURE: FMI returned None or empty.")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    test_fingrid()
    test_fmi()
