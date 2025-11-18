import requests
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

def check_pagination():
    api_key = os.getenv("FINGRID_API_KEY")
    url = "https://data.fingrid.fi/api/datasets/75/data"
    headers = {'x-api-key': api_key}
    
    # Request a small range or just one page
    start_str = "2024-01-01T00:00:00Z"
    end_str = "2024-01-02T00:00:00Z"
    
    params = {
        'startTime': start_str,
        'endTime': end_str,
        'page': 1,
        'pageSize': 10 # Small page size to force pagination if needed
    }
    
    print(f"Requesting {url} with params {params}")
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 200:
        data = response.json()
        print("Keys:", data.keys())
        if 'pagination' in data:
            print("Pagination:", data['pagination'])
        if 'data' in data:
            print(f"Data count: {len(data['data'])}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    check_pagination()
