import requests
import pandas as pd
from datetime import datetime
import json
from pathlib import Path

#we need to: get the SMARD data feed for day ahead electricty spot market prices
#transform raw JSON to time series data
#export into parquet file

#resikve data directory issues
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def fetch_smard_data(filter_id: int, region_id: str, resolution: str, timestamp: int) -> pd.DataFrame:
    """
    Fetches time-series data from the SMARD API
    
    Args:
        filter_id: 4169 for Spot Market Price, 1223 for Actual Generation
        region_id: 'DE' for Germany
        resolution: 'hour' or 'quarterhour'
        timestamp: Unix timestamp in milliseconds for the week to fetch
    """
    #construct the specific API endpoint URL dynamically using the provided fucntion arguments
    url = f"https://www.smard.de/app/chart_data/{filter_id}/{region_id}/{filter_id}_{region_id}_{resolution}_{timestamp}.json"
    
    response = requests.get(url)
    #check if successful
    response.raise_for_status()
    
    data = response.json()["series"]
    
    # convert to DataFrame
    df = pd.DataFrame(data, columns=["timestamp", "value"])
    
    # convert timestamps from milliseconds to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    
    return df

if __name__ == "__main__":
    print("Fetching German electricity spot prices...")
    
    # example: A specific week timestamp required by the API (e.g recent Monday)
    # SMARD requires timestamps aligned to specific weeks ie: 1709506800000 is approx early March 2024.
    sample_timestamp = 1709506800000 
    
    try:
        # fetch the day ahead spot price
        df_prices = fetch_smard_data(
            filter_id=4169, 
            region_id="DE", 
            resolution="hour", 
            timestamp=sample_timestamp
        )
        print("\nFetched sample records:")
        print(df_prices.head())
        
        # save to the data directory as Parquet (efficient storage)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        output_file = DATA_DIR / "spot_prices.parquet"
        
        df_prices.to_parquet(output_file)
        print(f"\nData saved successfully to {output_file}")
        
    except requests.exceptions.HTTPError as e:
        print(f"Failed to fetch data: {e}")