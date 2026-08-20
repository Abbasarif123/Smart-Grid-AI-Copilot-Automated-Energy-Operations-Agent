import pandas as pd
from langchain_core.tools import tool
from pathlib import Path
#convert func into ai tool, reads parquet file from storage, applies data filter as requested by agent, computer avg price
#and max price, then finally returns a natural language summary

@tool

def query_historical_spot_prices(start_data: str, end_date: str) -> str:
    """
    queries historical electricity spot market prices for germany

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
    Returns:
        A string summary of the average price and significant spikes

    """

    data_path = Path("../data/spot_prices.parquet")
    if not data_path.exists():
        return "Error: Data file not found"

    #read the parquet
    df = pd.read_parquet(data_path)

    #filter by dates and computer the metrics using Pandas
    mask = (df.index >= start_date) & (df.index <= end_date) #boolean mask to filter rows where the DatatimeIndex falls between the startdate and enddate
    filtered_df = df.loc[mask] #mask applied to extract only the matching date range
    if filtered_df.empty:
        return "No data available for the specified date range."
        
    avg_price = filtered_df["value"].mean()
    max_price = filtered_df["value"].max()
    
    return f"Average price: {avg_price:.2f} EUR/MWh. Peak price: {max_price:.2f} EUR/MWh."