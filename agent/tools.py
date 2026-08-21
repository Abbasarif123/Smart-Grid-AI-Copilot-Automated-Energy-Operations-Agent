import pandas as pd
from langchain_core.tools import tool
from pathlib import Path
import xgboost as xgb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
#convert func into ai tool, reads parquet file from storage , applies data filter as requested by agents
#computes avg and max prices and finally returns a final language summary

#resolve datapath issues
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "spot_prices.parquet"
MODEL_PATH = Path(__file__).resolve().parent.parent / "ml_pipeline" / "xgboost_price_model.json"
CHROMA_PATH = Path(__file__).resolve().parent.parent / "data" / "chroma_db"

@tool
def query_historical_spot_prices(start_date: str, end_date: str) -> str:
    """
    queries historical electricity spot market prices for germany
    
    Args:
        start_date: start date in YYYY-MM-DD format
        end_date: end date in YYYY-MM-DD format
        
    Returns:
        string summary of the average price and significant spikes
    """
    if not DATA_PATH.exists():
        return f"Error: Data file not found at {DATA_PATH}."

    #read the parquet
    df = pd.read_parquet(DATA_PATH)
    
    # filter by dates and computer the metricsusing pandas
    mask = (df.index >= start_date) & (df.index <= end_date) #boolean mask to filter rows where the DatatimeIndex fails between the start and end date
    filtered_df = df.loc[mask] #mask applied to extract only matchig date range

    
    if filtered_df.empty:
        # specific date range is empty, summarize available data range
        min_date = df.index.min().strftime('%Y-%m-%d')
        max_date = df.index.max().strftime('%Y-%m-%d')
        avg_price = df["value"].mean()
        return (
            f"No exact records found for {start_date} to {end_date}. "
            f"Available dataset spans {min_date} to {max_date} with an average price of {avg_price:.2f} EUR/MWh."
        )
        
    avg_price = filtered_df["value"].mean()
    max_price = filtered_df["value"].max()
    min_price = filtered_df["value"].min()
    
    return f"Period {start_date} to {end_date}: Average price = {avg_price:.2f} EUR/MWh, Min = {min_price:.2f} EUR/MWh, Peak = {max_price:.2f} EUR/MWh."


#give forcasting capabilities

@tool
def run_price_forecast(hours_ahead: int) -> str:
    """
    runs the trained XGBoost ML model to forecast electricity spot prices for the next N hours
    
    Args:
        hours_ahead: the number of future hours to predict (e.g., 12 or 24)
        
    Returns:
        formatted string with the hour-by-hour price forecast
    """

    #safety clause
    if not MODEL_PATH.exists():
        return "Error: ML model file not found. Train the model first."
    
    #load the trained model and the latest data

    #initialise an empty XGBRegressor instance and load the saved tree weights
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    df = pd.read_parquet(DATA_PATH)
    
    predictions = []
    history = df['value'].tolist()
    #get the latest known timestamp in the dataset to step forward from
    last_timestamp = df.index[-1]
    
    # autoregressive forecasting loop
    for i in range(1, hours_ahead + 1): #run till hours_ahead
        next_time = last_timestamp + pd.Timedelta(hours=i)
        
        # reconstruct features for the next hour based on recent history
        lag_1 = history[-1] #most recent value
        lag_24 = history[-24] if len(history) >= 24 else lag_1 #24 hours prior, else go to lag_1
        rolling_24 = sum(history[-24:]) / 24 if len(history) >= 24 else lag_1 #moving average across the prior 24 hours else lag_1
        
        X_next = pd.DataFrame([{
            'hour': next_time.hour,
            'dayofweek': next_time.dayofweek,
            'lag_1': lag_1,
            'lag_24': lag_24,
            'rolling_mean_24': rolling_24
        }])
        
        # predict the next hour
        #run inference for this single future hour step
        pred_value = model.predict(X_next)[0]
        #format time stamp and forecasted price into markdown list entry
        predictions.append(f"- {next_time.strftime('%Y-%m-%d %H:00')} -> {pred_value:.2f} EUR/MWh")
        
        # append the prediction to the history so it becomes the 'lag_1' for the next loop
        history.append(float(pred_value))
        
    return f"ML Forecast for the next {hours_ahead} hours:\n" + "\n".join(predictions)


#give the LLM legal and policy retrieval capabilities

@tool
def search_regulations(query: str) -> str:
    """
    searches the regulatory knowledge base for rules and guidelines related to the query
    ALWAYS use this before proposing actions to ensure legal compliance
    """
    #verify path
    if not CHROMA_PATH.exists():
        return "Error: Regulatory database not found. Run ingest_regulations.py first."
    #load the sentence transformer model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # load the existing vector store
    vectorstore = Chroma(
        persist_directory=str(CHROMA_PATH), 
        embedding_function=embeddings
    )
    
    # retrieve top 2 most relevant rules based on the agent's semantic search
    results = vectorstore.similarity_search(query, k=2)

    #if nothing is found
    if not results:
        return "No relevant regulations found."

    #extract the metadata source amd the document content
    formatted_results = "\n\n".join([f"Source: {doc.metadata['source']}\nRule: {doc.page_content}" for doc in results])
    return f"Found the following regulatory guidelines:\n\n{formatted_results}"

