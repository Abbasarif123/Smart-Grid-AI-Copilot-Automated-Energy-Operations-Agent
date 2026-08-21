import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from pathlib import Path

#load ingested price data by reading the parquet
#extract patterns, lags and moving averages to give the model predictive context
#train an xgboost model
#computer the mean absolute error on unseen test data to quantify how are off price predictions are
#serialise the trained weights to lightweight json file artifact for deployment or agent tool in



#setups paths dynamically
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_DIR = Path(__file__).resolve().parent

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    creats time-series features from the raw timestamp data
    """
    #create an independent copy to avoid changing original input dataframe
    df = df.copy()
    
    #temporal features
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    
    # lag features(what was the price 1 hour ago? 24 hours ago?)
    df['lag_1'] = df['value'].shift(1) #1 hour prior
    df['lag_24'] = df['value'].shift(24) #24 hour prior
    
    # rolling features(average price over the last 24 hours)
    df['rolling_mean_24'] = df['value'].rolling(window=24).mean()
    
    # drop rows with NaN values caused by shifting
    return df.dropna()



#FEATURE MATRIX-X AND TARGET VECTOR-Y SEPARATION
if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_parquet(DATA_DIR / "spot_prices.parquet")
    
    print("Engineering features...")
    df_features = create_features(df)
    
    #split into features (X) and target (y)
    X = df_features.drop("value", axis=1) #x contains predictors
    y = df_features["value"] #y contains the group truth target we want to predict
    
    # time-series split (Train on first 80%, Test on last 20%)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    print("Training XGBoost Regressor...")
    #initialise the gradient boosted tree regressor with foundational hyperparametters
    model = xgb.XGBRegressor(
        n_estimators=100, #100 sequential decision trees
        learning_rate=0.1, #shrinkage factor 
        max_depth=5, #max tree depth
        random_state=42 #fix seed for reproducibility across runs
    )
    model.fit(X_train, y_train)
    
    # evaluate
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"✅ Model trained! Test Mean Absolute Error: {mae:.2f} EUR/MWh")
    
    # save the trained model artifact
    model_path = MODEL_DIR / "xgboost_price_model.json"
    model.save_model(model_path)
    print(f"Model saved to {model_path}")