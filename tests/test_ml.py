import pytest
import pandas as pd
import numpy as np
from ml_pipeline.train import create_features

@pytest.fixture
def sample_price_data():
    """
    fixture that creates 48 hours of synthetic price data
    this gives the test isolated, predictable data to work with
    """
    dates = pd.date_range(start="2024-03-01 00:00:00", periods=48, freq="h")
    #generate prices steadily increasing from 50 to 97
    values = np.linspace(50.0, 97.0, 48)
    df = pd.DataFrame({"value": values}, index=dates)
    return df

def test_create_features(sample_price_data):
    """
    tests if the feature engineering logic calculates lags and drops NaNs correctly
    """
    df_features = create_features(sample_price_data)
    #because of lag_24 and rolling_mean_24, the first 24 rows should become NaN
    assert len(df_features) == 24, "Failed to drop exactly 24 NaN rows"

   #check if all columns were created 
    expected_cols = ['value', 'hour', 'dayofweek', 'lag_1', 'lag_24', 'rolling_mean_24']
    for col in expected_cols:
        assert col in df_features.columns, f"Missing engineered column: {col}"

    #verify the mathematical logic on the first valid row    
    first_valid_row = df_features.iloc[0]
    #the value 1 hour before index 24
    assert first_valid_row['lag_1'] == 73.0
    #the value 24 hours before index 24
    assert first_valid_row['lag_24'] == 50.0