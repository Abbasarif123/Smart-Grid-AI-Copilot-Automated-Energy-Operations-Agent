import pytest
import pandas as pd
from unittest.mock import patch
from agent.tools import query_historical_spot_prices

@pytest.fixture
def mock_parquet_data():
    """
    fixture providing a minimal dataset to mimic the SMARD API
    """
    dates = pd.date_range(start="2024-03-01", periods=4, freq="D")
    df = pd.DataFrame({"value": [40.0, 60.0, 80.0, 100.0]}, index=dates)
    return df


@patch("agent.tools.DATA_PATH")
@patch("agent.tools.pd.read_parquet")
def test_query_historical_spot_prices_valid_range(mock_read_parquet, mock_data_path, mock_parquet_data):
    """
    tests if the tool correctly calculates min, peak, and average for a valid date range
    """
    # configure our mocked variables
    mock_data_path.exists.return_value = True
    mock_read_parquet.return_value = mock_parquet_data
    
    # LangChain tools must be called using invoke()
    result = query_historical_spot_prices.invoke({
        "start_date": "2024-03-01",
        "end_date": "2024-03-02"
    })
    
    # filter should isolate the first two days (40.0 and 60.0)
    assert "Average price = 50.00 EUR/MWh" in result
    assert "Min = 40.00 EUR/MWh" in result
    assert "Peak = 60.00 EUR/MWh" in result

@patch("agent.tools.DATA_PATH")
def test_query_historical_spot_prices_missing_file(mock_data_path):
    """
    tests the tool's safety guardrail when the Parquet file hasn't been downloaded yet
    """
    mock_data_path.exists.return_value = False
    
    result = query_historical_spot_prices.invoke({
        "start_date": "2024-03-01",
        "end_date": "2024-03-05"
    })
    
    assert "Error: Data file not found" in result