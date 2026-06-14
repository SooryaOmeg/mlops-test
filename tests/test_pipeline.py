"""
Essential tests for LSTM Stock Predictor.
Run with: pytest tests/test_pipeline.py -v
"""

import numpy as np
import pandas as pd
import pytest
import torch
import requests
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from unittest.mock import MagicMock, patch

# ── Copy only what we need from app.py ────────────────────────────────────────
# (avoids importing app.py which runs Streamlit code at module level)

class LSTMModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def fetch_stock_data(ticker):
    import time
    end_time = int(time.time())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1={end_time - 200*86400}&period2={end_time}&interval=1d"
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    if resp.status_code != 200:
        return None
    try:
        r = resp.json()["chart"]["result"][0]
        indicators = r["indicators"]["quote"][0]
        df = pd.DataFrame({
            "Date": pd.to_datetime(r["timestamp"], unit="s").date,
            "Open": indicators["open"], "High": indicators["high"],
            "Low":  indicators["low"],  "Close": indicators["close"],
            "Volume": indicators["volume"],
        })
        return df.dropna().reset_index(drop=True)
    except (KeyError, TypeError, IndexError):
        return None

# ── Fixtures ──────────────────────────────────────────────────────────────────

def _yahoo_payload(n=150):
    import time
    base = int(time.time()) - n * 86400
    return {"chart": {"result": [{"timestamp": [base + i*86400 for i in range(n)],
        "indicators": {"quote": [{"open":  [100.0+i*.1 for i in range(n)],
                                  "high":  [101.0+i*.1 for i in range(n)],
                                  "low":   [ 99.0+i*.1 for i in range(n)],
                                  "close": [100.5+i*.1 for i in range(n)],
                                  "volume":[1_000_000  for _ in range(n)]}]}}]}}

@pytest.fixture
def model():
    return LSTMModel()

@pytest.fixture
def df():
    n = 170
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=n).date,
                         "Open": close*.99, "High": close*1.01,
                         "Low":  close*.98,  "Close": close,
                         "Volume": rng.integers(500_000, 2_000_000, n)})

# ── Model tests ───────────────────────────────────────────────────────────────

def test_output_shape(model):
    assert model(torch.randn(4, 60, 5)).shape == (4, 1)

def test_output_is_finite(model):
    assert torch.all(torch.isfinite(model(torch.randn(8, 60, 5))))

def test_custom_hidden_size():
    m = LSTMModel(hidden_size=128)
    assert m.fc.in_features == 128

def test_no_grad_eval(model):
    model.eval()
    with torch.no_grad():
        assert not model(torch.randn(2, 60, 5)).requires_grad

# ── Fetch tests ───────────────────────────────────────────────────────────────

@patch("requests.get")
def test_fetch_success(mock_get):
    mock_get.return_value = MagicMock(status_code=200, json=lambda: _yahoo_payload(150))
    result = fetch_stock_data("AAPL")
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"Date","Open","High","Low","Close","Volume"}

@patch("requests.get")
def test_fetch_non_200_returns_none(mock_get):
    mock_get.return_value = MagicMock(status_code=404)
    assert fetch_stock_data("BAD") is None

@patch("requests.get")
def test_fetch_bad_json_returns_none(mock_get):
    mock_get.return_value = MagicMock(status_code=200, json=lambda: {})
    assert fetch_stock_data("AAPL") is None

@patch("requests.get")
def test_fetch_drops_null_rows(mock_get):
    payload = _yahoo_payload(10)
    payload["chart"]["result"][0]["indicators"]["quote"][0]["close"][2] = None
    mock_get.return_value = MagicMock(status_code=200, json=lambda: payload)
    df = fetch_stock_data("AAPL")
    assert df["Close"].isna().sum() == 0

# ── Preprocessing & inverse transform ────────────────────────────────────────

def test_scaler_output_range(df):
    features = ["Open","High","Low","Close","Volume"]
    scaled = MinMaxScaler().fit_transform(df[features])
    assert scaled.min() >= 0.0 and scaled.max() <= 1.0

def test_inverse_transform_recovers_close(df):
    features = ["Open","High","Low","Close","Volume"]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])
    dummy = np.zeros((len(df), 5))
    dummy[:, 3] = scaled[:, 3]
    recovered = scaler.inverse_transform(dummy)[:, 3]
    np.testing.assert_allclose(recovered, df["Close"].values, rtol=1e-4)

# ── End-to-end ────────────────────────────────────────────────────────────────

def test_full_pipeline(df):
    features = ["Open","High","Low","Close","Volume"]
    predict_window, seq_length = 30, 60
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])
    n = len(scaled)
    x = torch.tensor(
        np.array([scaled[i-seq_length:i] for i in range(n-predict_window, n)]),
        dtype=torch.float32
    )
    model = LSTMModel()
    model.eval()
    with torch.no_grad():
        preds_scaled = model(x).numpy()
    dummy = np.zeros((predict_window, 5))
    dummy[:, 3] = preds_scaled[:, 0]
    prices = scaler.inverse_transform(dummy)[:, 3]
    assert prices.shape == (predict_window,)
    assert np.all(np.isfinite(prices))