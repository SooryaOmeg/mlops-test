import streamlit as st
import torch
from torch import nn
import numpy as np
import pandas as pd
import requests
import time
from sklearn.preprocessing import MinMaxScaler
import datetime

# --------------------------------------------------------------------
# 1. Define Model Architecture (Must match your trained model exactly)
# --------------------------------------------------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# Cache the model loading step so it doesn't reload on every button click
@st.cache_resource
def load_prediction_model():
    model = LSTMModel()
    # Ensure it maps to CPU for Hugging Face Spaces free instances
    model.load_state_dict(torch.load("lstm_stock_model.pth", map_location="cpu"))
    model.eval()
    return model

# --------------------------------------------------------------------
# 2. Fetch Data directly from Yahoo Finance Chart API
# --------------------------------------------------------------------
def fetch_stock_data(ticker):
    # Fetch roughly the last 200 days to guarantee at least 120 valid trading days
    end_time = int(time.time())
    start_time = end_time - (200 * 24 * 60 * 60)
    
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1={start_time}&period2={end_time}&interval=1d"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return None
        
    data = response.json()
    
    try:
        chart_result = data['chart']['result'][0]
        timestamps = chart_result['timestamp']
        indicators = chart_result['indicators']['quote'][0]
        
        df = pd.DataFrame({
            'Date': pd.to_datetime(timestamps, unit='s').date,
            'Open': indicators['open'],
            'High': indicators['high'],
            'Low': indicators['low'],
            'Close': indicators['close'],
            'Volume': indicators['volume']
        })
        
        # Drop any days with missing values
        df = df.dropna().reset_index(drop=True)
        return df
    except (KeyError, TypeError, IndexError):
        return None

# --------------------------------------------------------------------
# 3. Streamlit Interface
# --------------------------------------------------------------------
st.set_page_config(page_title="LSTM Stock Prediction Pipeline", layout="wide")

st.title("📈 LSTM Stock Value Predictor")
st.caption("An interactive MLOps inference dashboard powered by PyTorch & Streamlit.")

# Sidebar Input
st.sidebar.header("Configuration")
ticker_input = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, RELIANCE.NS)", value="AAPL").upper()
predict_window = st.sidebar.slider("Days to Predict Backwards (Rolling Window)", min_value=10, max_value=90, value=60)
run_prediction = st.sidebar.button("Fetch Data & Predict")

if run_prediction:
    with st.spinner(f"Fetching data for {ticker_input} and running pipeline..."):
        # 1. Load Model
        try:
            model = load_prediction_model()
        except FileNotFoundError:
            st.error("Error: `lstm_stock_model.pth` file not found in the root directory. Please upload your model weights.")
            st.stop()

        # 2. Fetch Data
        df = fetch_stock_data(ticker_input)
        
        # Ensure enough data for predict_window + 60 days of sequence context
        if df is None or len(df) < (predict_window + 60):
            st.error(f"Could not retrieve enough data for ticker. Need at least {predict_window + 60} trading days of history.")
        else:
            # Isolate standard OHLCV structural features
            features = ['Open', 'High', 'Low', 'Close', 'Volume']
            df_features = df[features].copy()
            
            # 3. Preprocessing (Scale data generically for sequence slicing)
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df_features)
            
            # 4. Generate rolling predictions for the last N days
            seq_length = 60
            n_total = len(df)
            
            x_batches = []
            pred_dates = []
            
            # We want predictions for the ending `predict_window` days up to today
            start_predict_idx = n_total - predict_window
            for i in range(start_predict_idx, n_total):
                x_batches.append(scaled_data[i - seq_length : i])
                pred_dates.append(df['Date'].iloc[i])
                
            x_tensor = torch.tensor(np.array(x_batches), dtype=torch.float32)
            
            with torch.no_grad():
                pred_scaled = model(x_tensor).numpy() # shape: (predict_window, 1)

            # 5. Inverse Transformation
            dummy = np.zeros((predict_window, 5))
            dummy[:, 3] = pred_scaled[:, 0]
            pred_prices = scaler.inverse_transform(dummy)[:, 3]
            
            # --------------------------------------------------------
            # 6. Render UI Layout
            # --------------------------------------------------------
            last_actual_close = df['Close'].iloc[-1]
            last_pred_close = pred_prices[-1]
            last_date = df['Date'].iloc[-1]
            
            col1, col2 = st.columns(2)
            col1.metric(label=f"Actual Close ({last_date})", value=f"${last_actual_close:.2f}")
            col2.metric(label=f"Model Prediction ({last_date})", value=f"${last_pred_close:.2f}", 
                        delta=f"{last_pred_close - last_actual_close:.2f} Error")
            
            st.subheader(f"Timeline: Last {predict_window} Days Rolling Predictions vs Actual")
            
            # Prepare plotting dataframe for the last predict_window days (Actual)
            actual_df = df.iloc[start_predict_idx:].copy()[['Date', 'Close']]
            actual_df['Type'] = 'Actual'
            
            # Prepare dataframe for predictions
            pred_df = pd.DataFrame({
                'Date': pred_dates,
                'Close': pred_prices,
                'Type': 'Prediction'
            })
            
            combined_df = pd.concat([actual_df, pred_df], ignore_index=True)
            
            # Generate Native Line Chart Grouped by Type
            st.line_chart(combined_df, x='Date', y='Close', color='Type', width='stretch')
            
            # Display Raw Data View
            with st.expander(f"View predictions vs actuals (Last {predict_window} Days)"):
                comparison_df = actual_df.copy()
                comparison_df = comparison_df.rename(columns={'Close': 'Actual Close'})
                comparison_df['Predicted Close'] = pred_prices
                comparison_df['Error Abs'] = abs(comparison_df['Actual Close'] - comparison_df['Predicted Close'])
                st.dataframe(comparison_df.drop(columns=['Type']))