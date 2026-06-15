import torch
from torch import nn
import pandas as pd
import numpy as np
import requests
import time
import sys
from sklearn.preprocessing import MinMaxScaler

# 1. Recreate Model Architecture (Same as your app.py)
class LSTMModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# 2. Fetch Data (Get the last 65 days)
ticker = "AAPL" # Or your preferred ticker
end_time = int(time.time())
start_time = end_time - (65 * 24 * 60 * 60)
url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1={start_time}&period2={end_time}&interval=1d"
headers = {"User-Agent": "Mozilla/5.0"}

response = requests.get(url, headers=headers)
data = response.json()
chart_result = data['chart']['result'][0]
indicators = chart_result['indicators']['quote'][0]

df = pd.DataFrame({
    'Open': indicators['open'], 'High': indicators['high'],
    'Low': indicators['low'], 'Close': indicators['close'], 'Volume': indicators['volume']
}).dropna().reset_index(drop=True)

# 3. Prepare Data for "Yesterday's" Prediction
# We use the 60 days ending TWO days ago to predict YESTERDAY.
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# Window for prediction: from index -61 up to -1 (excludes the very last day)
prediction_window = scaled_data[-61:-1]
x_input = torch.tensor(prediction_window, dtype=torch.float32).unsqueeze(0)

# 4. Run Model
model = LSTMModel()
model.load_state_dict(torch.load("lstm_stock_model.pth", map_location="cpu"))
model.eval()

with torch.no_grad():
    pred_scaled = model(x_input).item()

# Inverse scale the prediction
dummy = np.zeros((1, 5))
dummy[0, 3] = pred_scaled
predicted_yesterday = scaler.inverse_transform(dummy)[0, 3]

# 5. Get the ACTUAL price from yesterday (the very last row in our dataframe)
actual_yesterday = df['Close'].iloc[-1]

# 6. Calculate Mean Absolute Error (MAE)
error = abs(predicted_yesterday - actual_yesterday)

print(f"--- MLOps Daily Monitor: {ticker} ---")
print(f"Predicted Close: ${predicted_yesterday:.2f}")
print(f"Actual Close:    ${actual_yesterday:.2f}")
print(f"Absolute Error:  ${error:.2f}")

# 7. Alert System
ALERT_THRESHOLD = 5.00 # If the model is off by more than $5, trigger an alert

if error > ALERT_THRESHOLD:
    print(f"⚠️ ALERT: Model drift detected! Error (${error:.2f}) exceeds threshold (${ALERT_THRESHOLD:.2f}).")
    sys.exit(1) # This forces the GitHub Action to "Fail" and email you
else:
    print("✅ Model is performing within acceptable limits.")
    sys.exit(0)