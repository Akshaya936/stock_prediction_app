import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

# -------------------------
# 1. Page Setup
# -------------------------
st.set_page_config(page_title="NSE Nifty Stock Prediction", layout="centered")
st.title("📈 NSE Nifty Stock Prediction")
st.write("Select a company to predict the next day's closing price using LSTM.")

# -------------------------
# 2. Nifty 50 Company List
# -------------------------
nifty_50 = {
    'ADANIENT.NS': 'Adani Enterprises', 'ADANIPORTS.NS': 'Adani Ports',
    'ASIANPAINT.NS': 'Asian Paints', 'AXISBANK.NS': 'Axis Bank',
    'BAJAJ-AUTO.NS': 'Bajaj Auto', 'BAJFINANCE.NS': 'Bajaj Finance',
    'BAJAJFINSV.NS': 'Bajaj Finserv', 'BHARTIARTL.NS': 'Bharti Airtel',
    'BPCL.NS': 'BPCL', 'BRITANNIA.NS': 'Britannia', 'CIPLA.NS': 'Cipla',
    'COALINDIA.NS': 'Coal India', 'DIVISLAB.NS': 'Divi’s Labs',
    'DRREDDY.NS': 'Dr. Reddy’s Labs', 'EICHERMOT.NS': 'Eicher Motors',
    'GRASIM.NS': 'Grasim Industries', 'HCLTECH.NS': 'HCL Tech',
    'HDFCBANK.NS': 'HDFC Bank', 'HDFCLIFE.NS': 'HDFC Life',
    'HEROMOTOCO.NS': 'Hero MotoCorp', 'HINDALCO.NS': 'Hindalco',
    'HINDUNILVR.NS': 'Hindustan Unilever', 'ICICIBANK.NS': 'ICICI Bank',
    'INDUSINDBK.NS': 'IndusInd Bank', 'INFY.NS': 'Infosys', 'ITC.NS': 'ITC',
    'JSWSTEEL.NS': 'JSW Steel', 'KOTAKBANK.NS': 'Kotak Mahindra Bank',
    'LT.NS': 'Larsen & Toubro', 'M&M.NS': 'Mahindra & Mahindra',
    'MARUTI.NS': 'Maruti Suzuki', 'NESTLEIND.NS': 'Nestle India',
    'NTPC.NS': 'NTPC', 'ONGC.NS': 'ONGC', 'POWERGRID.NS': 'Power Grid',
    'RELIANCE.NS': 'Reliance Industries', 'SBILIFE.NS': 'SBI Life',
    'SBIN.NS': 'SBI', 'SHREECEM.NS': 'Shree Cement', 'SUNPHARMA.NS': 'Sun Pharma',
    'TATACONSUM.NS': 'Tata Consumer', 'TATAMOTORS.NS': 'Tata Motors',
    'TATASTEEL.NS': 'Tata Steel', 'TCS.NS': 'TCS', 'TECHM.NS': 'Tech Mahindra',
    'TITAN.NS': 'Titan Company', 'ULTRACEMCO.NS': 'UltraTech Cement',
    'UPL.NS': 'UPL', 'WIPRO.NS': 'Wipro'
}

symbol = st.selectbox("Choose a Company", list(nifty_50.keys()), format_func=lambda x: nifty_50[x])

# -------------------------
# 3. Get Live Data
# -------------------------
@st.cache_data
def get_data(symbol):
    today = datetime.today().strftime('%Y-%m-%d')
    df = yf.download(symbol, start="2024-01-01", end=today, interval="1d")
    return df

if symbol:
    df = get_data(symbol)

    if df.empty:
        st.error("❌ No data found. Please try a different stock or check again later.")
    else:
        df = df.reset_index()
        # Fix multi-index columns from yfinance
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        st.subheader(f"{nifty_50[symbol]} - Last 5 Trading Days")
        st.dataframe(
        df[['Date', 'Close', 'High']].tail(),
        use_container_width=True,
        height=200
        )
        st.write(f"🗓️ Latest Data Date: **{df['Date'].iloc[-1].date()}**")

# -------------------------
# 4. Preprocessing
# -------------------------
        data = df[['Close']].values
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

        sequence_length = 60
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(scaled_data[i])

        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        train_size = int(len(X) * 0.7)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

# -------------------------
# 5. Build LSTM Model
# -------------------------
@st.cache_resource
def train_model(X_train, y_train):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
    return model

model = train_model(X_train, y_train)

# -------------------------
# 6. Predict Next Day Price
# -------------------------
last_60 = scaled_data[-60:]
X_input = last_60.reshape(1, 60, 1)
pred_scaled = model.predict(X_input)
pred_price = scaler.inverse_transform(pred_scaled)[0][0]

st.success(f"🔮 Predicted Next Day Closing Price: ₹{pred_price:.2f}")

# -------------------------
# 7. Plot Forecast
# -------------------------
train_plot = scaler.inverse_transform(y_train)
actual_plot = scaler.inverse_transform(y_test)
predicted_plot = scaler.inverse_transform(model.predict(X_test))

fig, ax = plt.subplots()
ax.plot(range(len(train_plot)), train_plot, label='Train')
ax.plot(range(len(train_plot), len(train_plot) + len(actual_plot)), actual_plot, label='Actual', color='orange')
ax.plot(range(len(train_plot), len(train_plot) + len(predicted_plot)), predicted_plot, label='Predicted', color='green')
ax.set_title(f"{nifty_50[symbol]} - Price Forecast")
ax.set_xlabel("Time")
ax.set_ylabel("Closing Price (₹)")
ax.legend()
st.pyplot(fig)
