from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from alpha_vantage.timeseries import TimeSeries
from ta.momentum import RSIIndicator
from matplotlib.widgets import SpanSelector

# Step 1: Collecting real-time stock market data
api_key = 'K6D1WUI4R67Z2LFS'
ts = TimeSeries(key=api_key, output_format='pandas')
symbol = 'AAPL'  # Example stock symbol
data, meta_data = ts.get_intraday(symbol=symbol, interval='1min', outputsize='full')
data['date'] = data.index.date  # Extracting date for grouping

# Step 2: Data processing and analysis
data['close_pct_change'] = data['4. close'].pct_change()
data.dropna(inplace=True)  # Remove rows with missing values
data['target'] = np.where(data['close_pct_change'].shift(-1) > 0, 1, 0)  # Binary target for trend prediction
# Fetch historical stock data for Apple (AAPL)
data1 = yf.download('AAPL', period='60d')

def preprocess_data(data1):
    # Calculate additional features
    data1['SMA_5'] = data1['Close'].rolling(window=5).mean()
    data1['SMA_20'] = data1['Close'].rolling(window=20).mean()
    data1['RSI'] = calculate_rsi(data1['Close'], window=14)
    macd, signal_line = calculate_macd(data1['Close'], window_short=12, window_long=26, window_signal=9)
    data1['MACD'] = macd
    data1['Signal_Line'] = signal_line

    # Drop rows with missing values
    data1.dropna(inplace=True)

    # Define the target variable (next day's closing price)
    data1['Target'] = data1['Close'].shift(-1)

    # Drop the last row (since there's no target for it)
    data1.drop(data1.tail(1).index, inplace=True)

    return data1

def calculate_rsi(data1, window=14):
    # Calculate the RSI
    delta = data1.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi1 = 100 - (100 / (1 + rs))
    return rsi1

def calculate_macd(data1, window_short=12, window_long=26, window_signal=9):
    # Calculate the MACD
    ema_short = data1.ewm(span=window_short, adjust=False).mean()
    ema_long = data1.ewm(span=window_long, adjust=False).mean()
    macd = ema_short - ema_long
    signal_line = macd.ewm(span=window_signal, adjust=False).mean()
    return macd, signal_line

# Preprocess the data
data1 = preprocess_data(data1)

# Split the data into features (X) and target variable (y)
X = data1[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_5', 'SMA_20', 'RSI', 'MACD']]
y = data1['Target']

# Normalize the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_scaled, y)

def index():
    # Fetch new real-time stock data
    new_data = yf.download('AAPL', period='90d')  # Fetch data for 90 days

    if len(new_data) < 20:
        # If there are not enough data points for preprocessing, return a message
        return "Insufficient data available for prediction."

    # Preprocess the new data
    new_data = preprocess_data(new_data)

    if len(new_data) == 0:
        # If the preprocessed data is empty, return a message indicating so
        return "No data available for prediction after preprocessing."

    # Normalize the features
    new_data_scaled = scaler.transform(new_data[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_5', 'SMA_20', 'RSI', 'MACD']])
    new_data_scaled = pd.DataFrame(new_data_scaled, columns=['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_5', 'SMA_20', 'RSI', 'MACD'])

    # Use the trained model to make a prediction
    prediction = model.predict(new_data_scaled.iloc[-1].values.reshape(1, -1))[0]
    print("Apple stock Prediction for the next day's closing price:" , prediction)
index()
def plot_graphs(data):
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    # Plotting candlestick chart
    ohlc = data[['date', '1. open', '2. high', '3. low', '4. close']].copy()
    ohlc['date'] = pd.to_datetime(ohlc['date'])
    ohlc['date'] = ohlc['date'].apply(mdates.date2num)
    axes[0].xaxis_date()
    axes[0].grid(True)
    axes[0].set_title('Candlestick Chart')
    candlestick_ohlc(axes[0], ohlc.values, width=0.0005, colorup='g', colordown='r')

    # Plotting closing price chart with RSI indicator
    rsi_period = 14
    rsi = RSIIndicator(data['4. close'], window=rsi_period)
    data['RSI'] = rsi.rsi()
    axes[1].plot(data.index, data['4. close'], label='Close Price', color='blue')
    axes[1].set_title('Close Price with RSI (14)')
    axes[1].grid(True)
    axes[1].set_ylabel('Price')
    axes[1].legend()
    axes_rsi = axes[1].twinx()
    axes_rsi.plot(data.index, data['RSI'], label='RSI (14)', color='orange')
    axes_rsi.axhline(70, color='r', linestyle='--')
    axes_rsi.axhline(30, color='g', linestyle='--')
    axes_rsi.set_ylabel('RSI')
    axes_rsi.legend()

    # Plotting volume chart
    axes[2].bar(data.index, data['5. volume'], color='gray')
    axes[2].set_title('Volume')
    axes[2].grid(True)
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Volume')

    # Plotting buy and sell signals based on RSI
    axes[3].plot(data.index, data['4. close'], label='Close Price', color='blue')
    axes[3].set_title('Buy and Sell Signals based on RSI (14)')
    axes[3].grid(True)
    axes[3].set_ylabel('Price')
    axes[3].legend()
    axes_rsi_signals = axes[3].twinx()
    axes_rsi_signals.plot(data.index, data['RSI'], label='RSI (14)', color='orange')
    axes_rsi_signals.axhline(70, color='r', linestyle='--')
    axes_rsi_signals.axhline(30, color='g', linestyle='--')
    axes_rsi_signals.set_ylabel('RSI')

    # Highlighting buy and sell signals
    buy_signal = (data['RSI'] < 30) & (data['RSI'].shift(1) >= 30)
    sell_signal = (data['RSI'] > 70) & (data['RSI'].shift(1) <= 70)
    axes_rsi_signals.plot(data.index[buy_signal], data['4. close'][buy_signal], '^', markersize=10, color='g', lw=0, label='Buy Signal')
    axes_rsi_signals.plot(data.index[sell_signal], data['4. close'][sell_signal], 'v', markersize=10, color='r', lw=0, label='Sell Signal')
    axes_rsi_signals.legend()

    plt.tight_layout()
    plt.show()
plot_graphs(data)
# Step 4: Trend prediction
X = data[['1. open', '2. high', '3. low', '4. close', '5. volume']]
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predicting next trend
next_data = data.iloc[[-1]]
next_trend = model.predict(next_data[['1. open', '2. high', '3. low', '4. close', '5. volume']])
if next_trend[0] > 0.5:
    print("The trend is likely to go up in the next period.")
else:
    print("The trend is likely to go down in the next period.")



