import pandas as pd
import ta  # Import the ta library

# Load the candlestick data
data = pd.read_csv('.\\dataset\\candlestick_data1mBTC.csv')

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Set 'Date' as the index
data.set_index('Date', inplace=True)

# Calculate Simple Moving Average (SMA)
data['MA_20'] = ta.trend.sma_indicator(data['Close'], window=20)  # 20-period SMA
data['MA_50'] = ta.trend.sma_indicator(data['Close'], window=50)  # 50-period SMA

# Calculate Relative Strength Index (RSI)
data['RSI'] = ta.momentum.rsi(data['Close'], window=14)  # 14-period RSI

# Calculate MACD
data['MACD'] = ta.trend.macd(data['Close'])
data['MACD_Signal'] = ta.trend.macd_signal(data['Close'])
data['MACD_Diff'] = ta.trend.macd_diff(data['Close'])

# Calculate Bollinger Bands
data['Bollinger_Upper'] = data['MA_20'] + (2 * data['Close'].rolling(window=20).std())
data['Bollinger_Lower'] = data['MA_20'] - (2 * data['Close'].rolling(window=20).std())

# Drop rows with NaN values
data.dropna(inplace=True)

# Save the processed data to a new CSV file
data.to_csv('.\\dataset\\candlestick_data_preprocess.csv', index=True)

print("Processed data saved to 'preprocess_candlestick_data.csv'")
