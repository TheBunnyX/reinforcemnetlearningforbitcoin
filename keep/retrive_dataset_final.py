import ccxt
import pandas as pd
import time

# Initialize exchange (choose OKX or Binance)
exchange = ccxt.okx()  # or ccxt.binance()
symbol = 'SOL/USDT'
timeframe = '1d'

# Define start and end date (convert to milliseconds)
# 'YYYY-MM-DD HH:MM:SS'
start_date = '2024-09-10 00:00:00'
end_date = '2024-10-10 00:00:00'
start_timestamp = int(pd.Timestamp(start_date).timestamp() * 1000)
end_timestamp = int(pd.Timestamp(end_date).timestamp() * 1000)

# Fetch OHLCV data in a loop (if necessary)
all_data = []
while start_timestamp < end_timestamp:
    # Fetch data (limit to 1000 points per request if needed by the exchange)
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=start_timestamp, limit=1000)
    
    if not ohlcv:
        break  # Break the loop if no data is returned
    
    # Append fetched data
    all_data.extend(ohlcv)
    
    # Update the start_timestamp to the last fetched timestamp
    start_timestamp = ohlcv[-1][0] + 1  # Add 1 ms to avoid overlap

    # Sleep to avoid hitting rate limits
    time.sleep(exchange.rateLimit / 1000)

# Convert to DataFrame
data = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
data.set_index('timestamp', inplace=True)

# Filter data to ensure it stays within the exact start and end range
data = data.loc[start_date:end_date]

# Save to CSV
csv_filename = 'SOL_USDT_OHLCV_1m_2024-09-10_to_2024-10-10.csv'
data.to_csv(csv_filename)

# To display all rows without truncation
pd.set_option('display.max_rows', None)

# Print all the data
print(data)

print(f'Data saved to {csv_filename}')