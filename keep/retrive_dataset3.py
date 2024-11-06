# #pip install python-okx
# api_key = "a89247f7-d58d-4ed3-b23e-19c8074933b2"
# secret_key = "71DF873F472CABE3BE1BECABOE30EEDD"
# passphrase = "Chut05051@"

# import requests
# import time
# import hmac
# import hashlib
# import base64
# import csv
# from datetime import datetime
# import pandas as pd
# import mplfinance as mpf  # Importing mplfinance for candlestick plotting

# # Base URL for the OKX demo trading environment
# base_url = "https://www.okx.com"

# def sign(message, secret_key):
#     # HMAC-SHA256 signature
#     return base64.b64encode(hmac.new(secret_key.encode(), message.encode(), hashlib.sha256).digest())

# def get_headers(api_key, secret_key, passphrase):
#     timestamp = str(time.time())
#     message = timestamp + 'GET' + '/api/v5/market/candles'
#     signature = sign(message, secret_key)
    
#     return {
#         'OK-ACCESS-KEY': api_key,
#         'OK-ACCESS-SIGN': signature.decode(),
#         'OK-ACCESS-TIMESTAMP': timestamp,
#         'OK-ACCESS-PASSPHRASE': passphrase,
#         'Content-Type': 'application/json'
#     }

# def get_historical_data(symbol, bar, limit):
#     endpoint = f'/api/v5/market/candles?instId={symbol}&bar={bar}&limit={limit}'
#     url = base_url + endpoint
#     headers = get_headers(api_key, secret_key, passphrase)

#     response = requests.get(url, headers=headers)
#     if response.status_code == 200:
#         return response.json()
#     else:
#         print(f"Error: {response.status_code} {response.text}")
#         return None

# # Function to print data in date | OHLCV format
# def print_candlestick_data(data):
#     print(f"{'Date':<20} | {'Day':<4} | {'Open':<10} | {'High':<10} | {'Low':<10} | {'Close':<10} | {'Volume':<15}")
#     print('-' * 100)
    
#     for candle in data['data']:
#         timestamp = int(candle[0]) / 1000  # Convert milliseconds to seconds
#         date = datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
#         day_of_week = (datetime.utcfromtimestamp(timestamp).isoweekday())  # Day of week (1=Monday, 7=Sunday)
#         open_price = candle[1]  # Open price
#         high_price = candle[2]  # High price
#         low_price = candle[3]  # Low price
#         close_price = candle[4]  # Close price
#         volume = candle[5]  # Volume
        
#         print(f"{date:<20} | {day_of_week:<4} | {open_price:<10} | {high_price:<10} | {low_price:<10} | {close_price:<10} | {volume:<15}")

# # Function to save data to CSV
# def save_to_csv(data, filename):
#     with open(filename, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(['Date', 'Day', 'Open', 'High', 'Low', 'Close', 'Volume'])  # Header
#         for candle in data['data']:
#             timestamp = int(candle[0]) / 1000  # Convert milliseconds to seconds
#             date = datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
#             day_of_week = (datetime.utcfromtimestamp(timestamp).isoweekday())  # Day of week
#             open_price = candle[1]  # Open price
#             high_price = candle[2]  # High price
#             low_price = candle[3]  # Low price
#             close_price = candle[4]  # Close price
#             volume = candle[5]  # Volume
            
#             writer.writerow([date, day_of_week, open_price, high_price, low_price, close_price, volume])  # Write data

# # Function to plot candlestick data
# def plot_candlestick(data):
#     # Preparing the DataFrame
#     candles = []
#     for candle in data['data']:
#         timestamp = int(candle[0]) / 1000  # Convert milliseconds to seconds
#         date = datetime.utcfromtimestamp(timestamp)
#         open_price = float(candle[1])
#         high_price = float(candle[2])
#         low_price = float(candle[3])
#         close_price = float(candle[4])
#         volume = float(candle[5])
        
#         candles.append([date, open_price, high_price, low_price, close_price, volume])
    
#     df = pd.DataFrame(candles, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
#     df.set_index('Date', inplace=True)
    
#     # Plotting the candlestick chart
#     mpf.plot(df, type='candle', volume=False, style='charles', title='BTC-USDT Candlestick Chart', ylabel='Price', xlabel='Date')

# # Example: Fetch historical candlestick data for BTC-USDT in 1-day intervals
# symbol = "SOL-USDT"
# # 1m/5m/15m/30m/1H/2H/4H/6H/12H/1D/2D/3D/5D/1W/1M/3M
# data = get_historical_data(symbol, '1m', 300)

# if data:
#     print_candlestick_data(data)
#     save_to_csv(data, 'candlestick_data.csv')  # Save data to CSV
#     print("Data saved to candlestick_data.csv")
#     plot_candlestick(data)  # Plot the candlestick chart
import okx.MarketData as MarketData
from datetime import datetime
import pandas as pd

flag = "0"  # Production trading:0 , demo trading:1

marketDataAPI = MarketData.MarketAPI(flag=flag)

#2024-09-18 21:59:00 to 2024-09-18 22:59:00
#2024-09-18 22:59:00 to 2024-09-18 23:59:00
#             year / month / day / time
#dt = datetime(2024, 9, 16, 22, 59, 0)  # Example: 19th October 2024, 12:00:00
dt = datetime(2024, 9, 16, 21, 59, 0)  # Example: 19th October 2024, 12:00:00
# Convert to Unix timestamp in milliseconds
before = int(dt.timestamp() * 1000)
#print(timestamp_ms)

#dt = datetime(2024, 9, 17, 0, 0, 0)  # Example: 19th October 2024, 12:00:00
dt = datetime(2024, 9, 16, 23, 0, 0)  # Example: 19th October 2024, 12:00:00
# Convert to Unix timestamp in milliseconds
after = int(dt.timestamp() * 1000)
#print(timestamp_ms)

# 1m/5m/15m/30m/1H/2H/4H/6H/12H/1D/2D/3D/5D/1W/1M/3M

# 1m/1H/1D
# Retrieve the candlestick charts
result = marketDataAPI.get_history_candlesticks(#.get_candlesticks(            #before: 2024/10/19 20:41 and after: 2024/10/19 20:39
    instId="BTC-USDT", bar='1m', limit='300',before=before,after=after
)

if result['code'] == '0':
    print("Candlestick Data:")
    
    # Extract and sort candlestick data by timestamp
    candles = result['data']
    
    # Sort candles by the first element in each sublist (the timestamp)
    sorted_candles = sorted(candles, key=lambda x: x[0])
    
    # List to store the formatted data for the DataFrame
    data = []

    for candle in sorted_candles:
        # Ensure the timestamp is treated as an integer
        timestamp = int(candle[0])  # Convert to int
        open_price = candle[1]
        close_price = candle[2]
        low_price = candle[3]
        high_price = candle[4]
        volume = candle[5]

        # Convert timestamp from milliseconds to seconds
        timestamp_sec = timestamp / 1000

        # Format the timestamp to 'YYYY/MM/DD HH:MM'
        formatted_time = datetime.fromtimestamp(timestamp_sec).strftime('%Y-%m-%d %H:%M:%S')

        # Extract the day of the week (0 = Monday, 6 = Sunday)
        day_of_week = datetime.fromtimestamp(timestamp_sec).strftime('%w')

        # Append the formatted data to the list
        data.append([formatted_time, day_of_week, open_price, high_price, low_price, close_price, volume])
    
    # Create a DataFrame from the data
    df = pd.DataFrame(data, columns=['Date', 'Day', 'Open', 'High', 'Low', 'Close', 'Volume'])
    
    # Save the DataFrame as a CSV file
    df.to_csv('candlestick_data.csv', index=False)
    
    # Print the table to the console
    print(df)
    
    print("Data saved to 'candlestick_data.csv'")
else:
    print(f"Error: {result['msg']}")