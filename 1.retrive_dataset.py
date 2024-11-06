from datetime import datetime, timedelta
import okx.MarketData as MarketData
from datetime import datetime
import pandas as pd

flag = "0"  # Production trading:0 , demo trading:1

marketDataAPI = MarketData.MarketAPI(flag=flag)
# dt1 = datetime(2024, 10, 20, 0, 0, 0) #end
# dt2 = datetime(2024, 10, 1, 0, 0, 0) #start
dt1 = datetime(2024, 11, 4, 0, 0, 0) #end
dt2 = datetime(2024, 10, 1, 0, 0, 0) #start
delta = timedelta(hours=1)

all_data = []  # Store data from all iterations

while dt2 < dt1:
    start_time = dt2 - timedelta(minutes=1)
    end_time = dt2 + delta - timedelta(minutes=1) + timedelta(minutes=1)
    print(f"{start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    dt = start_time
    before = int(dt.timestamp() * 1000)
    dt = end_time
    after = int(dt.timestamp() * 1000)
    #[1s/1m/3m/5m/15m/30m/1H/2H/4H][6H/12H/1D/2D/3D/1W/1M/3M]
    result = marketDataAPI.get_history_candlesticks(
        instId="BTC-USDT", bar='1m', limit='60', before=before, after=after)

    if result['code'] == '0':
        print("Candlestick Data:")
        candles = result['data']
        sorted_candles = sorted(candles, key=lambda x: x[0])
        data = []

        for candle in sorted_candles:
            timestamp = int(candle[0])
            open_price = candle[1]
            close_price = candle[2]
            low_price = candle[3]
            high_price = candle[4]
            volume = candle[5]
            timestamp_sec = timestamp / 1000
            formatted_time = datetime.fromtimestamp(timestamp_sec).strftime('%Y-%m-%d %H:%M:%S')
            day_of_week = datetime.fromtimestamp(timestamp_sec).strftime('%w')
            all_data.append([formatted_time, day_of_week, open_price, high_price, low_price, close_price, volume])

    else:
        print(f"Error: {result['msg']}")    
    
    dt2 += delta

# After loop ends, create a DataFrame from all_data and save it as a CSV
df_all = pd.DataFrame(all_data, columns=['Date', 'Day', 'Open', 'High', 'Low', 'Close', 'Volume'])
df_all.to_csv('.\\dataset\\candlestick_data1mBTC.csv', index=False)
print(df_all)

