import ccxt
import pandas as pd

# Fetch historical data from OKX or Binance
exchange = ccxt.okx()  # or ccxt.binance()
symbol = 'BTC/USDT'
timeframe = '1m'

# Fetch OHLCV data
ohlcv = exchange.fetch_ohlcv(symbol, timeframe)
data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
data.set_index('timestamp', inplace=True)

# To display all rows without truncation
pd.set_option('display.max_rows', None)

# Print all the data
print(data)
