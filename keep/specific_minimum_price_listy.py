import requests

# OKX API URL for fetching trading pairs and ticker info
url_instruments = "https://www.okx.com/api/v5/public/instruments"
url_ticker = "https://www.okx.com/api/v5/market/ticker"

# Parameters for retrieving only USDT pairs in the SPOT market
params_instruments = {
    'instType': 'SPOT',
    'quoteCcy': 'USDT'  # Only pairs with USDT as quote currency
}

# List of specific pairs to retrieve
target_pairs = ['BTC-USDT', 'BCH-USDT', 'ETH-USDT', 'SOL-USDT', 'SUI-USDT', 'XRP-USDT', 'LTC-USDT', 'TON-USDT']

# Send a request to the OKX API to get instruments
response_instruments = requests.get(url_instruments, params=params_instruments)

# Parse the response JSON for instruments
data_instruments = response_instruments.json()

# Function to extract minimum trade amounts for COIN/USDT pairs with equivalent USDT value
def get_min_trade_amounts_with_usdt(instruments):
    min_trade_amounts = []
    
    for instrument in instruments:
        pair = instrument['instId']  # Trading pair, e.g., BTC/USDT
        
        # Check if the pair is one of the target pairs
        if pair in target_pairs:
            min_size = float(instrument['minSz'])  # Minimum trade size
            base_currency = instrument['baseCcy']  # Base currency, e.g., BTC

            # Get the current price for the pair (COIN-USDT)
            params_ticker = {
                'instId': pair
            }
            response_ticker = requests.get(url_ticker, params=params_ticker)
            data_ticker = response_ticker.json()

            if data_ticker['code'] == '0':  # Successful response
                current_price = float(data_ticker['data'][0]['last'])  # Current price of the pair in USDT
                
                # Calculate the equivalent value in USDT for the minimum trade amount
                min_trade_value_in_usdt = min_size * current_price
                
                # Create a formatted string
                min_trade_amounts.append(
                    f"Minimum trade amount of {pair} is {min_size} {base_currency} and {min_size} {base_currency} is approximately {min_trade_value_in_usdt:.6f} USDT"
                )
            else:
                # Handle error in ticker data request
                min_trade_amounts.append(f"Error fetching price for {pair}: {data_ticker['msg']}")
    
    return min_trade_amounts

# Extract the instruments data
if data_instruments['code'] == '0':  # Check if the request was successful
    instruments = data_instruments['data']
    min_trade_amounts = get_min_trade_amounts_with_usdt(instruments)
    
    # Print out the minimum trade amounts for the specified USDT pairs
    for min_trade in min_trade_amounts:
        print(min_trade)
else:
    print(f"Error: {data_instruments['msg']}")
