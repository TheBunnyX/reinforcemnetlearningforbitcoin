import pandas as pd
import numpy as np

def is_hammer(candle):
    body = abs(candle['close'] - candle['open'])
    range_ = candle['high'] - candle['low']
    lower_shadow = min(candle['open'], candle['close']) - candle['low']
    
    return lower_shadow > 2 * body and body < 0.3 * range_

def is_doji(candle):
    body = abs(candle['close'] - candle['open'])
    range_ = candle['high'] - candle['low']
    
    return body < 0.1 * range_

def is_engulfing(df, i):
    if i == 0 or df['open'][i] < df['close'][i]:  # Bullish Engulfing
        return (df['open'][i-1] > df['close'][i-1] and 
                df['open'][i] < df['close'][i] and 
                df['close'][i] > df['open'][i-1] and 
                df['open'][i] < df['close'][i-1])
    else:  # Bearish Engulfing
        return (df['open'][i-1] < df['close'][i-1] and 
                df['open'][i] > df['close'][i] and 
                df['close'][i] < df['open'][i-1] and 
                df['open'][i] > df['close'][i-1])

def is_shooting_star(candle):
    body = abs(candle['close'] - candle['open'])
    range_ = candle['high'] - candle['low']
    upper_shadow = candle['high'] - max(candle['open'], candle['close'])
    
    return upper_shadow > 2 * body and body < 0.3 * range_

def is_morning_star(df, i):
    return (i >= 2 and 
            df['close'][i-2] < df['open'][i-2] and 
            abs(df['close'][i-1] - df['open'][i-1]) < (df['high'][i-2] - df['low'][i-2]) * 0.5 and 
            df['close'][i] > df['open'][i])

def is_evening_star(df, i):
    return (i >= 2 and 
            df['close'][i-2] > df['open'][i-2] and 
            abs(df['close'][i-1] - df['open'][i-1]) < (df['high'][i-2] - df['low'][i-2]) * 0.5 and 
            df['close'][i] < df['open'][i])

def is_piercing_pattern(df, i):
    return (i > 0 and 
            df['close'][i-1] < df['open'][i-1] and 
            df['close'][i] > df['open'][i] and 
            df['close'][i] > (df['close'][i-1] + df['open'][i-1]) / 2)

def is_dark_cloud_cover(df, i):
    return (i > 0 and 
            df['close'][i-1] > df['open'][i-1] and 
            df['close'][i] < df['open'][i] and 
            df['close'][i] < (df['close'][i-1] + df['open'][i-1]) / 2)

def is_spinning_top(candle):
    body = abs(candle['close'] - candle['open'])
    range_ = candle['high'] - candle['low']
    
    return body < 0.3 * range_

def detect_patterns(df):
    patterns = []

    for i in range(len(df)):
        candle = df.iloc[i]
        if is_hammer(candle):
            patterns.append('Hammer')
        elif is_doji(candle):
            patterns.append('Doji')
        elif i > 0 and is_engulfing(df, i):
            patterns.append('Engulfing')
        elif is_shooting_star(candle):
            patterns.append('Shooting Star')
        elif i > 1 and is_morning_star(df, i):
            patterns.append('Morning Star')
        elif i > 1 and is_evening_star(df, i):
            patterns.append('Evening Star')
        elif i > 0 and is_piercing_pattern(df, i):
            patterns.append('Piercing Pattern')
        elif i > 0 and is_dark_cloud_cover(df, i):
            patterns.append('Dark Cloud Cover')
        elif is_spinning_top(candle):
            patterns.append('Spinning Top')
        else:
            patterns.append('None')
    
    df['Pattern'] = patterns
    return df

# Sample Data
data = {
    'open': [1, 2, 1.5, 3, 2.5, 3.5, 3, 2, 1.8, 2.2],
    'high': [1.2, 2.5, 1.7, 3.5, 3, 4, 3.8, 2.5, 2.5, 2.8],
    'low': [0.8, 1.8, 1.2, 2.5, 2, 2.5, 2.8, 1.5, 1.6, 1.7],
    'close': [1.1, 2.4, 1.6, 2.8, 2.7, 3.2, 2.5, 1.9, 2.4, 2.1]
}

df = pd.DataFrame(data)

# Detect patterns
df_with_patterns = detect_patterns(df)
print(df_with_patterns[['open', 'high', 'low', 'close', 'Pattern']])