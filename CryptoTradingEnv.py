# #CryptoTradingEnv.py
# import gymnasium as gym
# import numpy as np
# import pandas as pd

# class CryptoTradingEnv(gym.Env):
#     def __init__(self, data, initial_balance=100, trading_fee=0.001, trade_chunk_size=5, max_steps=None):
#         super(CryptoTradingEnv, self).__init__()
#         self.data = data.reset_index(drop=True)
#         self.initial_balance = initial_balance
#         self.trade_chunk_size = trade_chunk_size
#         self.trading_fee = trading_fee

#         self.current_step = 0
#         self.balance = initial_balance
#         self.crypto_held = 0
#         self.net_worth = initial_balance
#         self.last_net_worth = initial_balance

#         self.max_steps = max_steps if max_steps is not None else len(data) - 1
#         self.action_space = gym.spaces.Discrete(3)  # 0 = hold, 1 = buy, 2 = sell
#         self.observation_space = gym.spaces.Box(low=0, high=1, shape=(15,), dtype=np.float32)

#         # Tracking for analysis
#         self.buy_sell_actions = []
#         self.rewards = []
#         self.net_worths = []
        
#         # Track render data for CSV
#         self.render_data = []

#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)
#         self.current_step = 0
#         self.balance = self.initial_balance
#         self.crypto_held = 0
#         self.net_worth = self.initial_balance
#         self.last_net_worth = self.initial_balance

#         self.buy_sell_actions.clear()
#         self.rewards.clear()
#         self.net_worths.clear()
#         self.render_data.clear()  # Clear previous render data

#         return self._next_observation(), {}

#     def _next_observation(self):
#         row = self.data.iloc[self.current_step]
#         obs = np.array([
#             row['Open'], row['High'], row['Low'], row['Close'], row['Volume'], row['Day'],
#             row['MA_20'], row['MA_50'], row['RSI'], row['MACD'], row['MACD_Signal'],
#             row['MACD_Diff'], row['Bollinger_Upper'], row['Bollinger_Lower'],
#             self.net_worth  # Include net_worth
#         ])

#         max_values = np.max(self.data[['Open', 'High', 'Low', 'Close', 'Volume', 'Day',
#                                        'MA_20', 'MA_50', 'RSI', 'MACD',
#                                        'MACD_Signal', 'MACD_Diff',
#                                        'Bollinger_Upper', 'Bollinger_Lower']].values, axis=0)
#         max_values = np.append(max_values, self.initial_balance)

#         obs = obs / max_values
#         return obs.astype(np.float32)

#     def step(self, action):
#         self.current_step += 1
#         terminated = self.current_step >= self.max_steps
#         truncated = False
#         current_price = self.data.iloc[self.current_step]['Close']

#         trade_amount = self.trade_chunk_size / current_price

#         if action == 1:  # Buy
#             if self.balance >= self.trade_chunk_size * (1 + self.trading_fee):
#                 self.crypto_held += trade_amount
#                 self.balance -= self.trade_chunk_size * (1 + self.trading_fee)
#                 self.buy_sell_actions.append((self.current_step, 'buy', current_price))

#         elif action == 2:  # Sell
#             if self.crypto_held >= trade_amount:
#                 self.crypto_held -= trade_amount
#                 self.balance += self.trade_chunk_size * (1 - self.trading_fee)
#                 self.buy_sell_actions.append((self.current_step, 'sell', current_price))

#         self.net_worth = self.balance + self.crypto_held * current_price
#         reward = self.net_worth - self.last_net_worth
#         self.last_net_worth = self.net_worth

#         self.net_worths.append(self.net_worth)
#         self.rewards.append(reward)
        
#         # Append render data
#         self.render_data.append({
#             'Step': self.current_step,
#             'Net Worth': self.net_worth,
#             'Crypto Held': self.crypto_held,
#             'Balance': self.balance,
#             'Action': ['hold', 'buy', 'sell'][action]
#         })

#         return self._next_observation(), reward, terminated, truncated, {}

#     def render(self, save_path="render_output.csv"):
#         # Print to console
#         print(f'Step: {self.current_step}, Net Worth: {self.net_worth:.8f}, '
#               f'Crypto Held: {self.crypto_held:.8f}, Balance: {self.balance:.8f}')
        
#         # Save render data to CSV
#         if save_path:
#             pd.DataFrame(self.render_data).to_csv(save_path, index=False)
# import gymnasium as gym
# import numpy as np
# import pandas as pd

# class CryptoTradingEnv(gym.Env):
#     # Define action constants for clarity
#     HOLD = 0
#     BUY = 1
#     SELL = 2

#     def __init__(self, data, initial_balance=100, trading_fee=0.001, trade_chunk_size=5, max_steps=None):
#         super(CryptoTradingEnv, self).__init__()
#         self.data = data.reset_index(drop=True)
#         self.initial_balance = initial_balance
#         self.trade_chunk_size = trade_chunk_size
#         self.trading_fee = trading_fee

#         self.current_step = 0
#         self.balance = initial_balance
#         self.crypto_held = 0
#         self.net_worth = initial_balance
#         self.last_net_worth = initial_balance

#         self.max_steps = max_steps if max_steps is not None else len(data) - 1
#         self.action_space = gym.spaces.Discrete(3)  # 0 = hold, 1 = buy, 2 = sell
#         self.observation_space = gym.spaces.Box(low=0, high=1, shape=(15,), dtype=np.float32)

#         # Tracking for analysis
#         self.buy_sell_actions = []
#         self.rewards = []
#         self.net_worths = []
        
#         # Track render data for CSV
#         self.render_data = []

#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)
#         self.current_step = 0
#         self.balance = self.initial_balance
#         self.crypto_held = 0
#         self.net_worth = self.initial_balance
#         self.last_net_worth = self.initial_balance

#         self.buy_sell_actions.clear()
#         self.rewards.clear()
#         self.net_worths.clear()
#         self.render_data.clear()  # Clear previous render data

#         return self._next_observation(), {}

#     def _next_observation(self):
#         row = self.data.iloc[self.current_step]
#         obs = np.array([
#             row['Open'], row['High'], row['Low'], row['Close'], row['Volume'], row['Day'],
#             row['MA_20'], row['MA_50'], row['RSI'], row['MACD'], row['MACD_Signal'],
#             row['MACD_Diff'], row['Bollinger_Upper'], row['Bollinger_Lower'],
#             self.net_worth  # Include net_worth
#         ])

#         max_values = np.max(self.data[['Open', 'High', 'Low', 'Close', 'Volume', 'Day',
#                                        'MA_20', 'MA_50', 'RSI', 'MACD',
#                                        'MACD_Signal', 'MACD_Diff',
#                                        'Bollinger_Upper', 'Bollinger_Lower']
#                                        ].values, axis=0)
#         max_values = np.append(max_values, self.initial_balance)

#         obs = obs / max_values
#         return obs.astype(np.float32)

#     def step(self, action):
#         self.current_step += 1
#         terminated = self.current_step >= self.max_steps
#         truncated = False
#         current_price = self.data.iloc[self.current_step]['Close']

#         # Calculate trade amount based on current price
#         trade_amount = self.trade_chunk_size / current_price

#         # Logging the action taken
#         print(f'Step: {self.current_step}, Action: {action}, Current Price: {current_price:.2f}')

#         if action == self.BUY:  # Buy
#             if self.balance >= self.trade_chunk_size * (1 + self.trading_fee):
#                 self.crypto_held += trade_amount
#                 self.balance -= self.trade_chunk_size * (1 + self.trading_fee)
#                 self.buy_sell_actions.append((self.current_step, 'buy', current_price))
#                 print(f'Bought: {trade_amount:.8f} at price {current_price:.2f}')
#             else:
#                 print(f'Failed to buy: Not enough money. Current holding: {self.balance:.8f}')

#         elif action == self.SELL:  # Sell
#             if self.crypto_held >= trade_amount:
#                 # Calculate minimum sell price to cover fee
#                 minimum_sell_price = (self.trade_chunk_size / trade_amount) * (1 + self.trading_fee)

#                 if current_price >= minimum_sell_price:
#                     self.crypto_held -= trade_amount
#                     self.balance += self.trade_chunk_size * (1 - self.trading_fee)
#                     self.buy_sell_actions.append((self.current_step, 'sell', current_price))
#                     print(f'Sold: {trade_amount:.8f} at price {current_price:.2f}')
#                 else:
#                     print(f'Failed to sell: Price too low to cover fee. Current Price: {current_price:.2f}, '
#                         f'Minimum Required Sell Price: {minimum_sell_price:.2f}')
#             else:
#                 print(f'Failed to sell: Not enough crypto held. Current holding: {self.crypto_held:.8f}')

#         # Update net worth
#         self.net_worth = self.balance + self.crypto_held * current_price
#         reward = self.net_worth - self.last_net_worth
#         self.last_net_worth = self.net_worth 

#         self.net_worths.append(self.net_worth)
#         self.rewards.append(reward)

#         # Append render data
#         self.render_data.append({
#             'Step': self.current_step,
#             'Net Worth': self.net_worth,
#             'Crypto Held': self.crypto_held,
#             'Balance': self.balance,
#             'Action': ['hold', 'buy', 'sell'][action]
#         })

#         return self._next_observation(), reward, terminated, truncated, {}

#     def render(self, save_path=None):
#         # Print to console
#         print(f'Step: {self.current_step}, Net Worth: {self.net_worth:.8f}, '
#               f'Crypto Held: {self.crypto_held:.8f}, Balance: {self.balance:.8f}')
        
#         # Save render data to CSV if a path is provided
#         if save_path:
#             pd.DataFrame(self.render_data).to_csv(save_path, index=False)
import gymnasium as gym
import numpy as np
import pandas as pd

class CryptoTradingEnv(gym.Env):
    # Define action constants for clarity
    HOLD = 0
    BUY = 1
    SELL = 2

    def __init__(self, data, initial_balance=20, trading_fee=0.001, trade_chunk_size=1, max_steps=None):
        super(CryptoTradingEnv, self).__init__()
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.trade_chunk_size = trade_chunk_size
        self.trading_fee = trading_fee

        self.current_step = 0
        self.balance = initial_balance
        self.crypto_held = 0
        self.net_worth = initial_balance
        self.last_net_worth = initial_balance

        self.max_steps = max_steps if max_steps is not None else len(data) - 1
        self.action_space = gym.spaces.Discrete(3)  # 0 = hold, 1 = buy, 2 = sell
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(15,), dtype=np.float32)

        # Tracking for analysis
        self.buy_sell_actions = []
        self.rewards = []
        self.net_worths = []
        
        # Track render data for CSV
        self.render_data = []

        # New attributes for tracking failed buy attempts
        self.failed_buy_attempts = 0  # Counter for failed buy attempts
        self.failed_buy_threshold = 2  # Threshold for forced sell

        # New attributes for tracking consecutive holds
        self.consecutive_holds = 0  # Counter for consecutive hold actions
        self.hold_penalty_threshold = 30  # Threshold for hold penalty
        self.hold_penalty = -100  # Penalty for holding too long

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.crypto_held = 0
        self.net_worth = self.initial_balance
        self.last_net_worth = self.initial_balance

        self.buy_sell_actions.clear()
        self.rewards.clear()
        self.net_worths.clear()
        self.render_data.clear()  # Clear previous render data
        
        # Reset the failed buy counter and consecutive holds
        self.failed_buy_attempts = 0
        self.consecutive_holds = 0

        return self._next_observation(), {}

    def _next_observation(self):
        row = self.data.iloc[self.current_step]
        obs = np.array([  
            row['Open'], row['High'], row['Low'], row['Close'], row['Volume'], row['Day'],
            row['MA_20'], row['MA_50'], row['RSI'], row['MACD'], row['MACD_Signal'],
            row['MACD_Diff'], row['Bollinger_Upper'], row['Bollinger_Lower'],
            self.net_worth  # Include net_worth
        ])

        max_values = np.max(self.data[['Open', 'High', 'Low', 'Close', 'Volume', 'Day',
                                       'MA_20', 'MA_50', 'RSI', 'MACD',
                                       'MACD_Signal', 'MACD_Diff',
                                       'Bollinger_Upper', 'Bollinger_Lower']
                                       ].values, axis=0)
        max_values = np.append(max_values, self.initial_balance)

        obs = obs / max_values
        return obs.astype(np.float32)

    def step(self, action):
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        current_price = self.data.iloc[self.current_step]['Close']

        # Calculate trade amount based on current price
        trade_amount = self.trade_chunk_size / current_price

        # Logging the action taken
        print(f'Step: {self.current_step}, Action: {action}, Current Price: {current_price:.2f}')

        # Initialize reward and penalties
        reward = 0

        # Check for hold actions
        if action == self.HOLD:
            self.consecutive_holds += 1  # Increment the hold counter
            if self.consecutive_holds > self.hold_penalty_threshold:
                reward += self.hold_penalty  # Apply penalty for too many holds
                print(f'Penalty applied for holding too long: {self.hold_penalty}')
            if self.consecutive_holds > self.hold_penalty_threshold-7:
                reward += self.hold_penalty-5  # Apply penalty for too many holds
                print(f'Penalty applied for holding too long: {self.hold_penalty}')

        else:
            self.consecutive_holds = 0  # Reset the hold counter if not holding

        if action == self.BUY:  # Buy
            if self.balance >= self.trade_chunk_size * (1 + self.trading_fee):
                self.crypto_held += trade_amount
                self.balance -= self.trade_chunk_size * (1 + self.trading_fee)
                self.buy_sell_actions.append((self.current_step, 'buy', current_price))
                print(f'Bought: {trade_amount:.8f} at price {current_price:.2f}')
                self.failed_buy_attempts = 0  # Reset the failed buy counter
            else:
                print(f'Failed to buy: Not enough money. Current holding: {self.balance:.8f}')
                self.failed_buy_attempts += 1  # Increment the failed buy counter
                reward -= 15  # Penalty for failed buy

                # Check if the failed buy threshold is reached for forced selling
                if self.failed_buy_attempts > self.failed_buy_threshold:
                    # Force sell all crypto held
                    if self.crypto_held > 0:
                        self.balance += self.crypto_held * current_price * (1 - self.trading_fee)
                        print(f'Forced Sell: Sold all crypto at price {current_price:.2f}')
                        self.buy_sell_actions.append((self.current_step, 'forced sell', current_price))
                        self.crypto_held = 0  # Clear crypto held
                        self.failed_buy_attempts = 0  # Reset the counter after forced sell

        elif action == self.SELL:  # Sell
            if self.crypto_held >= trade_amount:
                minimum_sell_price = (self.trade_chunk_size / trade_amount) * (1 + self.trading_fee+0.001)

                if current_price >= minimum_sell_price:
                    self.crypto_held -= trade_amount
                    self.balance += self.trade_chunk_size * (1 - self.trading_fee)
                    self.buy_sell_actions.append((self.current_step, 'sell', current_price))
                    print(f'Sold: {trade_amount:.8f} at price {current_price:.2f}')
                else:
                    print(f'Failed to sell: Price too low to cover fee. Current Price: {current_price:.2f}, '
                          f'Minimum Required Sell Price: {minimum_sell_price:.2f}')
                    reward -= 15  # Penalty for failed sell
            else:
                print(f'Failed to sell: Not enough crypto held. Current holding: {self.crypto_held:.8f}')
                reward -= 15  # Penalty for failed sell

        # Update net worth
        self.net_worth = self.balance + self.crypto_held * current_price
        current_reward = self.net_worth - self.last_net_worth
        self.last_net_worth = self.net_worth 

        current_reward += reward

        self.net_worths.append(self.net_worth)
        self.rewards.append(reward)

        # Append render data
        self.render_data.append({
            'Step': self.current_step,
            'Net Worth': self.net_worth,
            'Crypto Held': self.crypto_held,
            'Balance': self.balance,
            'Action': ['hold', 'buy', 'sell'][action]
        })

        return self._next_observation(), current_reward, terminated, truncated, {}

    def render(self, save_path=None):
        # Print to console
        print(f'Step: {self.current_step}, Net Worth: {self.net_worth:.8f}, '
              f'Crypto Held: {self.crypto_held:.8f}, Balance: {self.balance:.8f}')
        
        # Save render data to CSV if a path is provided
        if save_path:
            pd.DataFrame(self.render_data).to_csv(save_path, index=False)
