#CryptoTradingEnv.py
import gymnasium as gym
import numpy as np
import pandas as pd

class CryptoTradingEnv(gym.Env):
    def __init__(self, data, initial_balance=1000, trading_fee=0.001, trade_chunk_size=10, max_steps=None):
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
                                       'Bollinger_Upper', 'Bollinger_Lower']].values, axis=0)
        max_values = np.append(max_values, self.initial_balance)

        obs = obs / max_values
        return obs.astype(np.float32)

    def step(self, action):
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        current_price = self.data.iloc[self.current_step]['Close']

        trade_amount = self.trade_chunk_size / current_price

        if action == 1:  # Buy
            if self.balance >= self.trade_chunk_size * (1 + self.trading_fee):
                self.crypto_held += trade_amount
                self.balance -= self.trade_chunk_size * (1 + self.trading_fee)
                self.buy_sell_actions.append((self.current_step, 'buy', current_price))

        elif action == 2:  # Sell
            if self.crypto_held >= trade_amount:
                self.crypto_held -= trade_amount
                self.balance += self.trade_chunk_size * (1 - self.trading_fee)
                self.buy_sell_actions.append((self.current_step, 'sell', current_price))

        self.net_worth = self.balance + self.crypto_held * current_price
        reward = self.net_worth - self.last_net_worth
        self.last_net_worth = self.net_worth

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

        return self._next_observation(), reward, terminated, truncated, {}

    def render(self, save_path="render_output.csv"):
        # Print to console
        print(f'Step: {self.current_step}, Net Worth: {self.net_worth:.2f}, '
              f'Crypto Held: {self.crypto_held:.4f}, Balance: {self.balance:.2f}')
        
        # Save render data to CSV
        if save_path:
            pd.DataFrame(self.render_data).to_csv(save_path, index=False)
