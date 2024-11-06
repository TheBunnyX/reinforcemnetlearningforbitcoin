# # CryptoTradingEnv.py
import gymnasium as gym
import numpy as np

class CryptoTradingEnv(gym.Env):
    def __init__(self, data, initial_balance=100, trading_fee=0.001, trade_chunk_size=10, max_steps=None):
        super(CryptoTradingEnv, self).__init__()
        self.data = data
        self.initial_balance = initial_balance
        self.trade_chunk_size = trade_chunk_size  # Amount per trade
        self.trading_fee = trading_fee

        self.current_step = 0
        self.balance = initial_balance
        self.crypto_held = 0
        self.net_worth = initial_balance
        self.last_net_worth = initial_balance

        self.max_steps = max_steps if max_steps is not None else len(data) - 1
        self.action_space = gym.spaces.Discrete(3)  # 0 = hold, 1 = buy, 2 = sell
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(15,), dtype=np.float32)

        # Store values for analysis
        self.buy_sell_actions = []
        self.rewards = []
        self.net_worths = []

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.crypto_held = 0
        self.net_worth = self.initial_balance
        
        self.buy_sell_actions.clear()
        self.rewards.clear()
        self.net_worths.clear()

        return self._next_observation(), {}

    def _next_observation(self):
        row = self.data.iloc[self.current_step]
        obs = np.array([
            row['Open'], row['High'], row['Low'], row['Close'], row['Volume'], row['Day'],
            row['MA_20'], row['MA_50'], row['RSI'], row['MACD'], row['MACD_Signal'], 
            row['MACD_Diff'], row['Bollinger_Upper'], row['Bollinger_Lower'],
            self.net_worth  # Make sure to include net_worth
        ])
        
        # Adjust max_values to include all features in obs
        max_values = np.max(self.data[['Open', 'High', 'Low', 'Close', 'Volume', 'Day',
                                        'MA_20', 'MA_50', 'RSI', 'MACD', 
                                        'MACD_Signal', 'MACD_Diff', 
                                        'Bollinger_Upper', 'Bollinger_Lower']].values, axis=0)
        # Include net_worth max for normalization
        max_values = np.append(max_values, self.initial_balance)  # or set a fixed max for net_worth

        # Normalize the observation array
        obs = obs / max_values
        return obs.astype(np.float32)
    
    def step(self, action):
        # Advance step and check if episode is done
        self.current_step += 1
        terminated = self.current_step >= len(self.data) - 1  # Assuming `self.data` has the stock/crypto data
        truncated = False  # Set truncated to False unless you want to add a custom stopping condition
        current_price = self.data.iloc[self.current_step]['Close']

        # Initialize reward
        reward = 0
        
        # Action logic
        if action == 0:  # Buy
            if self.balance >= current_price:  # Only buy if there's enough balance
                self.crypto_held += 1
                self.balance -= current_price
        elif action == 1:  # Sell
            if self.crypto_held > 0:  # Only sell if there's crypto to sell
                self.crypto_held -= 1
                self.balance += current_price

        # Calculate current portfolio value (balance + value of held crypto)
        self.net_worth = self.balance + self.crypto_held * current_price
        reward = self.net_worth - self.initial_balance  # Reward based on change in net worth

        # Add the updated net worth and reward to the tracking lists
        self.net_worths.append(self.net_worth)
        self.rewards.append(reward)

        # Return the next state, reward, terminated, truncated status, and additional info
        return self._next_observation(), reward, terminated, truncated, {}

    def render(self):
        print(f'Step: {self.current_step}, Net Worth: {self.net_worth}, Crypto Held: {self.crypto_held}, Balance: {self.balance}')