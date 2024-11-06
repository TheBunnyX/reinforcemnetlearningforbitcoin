# # CryptoTradingEnv.py
# import gymnasium as gym
# import numpy as np

# class CryptoTradingEnv(gym.Env):
#     def __init__(self, data, initial_balance=30, trading_fee=0.001):
#         super(CryptoTradingEnv, self).__init__()
#         self.data = data
#         self.initial_balance = initial_balance
#         self.trading_fee = trading_fee

#         self.current_step = 0
#         self.balance = initial_balance
#         self.crypto_held = 0
#         self.net_worth = initial_balance
#         self.last_net_worth = initial_balance

#         self.action_space = gym.spaces.Discrete(3)  # 0 = hold, 1 = buy, 2 = sell
#         self.observation_space = gym.spaces.Box(low=0, high=1, shape=(7,), dtype=np.float32)

#     def reset(self, seed = None):
#         if seed is not None:
#             np.random.seed(seed)  # Set the random seed for reproducibility (if needed)
#         self.current_step = 0
#         self.balance = self.initial_balance
#         self.crypto_held = 0
#         self.net_worth = self.initial_balance
#         self.last_net_worth = self.initial_balance
#         return self._next_observation(), {}

#     def _next_observation(self):
#         # Get market data for the current step
#         row = self.data.iloc[self.current_step]
#         obs = np.array([
#             row['Open'], row['High'], row['Low'], row['Close'], row['Volume'],row['Day'],
#             self.net_worth  # Adding the agent's current net worth as part of the state
#         ])
#         return (obs / np.max(obs)).astype(np.float32)  # Normalize observation and cast to float32

#     def step(self, action):
#         current_price = self.data.iloc[self.current_step]['Close']

#         # Hold action
#         if action == 0:
#             pass
#         # Buy action
#         elif action == 1:
#             if self.balance > 0:
#                 crypto_bought = self.balance / current_price
#                 self.crypto_held += crypto_bought * (1 - self.trading_fee)
#                 self.balance = 0
#         # Sell action
#         elif action == 2:
#             if self.crypto_held > 0:
#                 self.balance += self.crypto_held * current_price * (1 - self.trading_fee)
#                 self.crypto_held = 0

#         self.net_worth = self.balance + self.crypto_held * current_price

#         # Calculate reward as the change in net worth
#         reward = self.net_worth - self.last_net_worth
#         self.last_net_worth = self.net_worth

#         self.current_step += 1
#         done = self.current_step >= len(self.data) - 1

#         # Here, we assume 'terminated' is same as 'done' and 'truncated' is False
#         terminated = done
#         truncated = False

#         return self._next_observation(), reward, terminated, truncated, {}

#     def render(self):
#         print(f'Step: {self.current_step}, Net Worth: {self.net_worth}, Crypto Held: {self.crypto_held}, Balance: {self.balance}')
# # CryptoTradingEnv.py
import gymnasium as gym
import numpy as np

class CryptoTradingEnv(gym.Env):
    def __init__(self, data, initial_balance=30, trading_fee=0.001, trade_chunk_size=3):
        super(CryptoTradingEnv, self).__init__()
        self.data = data
        self.initial_balance = initial_balance
        self.trade_chunk_size = trade_chunk_size  # Amount per trade (e.g., 3 USD)
        self.trading_fee = trading_fee

        self.current_step = 0
        self.balance = initial_balance
        self.crypto_held = 0
        self.net_worth = initial_balance
        self.last_net_worth = initial_balance

        self.action_space = gym.spaces.Discrete(3)  # 0 = hold, 1 = buy, 2 = sell
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(7,), dtype=np.float32)

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)  # Set the random seed for reproducibility (if needed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.crypto_held = 0
        self.net_worth = self.initial_balance
        self.last_net_worth = self.initial_balance
        return self._next_observation(), {}

    def _next_observation(self):
        # Get market data for the current step
        row = self.data.iloc[self.current_step]
        obs = np.array([
            row['Open'], row['High'], row['Low'], row['Close'], row['Volume'], row['Day'],
            self.net_worth  # Adding the agent's current net worth as part of the state
        ])
        return (obs / np.max(obs)).astype(np.float32)  # Normalize observation and cast to float32

    def step(self, action):
        current_price = self.data.iloc[self.current_step]['Close']

        # Hold action
        if action == 0:
            pass
        # Buy action
        elif action == 1:
            # Ensure we have at least the trade_chunk_size in balance to buy
            if self.balance >= self.trade_chunk_size:
                crypto_bought = self.trade_chunk_size / current_price
                self.crypto_held += crypto_bought * (1 - self.trading_fee)
                self.balance -= self.trade_chunk_size
        # Sell action
        elif action == 2:
            if self.crypto_held > 0:
                self.balance += self.crypto_held * current_price * (1 - self.trading_fee)
                self.crypto_held = 0

        self.net_worth = self.balance + self.crypto_held * current_price

        # Calculate reward as the change in net worth
        reward = self.net_worth - self.last_net_worth
        self.last_net_worth = self.net_worth

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        # Here, we assume 'terminated' is same as 'done' and 'truncated' is False
        terminated = done
        truncated = False

        return self._next_observation(), reward, terminated, truncated, {}

    def render(self):
        print(f'Step: {self.current_step}, Net Worth: {self.net_worth}, Crypto Held: {self.crypto_held}, Balance: {self.balance}')