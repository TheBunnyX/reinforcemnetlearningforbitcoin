# # CryptoTradingEnv.py
# import gymnasium as gym
# import numpy as np

# class CryptoTradingEnv(gym.Env):
#     def __init__(self, data, initial_balance=100, trading_fee=0.001, trade_chunk_size=5):
#         super(CryptoTradingEnv, self).__init__()
#         self.data = data
#         self.initial_balance = initial_balance
#         self.trade_chunk_size = trade_chunk_size  # Amount per trade (e.g., 5 USD)
#         self.trading_fee = trading_fee

#         self.current_step = 0
#         self.balance = initial_balance
#         self.crypto_held = 0
#         self.net_worth = initial_balance
#         self.last_net_worth = initial_balance

#         self.action_space = gym.spaces.Discrete(3)  # 0 = hold, 1 = buy, 2 = sell
#         self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(7,), dtype=np.float32)

#         # Lists to store values for plotting
#         self.buy_sell_actions = []
#         self.rewards = []
#         self.net_worths = []

#     def reset(self, seed= None):
#         if seed is not None:
#             np.random.seed(seed)  # Set the random seed for reproducibility (if needed)
#         self.current_step = 0
#         self.balance = self.initial_balance
#         self.crypto_held = 0
#         self.net_worth = self.initial_balance
#         self.last_net_worth = self.initial_balance
        
#         # Reset and clear stored data
#         self.buy_sell_actions.clear()
#         self.rewards.clear()
#         self.net_worths.clear()

#         return self._next_observation(), {}

#     def _next_observation(self):
#         # Get market data for the current step
#         row = self.data.iloc[self.current_step]
#         obs = np.array([
#             row['Open'], row['High'], row['Low'], row['Close'], row['Volume'], row['Day'],
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
#             # Ensure we have at least the trade_chunk_size in balance to buy
#             if self.balance >= self.trade_chunk_size:
#                 crypto_bought = self.trade_chunk_size / current_price
#                 self.crypto_held += crypto_bought * (1 - self.trading_fee)
#                 self.balance -= self.trade_chunk_size
#                 self.buy_sell_actions.append(1)  # Append Buy action
#         # Sell action
#         elif action == 2:
#             # Ensure we can only sell if we have crypto to sell
#             if self.crypto_held > 0:
#                 self.balance += self.crypto_held * current_price * (1 - self.trading_fee)
#                 self.crypto_held = 0
#                 self.buy_sell_actions.append(2)  # Append Sell action
#             else:
#                 # If no crypto is held, treat as hold (no action) or penalize
#                 self.buy_sell_actions.append(0)  # Treat as Hold action

#         self.net_worth = self.balance + self.crypto_held * current_price

#         # Calculate reward as the change in net worth
#         reward = self.net_worth - self.last_net_worth
#         self.last_net_worth = self.net_worth

#         self.current_step += 1
#         done = self.current_step >= len(self.data) - 1

#         # Store reward
#         self.rewards.append(reward)
        
#         # Store net worth
#         self.net_worths.append(self.net_worth)

#         # Here, we assume 'terminated' is same as 'done' and 'truncated' is False
#         terminated = done
#         truncated = False

#         return self._next_observation(), reward, terminated, truncated, {}

#     def render(self):
#         print(f'Step: {self.current_step}, Net Worth: {self.net_worth}, Crypto Held: {self.crypto_held}, Balance: {self.balance}')
import gymnasium as gym
import numpy as np

class CryptoTradingEnv(gym.Env):
    def __init__(self, data, initial_balance=100, trading_fee=0.001, trade_chunk_size=5, max_steps=None):
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
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(7,), dtype=np.float32)

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
        self.last_net_worth = self.initial_balance
        
        self.buy_sell_actions.clear()
        self.rewards.clear()
        self.net_worths.clear()

        return self._next_observation(), {}

    def _next_observation(self):
        row = self.data.iloc[self.current_step]
        obs = np.array([
            row['Open'], row['High'], row['Low'], row['Close'], row['Volume'], row['Day'],
            self.net_worth
        ])
        # Normalize data for stability
        obs = obs / np.max(self.data[['Open', 'High', 'Low', 'Close', 'Volume']].values)
        return obs.astype(np.float32)

    def step(self, action):
        current_price = self.data.iloc[self.current_step]['Close']
        reward = 0

        # Execute actions
        if action == 0:  # Hold
            reward -= 0.01  # Penalize holding without action for too long
        elif action == 1:  # Buy
            if self.balance >= self.trade_chunk_size:
                crypto_bought = self.trade_chunk_size / current_price
                self.crypto_held += crypto_bought * (1 - self.trading_fee)
                self.balance -= self.trade_chunk_size
                self.buy_sell_actions.append(1)
                reward += 0.1  # Reward for successful buy
            else:
                reward -= 0.1  # Penalize if attempted to buy without balance
        elif action == 2:  # Sell
            if self.crypto_held > 0:
                self.balance += self.crypto_held * current_price * (1 - self.trading_fee)
                self.crypto_held = 0
                self.buy_sell_actions.append(2)
                reward += 0.1  # Reward for successful sell
            else:
                reward -= 0.1  # Penalize sell when no crypto is held

        self.net_worth = self.balance + self.crypto_held * current_price
        reward += (self.net_worth - self.last_net_worth) * 0.1  # Incentivize increasing net worth

        self.last_net_worth = self.net_worth
        self.current_step += 1
        done = self.current_step >= self.max_steps

        # Store rewards and net worths
        self.rewards.append(reward)
        self.net_worths.append(self.net_worth)

        terminated = done
        truncated = False

        return self._next_observation(), reward, terminated, truncated, {}

    def render(self):
        print(f'Step: {self.current_step}, Net Worth: {self.net_worth}, Crypto Held: {self.crypto_held}, Balance: {self.balance}')
