# test_model.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import A2C
from CryptoTradingEnv import CryptoTradingEnv  # Ensure this matches the filename

# Load your dataset
data = pd.read_csv('./dataset/candlestick_train.csv', parse_dates=['Date'])
env = CryptoTradingEnv(data)

# Load the trained model
model = A2C.load('a2c_crypto_trading_model')

# Reset the environment
obs, _ = env.reset()

# Initialize variables to track rewards, net worth, and actions
rewards = []
net_worths = []
buy_sell_actions = []  # 1 for buy, 2 for sell, 0 for hold
total_trades = 0

done = False

while not done:
    action, _states = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)

    rewards.append(reward)  # Store reward
    net_worths.append(env.net_worth)  # Store net worth
    buy_sell_actions.append(action)  # Store action

    # Count trades
    if action in [1, 2]:
        total_trades += 1

# Final calculations
begin_total_asset = env.initial_balance
end_total_asset = env.net_worth
total_reward = sum(rewards)
total_cost = begin_total_asset - end_total_asset + sum(buy_sell_actions) * env.trade_chunk_size

# Calculate Sharpe ratio
if len(rewards) > 0:
    sharpe_ratio = np.mean(rewards) / np.std(rewards) if np.std(rewards) > 0 else 0
else:
    sharpe_ratio = 0

# Print results
print(f"day: {env.current_step}, episode: 10")  # Change episode number as needed
print(f"begin_total_asset: {begin_total_asset:.2f}")
print(f"end_total_asset: {end_total_asset:.2f}")
print(f"total_reward: {total_reward:.2f}")
print(f"total_cost: {total_cost:.2f}")
print(f"total_trades: {total_trades}")
print(f"Sharpe: {sharpe_ratio:.3f}")

# # Visualization 1: Graph of Rewards
# plt.figure(figsize=(12, 6))
# plt.plot(np.arange(len(rewards)), rewards, label='Rewards', color='blue')
# plt.title('Rewards Over Time')
# plt.xlabel('Steps')
# plt.ylabel('Reward')
# plt.legend()
# plt.grid()
# plt.show()

# # Visualization 2: Buy and Sell Actions
# plt.figure(figsize=(12, 6))
# buy_signals = [i for i in range(len(buy_sell_actions)) if buy_sell_actions[i] == 1]
# sell_signals = [i for i in range(len(buy_sell_actions)) if buy_sell_actions[i] == 2]

# plt.plot(buy_signals, np.array(net_worths)[buy_signals], '^', markersize=10, color='g', label='Buy Signal')
# plt.plot(sell_signals, np.array(net_worths)[sell_signals], 'v', markersize=10, color='r', label='Sell Signal')
# plt.plot(net_worths, label='Net Worth', color='purple')
# plt.title('Buy and Sell Actions Over Time')
# plt.xlabel('Steps')
# plt.ylabel('Net Worth')
# plt.legend()
# plt.grid()
# plt.show()

# # Visualization 3: Graph of Profit and Loss
# profit_loss = np.array(net_worths) - env.initial_balance  # Calculate profit and loss
# plt.figure(figsize=(12, 6))
# plt.plot(profit_loss, label='Profit and Loss', color='orange')
# plt.axhline(0, color='gray', linestyle='--', linewidth=1)
# plt.title('Profit and Loss Over Time')
# plt.xlabel('Steps')
# plt.ylabel('Profit/Loss')
# plt.legend()
# plt.grid()
# plt.show()

# # Visualization 4: Graph of Net Worth
# plt.figure(figsize=(12, 6))
# plt.plot(net_worths, label='Net Worth', color='purple')
# plt.title('Net Worth Over Time')
# plt.xlabel('Steps')
# plt.ylabel('Net Worth')
# plt.legend()
# plt.grid()
# plt.show()

# Set up the figure and axes for 4 subplots
fig, axs = plt.subplots(2, 2, figsize=(16, 12))  # 2 rows, 2 columns

# Visualization 1: Graph of Rewards
axs[0, 0].plot(np.arange(len(rewards)), rewards, label='Rewards', color='blue')
axs[0, 0].set_title('Rewards Over Time')
axs[0, 0].set_xlabel('Steps')
axs[0, 0].set_ylabel('Reward')
axs[0, 0].legend()
axs[0, 0].grid()

# Visualization 2: Buy and Sell Actions
buy_signals = [i for i in range(len(buy_sell_actions)) if buy_sell_actions[i] == 1]
sell_signals = [i for i in range(len(buy_sell_actions)) if buy_sell_actions[i] == 2]

axs[0, 1].plot(buy_signals, np.array(net_worths)[buy_signals], '^', markersize=10, color='g', label='Buy Signal')
axs[0, 1].plot(sell_signals, np.array(net_worths)[sell_signals], 'v', markersize=10, color='r', label='Sell Signal')
axs[0, 1].plot(net_worths, label='Net Worth', color='purple')
axs[0, 1].set_title('Buy and Sell Actions Over Time')
axs[0, 1].set_xlabel('Steps')
axs[0, 1].set_ylabel('Net Worth')
axs[0, 1].legend()
axs[0, 1].grid()

# Visualization 3: Graph of Profit and Loss
profit_loss = np.array(net_worths) - env.initial_balance  # Calculate profit and loss
axs[1, 0].plot(profit_loss, label='Profit and Loss', color='orange')
axs[1, 0].axhline(0, color='gray', linestyle='--', linewidth=1)
axs[1, 0].set_title('Profit and Loss Over Time')
axs[1, 0].set_xlabel('Steps')
axs[1, 0].set_ylabel('Profit/Loss')
axs[1, 0].legend()
axs[1, 0].grid()

# Visualization 4: Graph of Net Worth
axs[1, 1].plot(net_worths, label='Net Worth', color='purple')
axs[1, 1].set_title('Net Worth Over Time')
axs[1, 1].set_xlabel('Steps')
axs[1, 1].set_ylabel('Net Worth')
axs[1, 1].legend()
axs[1, 1].grid()

# Adjust layout for better spacing
plt.tight_layout()
plt.show()

