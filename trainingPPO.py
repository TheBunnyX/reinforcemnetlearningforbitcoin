# trainingPPO.py
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt
from CryptoTradingEnv import CryptoTradingEnv  # Ensure this matches the filename

# Load your dataset
#data = pd.read_csv('candlestick_data.csv', parse_dates=['Date'])
data = pd.read_csv('.\\dataset\\preprocess_candlestick_data.csv', parse_dates=['Date'])
env = CryptoTradingEnv(data)

# Check the environment
check_env(env)

# Train the model with GPU support
model = PPO('MlpPolicy', env, verbose=2)#, device='cuda')  # Use 'cuda' to specify GPU training
model.learn(total_timesteps = 10000)  # Increase timesteps for better training

# Save the model
model.save('ppo_crypto_trading_model')

# Load the trained model
model = PPO.load('ppo_crypto_trading_model')#, device='cuda')  # Load the model to GPU

# Reset the environment
obs, _ = env.reset()

# Run the trained model to get the actions and metrics
while True :
    action, _states = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    if done:
        break

# Now you can plot the results
def plot_signals_and_profit(data, actions, net_worths):
    plt.figure(figsize=(15, 8))

    # Create a buy/sell signal series
    buy_signals = [i for i in range(len(actions)) if actions[i] == 1]  # Buy action
    sell_signals = [i for i in range(len(actions)) if actions[i] == 2]  # Sell action

    # Plot price and buy/sell signals
    plt.subplot(2, 2, 1)
    plt.plot(data['Close'], label='Close Price', alpha=0.5)
    plt.scatter(buy_signals, data['Close'].iloc[buy_signals], marker='^', color='g', label='Buy Signal', alpha=1)
    plt.scatter(sell_signals, data['Close'].iloc[sell_signals], marker='v', color='r', label='Sell Signal', alpha=1)
    plt.title('Buy and Sell Signals')
    plt.legend()

    # Plot net worth
    plt.subplot(2, 2, 2)
    plt.plot(net_worths, label='Net Worth', color='b')
    plt.title('Net Worth Over Time')
    plt.xlabel('Steps')
    plt.ylabel('Net Worth')
    plt.legend()

    # Plot rewards
    plt.subplot(2, 2, 3)
    plt.plot(env.rewards, label='Rewards', color='purple')
    plt.title('Rewards Over Time')
    plt.xlabel('Steps')
    plt.ylabel('Rewards')
    plt.legend()

    # Calculate profit from the initial balance
    profits = [nw - env.initial_balance for nw in net_worths]
    plt.subplot(2, 2, 4)
    plt.plot(profits, label='Profit', color='orange')
    plt.title('Profit Over Time')
    plt.xlabel('Steps')
    plt.ylabel('Profit')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Call the plotting function
plot_signals_and_profit(data, env.buy_sell_actions, env.net_worths)