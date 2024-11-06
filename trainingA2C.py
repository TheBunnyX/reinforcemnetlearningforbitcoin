# trainingA2C.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
from CryptoTradingEnv import CryptoTradingEnv  # Ensure this matches the filename

# Load your dataset
data = pd.read_csv('./dataset/candlestick_train.csv', parse_dates=['Date'])
env = CryptoTradingEnv(data)

# Check the environment
check_env(env)

# Set the number of training timesteps and learning parameters
total_timesteps = 10000  # Increased timesteps for better training
learning_rate = 0.0007  # Adjusted learning rate for stability
gamma = 0.99  # Discount factor

# Train the model with GPU support
model = A2C(
    'MlpPolicy', 
    env, 
    verbose=2,
    learning_rate=learning_rate,
    gamma=gamma,
    n_steps=5,  # Number of steps to run in each environment before updating
    ent_coef=0.01,  # Coefficient for the entropy term
    vf_coef=0.5,    # Coefficient for the value function
    max_grad_norm=0.5,  # Clipping for the gradient
    gae_lambda=0.95,  # Generalized Advantage Estimation
)

# Learning process
model.learn(total_timesteps=total_timesteps)

# Save the model
model.save('a2c_crypto_trading_model')

# Load the trained model
model = A2C.load('a2c_crypto_trading_model')  # Load the model

# Reset the environment
obs, _ = env.reset()

# Initialize variables to calculate total assets and trades
total_trades = 0
begin_total_asset = env.initial_balance
total_reward = 0

done = False

# Evaluate the trained model
while not done:
    action, _states = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    total_reward += reward  # Sum the rewards for total reward
    total_trades += 1 if action in [1, 2] else 0  # Count trades

# Calculate end total asset
end_total_asset = env.net_worth

# Calculate Sharpe ratio
if len(env.rewards) > 0:
    sharpe_ratio = np.mean(env.rewards) / np.std(env.rewards) if np.std(env.rewards) > 0 else 0
else:
    sharpe_ratio = 0

# Print results
print(f"day: {env.current_step}, episode: 10")
print(f"begin_total_asset: {begin_total_asset:.2f}")
print(f"end_total_asset: {end_total_asset:.2f}")
print(f"total_reward: {total_reward:.2f}")
print(f"total_trades: {total_trades}")
print(f"Sharpe: {sharpe_ratio:.3f}")

