import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3 import PPO
from CryptoTradingEnv import CryptoTradingEnv

# Load the test dataset
data = pd.read_csv('./dataset/candlestick_test.csv', parse_dates=['Date'])

data = data.iloc[0:6440] 
env = CryptoTradingEnv(data)
# Load the trained PPO model
best_model_path = './models/best_model.zip'  # Path to your saved model
model = PPO.load(best_model_path)

# Reset the environment for testing
observation, _ = env.reset()
done = False
total_reward = 0

# Run the model in the test environment
while not done:
    action, _ = model.predict(observation, deterministic=True)  # Get the action from the model
    observation, reward, terminated, truncated, info = env.step(action)  # Step the environment
    total_reward += reward  # Accumulate the reward
    done = terminated or truncated  # Check if the episode is done

    # Optionally render the environment
    env.render()

# Print the total reward received during testing
print(f"Total reward from PPO model on test data: {total_reward}")