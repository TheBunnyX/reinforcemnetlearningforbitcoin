import numpy as np
import pandas as pd
import gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from CryptoTradingEnv import CryptoTradingEnv

# Load the dataset
data = pd.read_csv('./dataset/candlestick_train.csv', parse_dates=['Date'])

# Select rows from 0 to 1439
data_subset = data.iloc[0:1440]  # This will include rows 0 to 1439

# Instantiate the custom environment with the sliced data
env = CryptoTradingEnv(data_subset)

# Define the RecurrentPPO model with MlpLstmPolicy
model = RecurrentPPO(
    "MlpLstmPolicy",  # Use LSTM policy
    env,
    verbose=2,
    learning_rate=1e-4,
    n_steps=2048,
    batch_size=128,
    n_epochs=30,
    clip_range=0.2,
    gamma=0.99,
    device="cpu",
    seed=42
)

# Wrap the evaluation environment with Monitor
eval_env = Monitor(CryptoTradingEnv(data_subset))
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='./models/',
    log_path='./logs/',
    eval_freq=1000,  # Evaluate every 1000 timesteps
    deterministic=True,
    render=False
)

# Train the RecurrentPPO model
print("Training RecurrentPPO with LSTM...")
model.learn(total_timesteps=2000, callback=eval_callback, progress_bar=True)

print("RecurrentPPO model with LSTM trained!")

# Load the best model
best_model_path = './models/best_model.zip'
model = RecurrentPPO.load(best_model_path)

# Testing the trained RecurrentPPO model in the environment
observation, _ = env.reset()
done = False
total_reward = 0
lstm_states = None  # For RecurrentPPO, we need to track LSTM states
print("Testing the LSTM-based RecurrentPPO model...")

while not done:
    action, lstm_states = model.predict(observation, state=lstm_states, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated

    # Render the environment (or save render data if preferred)
    env.render()

print(f"Total reward from LSTM-based RecurrentPPO model: {total_reward}")
