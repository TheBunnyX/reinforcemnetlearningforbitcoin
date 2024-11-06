# #trainingppo.py
# import numpy as np
# import pandas as pd
# import gymnasium as gym
# from stable_baselines3 import PPO
# from CryptoTradingEnv import CryptoTradingEnv

# # Load the dataset
# data = pd.read_csv('./dataset/candlestick_train.csv', parse_dates=['Date'])

# # Instantiate the custom environment with the loaded data
# #env = CryptoTradingEnv(data)

# # Step 2: Select rows from 0 to 1439
# data_subset = data.iloc[0:1440]  # This will include rows 0 to 1439

# # Step 3: Instantiate the custom environment with the sliced data
# env = CryptoTradingEnv(data_subset)

# # Define the PPO model
# model = PPO("MlpPolicy", env, verbose = 2)

# # Train the PPO model
# print("Training PPO...")
# model.learn(total_timesteps=10000)  # Adjust timesteps based on your requirements
# model.save("PPO_crypto_trading_model")

# print("PPO model trained and saved!")

# # Testing the trained PPO model in the environment
# observation, _ = env.reset()
# done = False
# total_reward = 0

# while not done:
#     action, _ = model.predict(observation, deterministic=True)
#     observation, reward, terminated, truncated, info = env.step(action)
#     total_reward += reward
#     done = terminated or truncated

#     # Render the environment (or save render data if preferred)
#     env.render()

# print(f"Total reward from PPO model: {total_reward}")
import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from CryptoTradingEnv import CryptoTradingEnv

# Load the dataset
data = pd.read_csv('./dataset/candlestick_train.csv', parse_dates=['Date'])

# Select rows from 0 to 1439
data_subset = data.iloc[0:1440]  # This will include rows 0 to 1439

# Instantiate the custom environment with the sliced data
env = CryptoTradingEnv(data_subset)

# Define the PPO model
#model = PPO("MlpPolicy", env, verbose=2)
model = PPO("MlpPolicy", env, verbose=2,
             #learning_rate=1e-4,
             learning_rate=1e-4,
             n_steps=2048, 
             batch_size=128, 
             n_epochs=30, 
             clip_range=0.2,#clip_range=0.1 
             gamma=0.99, 
             device="cpu",
             seed=42
             )

# Wrap the evaluation environment with Monitor
eval_env = Monitor(CryptoTradingEnv(data_subset))  # Create a Monitor-wrapped evaluation environment
eval_callback = EvalCallback(eval_env, 
                              best_model_save_path='./models/', 
                              log_path='./logs/', 
                              eval_freq=1000,  # Evaluate every 1000 timesteps
                              deterministic=True,
                              render=False)  # Don't render during evaluation

# Train the PPO model
print("Training PPO...")
model.learn(total_timesteps=2000, callback=eval_callback, progress_bar=True)  # Adjust timesteps based on your requirements

print("PPO model trained!")


# Load the best model
best_model_path = './models/best_model.zip'
model = PPO.load(best_model_path)

# Testing the trained PPO model in the environment
observation, _ = env.reset()
done = False
total_reward = 0

print("testing")
while not done:
    action, _ = model.predict(observation, deterministic=True)
    #action_probs = model.policy.predict(observation, deterministic=True)
    #print(f"Action probabilities: {action_probs}")
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward 
    done = terminated or truncated

    # Render the environment (or save render data if preferred)
    env.render()

print(f"Total reward from PPO model: {total_reward}")
