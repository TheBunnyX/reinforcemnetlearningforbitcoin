#trainingPPO.py
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from CryptoTradingEnv import CryptoTradingEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

# Load and preprocess historical crypto data with added technical indicators
data = pd.read_csv('./dataset/candlestick_train.csv', parse_dates=['Date'])

# Initialize the custom trading environment with a refined reward function
env = CryptoTradingEnv(data=data)
check_env(env)

# Vectorize environment for stable-baselines compatibility
vec_env = DummyVecEnv([lambda: env])

# Set up callback for early stopping based on target reward threshold
eval_callback = EvalCallback(
    vec_env, 
    best_model_save_path="./logs/",
    log_path="./logs/",
    eval_freq=1000,
    deterministic=True,
    render=False,
    callback_on_new_best=StopTrainingOnRewardThreshold(reward_threshold=1.0, verbose=1)
)

# Initialize PPO model with tuned parameters for more accurate trading
model = PPO(
    'MlpLstmPolicy',  # Use an LSTM policy to account for temporal dependencies
    vec_env,
    verbose=1,
    learning_rate=0.0001,  # Lower learning rate for better accuracy
    n_steps=4096,          # Higher steps per rollout for more stable updates
    batch_size=128,        # Larger batch size
    gae_lambda=0.95,
    gamma=0.99,
    ent_coef=0.001         # Lower entropy to prioritize exploitation
)

# Train the model with the callback for stopping if reward threshold is reached
timesteps = 20000  # Extend training time for better convergence
model.learn(total_timesteps=timesteps, callback=eval_callback)

# Save and reload model for evaluation
model.save("optimized_ppo_crypto_trading_model")
model = PPO.load("optimized_ppo_crypto_trading_model")

# Run a single episode with render data saved to CSV
obs, _ = env.reset()
for _ in range(len(data) - 1):
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        break

# Save render output to CSV for evaluation
env.render(save_path="render_output.csv")