import pandas as pd

# Load the dataset
df = pd.read_csv('.\\dataset\\candlestick_data_preprocess.csv')

# Calculate the index to split (70% training, 30% testing)
split_index = int(0.7 * len(df))

# Split the data into training and testing sets
train_df = df[:split_index]
test_df = df[split_index:]

# Save the split datasets as separate CSV files (optional)
train_df.to_csv('.\\dataset\\candlestick_train.csv', index=False)
test_df.to_csv('.\\dataset\\candlestick_test.csv', index=False)

# Display the split datasets
print("Training Set:")
print(train_df.head())
print("\nTesting Set:")
print(test_df.head())
