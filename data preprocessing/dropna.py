import pandas as pd

# Load the dataset
file_path = '/Users/triahavijayekkumaran/Downloads/labelled_train_set.csv'
dataset = pd.read_csv(file_path)

# Inspect the dataset for missing values
print("Initial dataset info:")
print(dataset.info())
print("\nRows with missing values before dropping:")
print(dataset.isnull().sum())

# Drop rows with any missing values
dataset_cleaned = dataset.dropna(axis=0, how='any')

# Inspect the cleaned dataset
print("\nCleaned dataset info:")
print(dataset_cleaned.info())
print("\nRows with missing values after dropping:")
print(dataset_cleaned.isnull().sum())

# Save the cleaned dataset
cleaned_file_path = '/Users/triahavijayekkumaran/Downloads/droppedna.csv'
dataset_cleaned.to_csv(cleaned_file_path, index=False)

print(f"Rows with missing values have been dropped and saved to {cleaned_file_path}")
