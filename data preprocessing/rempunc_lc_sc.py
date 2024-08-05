import pandas as pd
import re

# Load the dataset
file_path = '/Users/triahavijayekkumaran/Downloads/droppedna.csv'
dataset = pd.read_csv(file_path)

# Inspect the dataset
print("Initial dataset info:")
print(dataset.info())
print(dataset.head())

# Function to clean the text
def clean_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation and special characters except for Malayalam script
    text = re.sub(r'[^\w\s\u0D00-\u0D7F]', '', text)
    return text

# Apply the cleaning function to the News/Comment column
dataset['News/Comment'] = dataset['News/Comment'].apply(clean_text)

# Retain only the ID, Type, and cleaned News/Comment columns
dataset_cleaned = dataset[['ID', 'Type', 'News/Comment']]

# Inspect the cleaned dataset
print("\nCleaned dataset info:")
print(dataset_cleaned.info())
print(dataset_cleaned.head())

# Save the cleaned dataset
cleaned_file_path = '/Users/triahavijayekkumaran/Downloads/datacleaned2.csv'
dataset_cleaned.to_csv(cleaned_file_path, index=False)

print(f"Text has been cleaned and saved to {cleaned_file_path}")
