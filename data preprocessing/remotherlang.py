import pandas as pd
import re

def preprocess_text(file_path):
    # Load the dataset
    dataset = pd.read_csv(file_path)

    # Inspect the dataset
    print("Initial dataset info:")
    print(dataset.info())
    print(dataset.head())

    # Function to clean the text by removing punctuation and special characters except for Malayalam script and numbers
    def clean_text(text):
        # Convert text to lowercase
        text = text.lower()
        # Remove punctuation and special characters except for Malayalam script and numbers
        text = re.sub(r'[^\w\s\u0D00-\u0D7F0-9]', '', text)
        return text

    # Function to check if text contains any Malayalam characters
    def contains_malayalam(text):
        return bool(re.search(r'[\u0D00-\u0D7F]', text))

    # Apply the language detection function and filter only entries with Malayalam characters
    dataset = dataset[dataset['News/Comment'].apply(contains_malayalam)]

    # Apply the cleaning function to the News/Comment column
    dataset['News/Comment'] = dataset['News/Comment'].apply(clean_text)

    # Retain only the ID, Type, and cleaned News/Comment columns
    dataset_cleaned = dataset[['ID', 'Type', 'News/Comment']]

    # Inspect the cleaned dataset
    print("\nCleaned dataset info:")
    print(dataset_cleaned.info())
    print(dataset_cleaned.head())

    # Save the cleaned dataset
    cleaned_file_path = file_path.replace('.csv', '_cleaned.csv')
    dataset_cleaned.to_csv(cleaned_file_path, index=False)

    print(f"Text has been cleaned (entries with Malayalam characters retained) and saved to {cleaned_file_path}")

# Example usage
file_path = '/Users/triahavijayekkumaran/Downloads/dataremnums.csv'
preprocess_text(file_path)
