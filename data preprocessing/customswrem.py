import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Ensure NLTK stopwords are downloaded
nltk.download('stopwords')

def load_stopwords(file_path):
    # Load the stopwords from a CSV file without column names
    stopwords_df = pd.read_csv(file_path, header=None)
    # Use the first column for stopwords
    custom_stopwords = set(stopwords_df.iloc[:, 0].tolist())
    return custom_stopwords

def preprocess_text(data_file_path, stopwords_file_path):
    # Load the dataset
    dataset = pd.read_csv(data_file_path)

    # Load custom Malayalam stopwords
    malayalam_stopwords = load_stopwords(stopwords_file_path)

    # Combine default English stopwords with Malayalam stopwords
    stop_words = set(stopwords.words('english')).union(malayalam_stopwords)

    # Inspect the dataset
    print("Initial dataset info:")
    print(dataset.info())
    print(dataset.head())

    # Function to clean the text
    def clean_text(text):
        # Tokenize the text
        words = text.split()
        # Remove stopwords
        words = [word for word in words if word not in stop_words]
        return ' '.join(words)

    # Apply the cleaning function to the News/Comment column
    dataset['News/Comment'] = dataset['News/Comment'].apply(clean_text)

    # Retain only the ID, Type, and cleaned News/Comment columns
    dataset_cleaned = dataset[['ID', 'Type', 'News/Comment']]

    # Inspect the cleaned dataset
    print("\nCleaned dataset info:")
    print(dataset_cleaned.info())
    print(dataset_cleaned.head())

    # Save the cleaned dataset
    cleaned_file_path = data_file_path.replace('.csv', '_customsw.csv')
    dataset_cleaned.to_csv(cleaned_file_path, index=False)

    print(f"Text has been cleaned (custom stopwords removed) and saved to {cleaned_file_path}")

# Example usage
data_file_path = '/Users/triahavijayekkumaran/Downloads/dataremnums_cleaned.csv'
stopwords_file_path = '/Users/triahavijayekkumaran/Desktop/studies/iit bsc/ML NLP Hackathon/stopwords/transmalayalamstopwords.csv'
preprocess_text(data_file_path, stopwords_file_path)
