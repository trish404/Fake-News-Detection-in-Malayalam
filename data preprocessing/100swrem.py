import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Ensure NLTK stopwords are downloaded
nltk.download('stopwords')

# List of 100 common Malayalam stopwords
malayalam_stopwords = [
    'ആ', 'ആക', 'ആണ്', 'ആണു', 'അവൻ', 'അവളുടെ', 'അവൻറെ', 'അവരെ', 'അവർ', 'ഇ', 'ഇത്', 'ഇനി', 'ഇവിടെ', 'ഇവരുടെ', 
    'ഇപ്പോൾ', 'എങ്കിൽ', 'എങ്കിലും', 'എന്ത്', 'എവിടെ', 'എപ്പോൾ', 'ഒന്നും', 'കൂടെ', 'കൂടാതെ', 'കൂടിയ', 'കൂടെ', 'ഞാൻ', 
    'ഞങ്ങൾ', 'ഞങ്ങളുടെ', 'അല്ല', 'അല്ലെങ്കിൽ', 'അവിടെ', 'അതിനാൽ', 'അതുവരെ', 'അവള്', 'അതുപോലെ', 'അതുകൊണ്ട്', 'എന്ന', 
    'എന്നാല്', 'എന്തിനാൽ', 'അത്', 'അവിടെ', 'ഇത്', 'ഇങ്ങനെയാണ്', 'ഇങ്ങനെ', 'അവന്', 'ഇവ', 'ഇതിൽ', 'അതിൽ', 'ഇവിടം', 
    'ഇവിടേക്ക്', 'അവരുടെ', 'ഇന്നും', 'അങ്ങനെ', 'ഇതിനു', 'അതിനു', 'അവന്', 'എന്ന', 'അല്ലെങ്കിൽ', 'പിന്നെ', 'എന്താ', 
    'മറ്റൊരു', 'പക്ഷെ', 'ഇല്ല', 'ഇപ്പോൾ', 'അപ്പോള്', 'അതെ', 'അല്ല', 'ഇപ്പോള്', 'ഇല്ല', 'ഇത്', 'എന്താണ്', 'ഇല്ല', 'നിങ്ങള്', 
    'നിന്റെ', 'അതുപോലെ', 'നല്ല', 'എല്ലാ', 'ഇല്ല', 'എന്താണെന്ന്', 'അവളുടെ', 'മറ്റെന്താണ്', 'കൂടാതെ', 'വരെ', 'പിന്നെ', 'എവിടെ', 
    'പോയി', 'അവരുടെ', 'എന്തെങ്കിലും', 'അവന്റെ', 'അതിന്റെ', 'അല്ല', 'അതിനെ', 'ഇല്ല'
]

# Combine default English stopwords with Malayalam stopwords
stop_words = set(stopwords.words('english')).union(set(malayalam_stopwords))

def preprocess_text(file_path):
    # Load the dataset
    dataset = pd.read_csv(file_path)

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
    cleaned_file_path = file_path.replace('.csv', '_swrem.csv')
    dataset_cleaned.to_csv(cleaned_file_path, index=False)

    print(f"Text has been cleaned (stopwords removed) and saved to {cleaned_file_path}")

# Example usage
file_path = '/Users/triahavijayekkumaran/Downloads/datacleaned2_cleaned.csv'
preprocess_text(file_path)
