import pandas as pd

# Define the simple rule-based stemmer function with the updated list of suffixes
def simple_malayalam_stemmer(word):
    suffixes = [
        'അ', 'അം', 'അട്ടെ', 'അണം', 'അണ്', 'അന്നു', 'അരുത്', 'അൻ', 'അർ', 'അൽ', 'അൾ', 
        'ആ', 'ആം', 'ആവുന്ന', 'ആൽ', 
        'ഇ', 'ഇക്ക്', 'ഇന്', 'ഇന്റെ', 'ഇയ', 'ഇല്ല', 'ഇൽ', 
        'ഈ', 'ഈയ', 
        'ഉ', 'ഉം', 'ഉത്തു', 'ഉന്നു', 'ഉ്', 
        'ഊ', 
        'എ', 'എങ്കിൽ', 'എന്ന്യേ', 
        'ഏ', 
        'ഓ', 'ഓട്', 
        'ക', 'കാരി', 'കാരൻ', 'കാർ', 'ക്ക', 'ക്കുക', 'കൾ', 
        'ങ', 'ങ്ങ', 
        'ച', 'ച്ചാൻ', 
        'ട', 'ട്ടു', 
        'ത', 'ത്ത', 'ത്തം', 'ത്തി', 'ത്വം', 
        'പ', 'പ്', 'പ്പോഴ്', 'പോൾ', 'പ്പ്', 
        'മ', 'മാർ', 'മ്', 
        'യ', 'യ്തു', 
        'വ', 'വിൻ', 'വ്', 'വൻ', 'വൾ', 
        'ശ', 'ശ്ശേരി', 
        'സ', 'സ്', 
        'ൻ'
    ]
    for suffix in suffixes:
        if word.endswith(suffix):
            print(f"Removing suffix '{suffix}' from word '{word}'")
            return word[:-len(suffix)]
    return word

def stem_text(text):
    # Tokenize the text
    words = text.split()
    # Apply the simple stemmer to each word
    stemmed_words = [simple_malayalam_stemmer(word) for word in words]
    return ' '.join(stemmed_words)

def preprocess_text(data_file_path):
    # Load the dataset
    dataset = pd.read_csv(data_file_path)

    # Inspect the dataset
    print("Initial dataset info:")
    print(dataset.info())
    print(dataset.head())

    # Apply the stemming function to the News/Comment column
    dataset['News/Comment'] = dataset['News/Comment'].apply(stem_text)

    # Inspect the cleaned dataset
    print("\nStemmed dataset info:")
    print(dataset.info())
    print(dataset.head())

    # Save the stemmed dataset
    stemmed_file_path = data_file_path.replace('.csv', '_stemmed.csv')
    dataset.to_csv(stemmed_file_path, index=False)

    print(f"Text has been stemmed and saved to {stemmed_file_path}")

# Example usage
data_file_path = '/Users/triahavijayekkumaran/Downloads/dataremnums_cleaned_customsw.csv'
preprocess_text(data_file_path)
