import pandas as pd
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# Function to transliterate using indic-transliteration
def transliterate_word(word):
    return transliterate(word, sanscript.ITRANS, sanscript.MALAYALAM)

# Function to read, transliterate, and save the CSV file
def transliterate_csv(input_csv_path, output_csv_path):
    # Read the CSV file
    df = pd.read_csv(input_csv_path, header=None, names=['word'])
    
    # Apply the transliteration
    df['malayalam_word'] = df['word'].apply(transliterate_word)
    
    # Create a new dataframe with only the Malayalam words
    malayalam_df = df[['malayalam_word']]
    
    # Save the dataframe to a new CSV file
    malayalam_df.to_csv(output_csv_path, index=False, header=False, encoding='utf-8')

# File paths
input_csv_path = '/Users/triahavijayekkumaran/Desktop/studies/iit bsc/ML NLP Hackathon/malayalamstopwords.csv'  # Change this to your input file path
output_csv_path = '/Users/triahavijayekkumaran/Desktop/studies/iit bsc/ML NLP Hackathon/transmalayalamstopwords.csv'  # Change this to your output file path

# Call the function
transliterate_csv(input_csv_path, output_csv_path)

print(f"Transliterated CSV file saved at: {output_csv_path}")
