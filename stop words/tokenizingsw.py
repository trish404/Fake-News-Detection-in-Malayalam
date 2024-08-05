import csv

# Path to the input text file
input_file_path = '/Users/triahavijayekkumaran/Desktop/studies/iit bsc/ML NLP Hackathon/stopwords.txt'  # Adjust the path and file extension as needed

# Read the content of the text file
with open(input_file_path, 'r', encoding='utf-8') as file:
    text_content = file.read()

# Split the content into individual words using whitespace as the delimiter
words = text_content.split()

# Path to save the output CSV file
output_csv_path = '/Users/triahavijayekkumaran/Desktop/studies/iit bsc/ML NLP Hackathon/malayalamstopwords.csv'

# Save the words into a CSV file
with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile)
    for word in words:
        csvwriter.writerow([word])

print(f"CSV file saved at: {output_csv_path}")
