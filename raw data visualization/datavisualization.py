import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Path to the downloaded Noto Sans Malayalam font file
font_path = '/Users/triahavijayekkumaran/Downloads/Noto_Sans_Malayalam/NotoSansMalayalam-VariableFont_wdth,wght.ttf'  # Update this path

# Mapping of categories to numerical labels
label_mapping = {
    'FALSE': 0,
    'MOSTLY FALSE': 1,
    'HALF TRUE': 2,
    'PARTLY FALSE': 3,
    'MOSTLY TRUE': 4
}

# Function to load the dataset
def load_dataset(file_path):
    return pd.read_csv(file_path)

# Function to plot the class distribution
def plot_class_distribution(data):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x='Label', order=[0, 1, 2, 3, 4], palette='viridis')
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(ticks=[0, 1, 2, 3, 4], labels=['FALSE', 'MOSTLY FALSE', 'HALF TRUE', 'PARTLY FALSE', 'MOSTLY TRUE'])
    plt.show()

# Function to generate and plot a word cloud
def plot_word_cloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white', font_path=font_path).generate(text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Main function to visualize data
def visualize_data(file_path):
    # Load the dataset
    dataset = load_dataset(file_path)
    
    # Check for missing values
    missing_values = dataset.isnull().sum()
    print("Missing Values:\n", missing_values)
    
    # Map the text labels to numerical labels
    dataset['Label'] = dataset['Type'].map(label_mapping)
    
    # Analyze the distribution of different classes
    class_distribution = dataset['Label'].value_counts()
    print("Class Distribution:\n", class_distribution)
    
    # Plotting the class distribution
    plot_class_distribution(dataset)
    
    # Generating and plotting word clouds for each class
    for class_type in dataset['Label'].unique():
        class_text = ' '.join(dataset[dataset['Label'] == class_type]['News/Comment'].fillna(''))
        plot_word_cloud(class_text, f'Word Cloud for {list(label_mapping.keys())[list(label_mapping.values()).index(class_type)]}')

# Example usage
file_path = '/Users/triahavijayekkumaran/Downloads/labelled_train_set.csv'
visualize_data(file_path)
