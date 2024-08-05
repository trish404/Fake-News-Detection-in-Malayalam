import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Path to the downloaded Noto Sans Malayalam font file
font_path = '/Users/triahavijayekkumaran/Downloads/Noto_Sans_Malayalam/NotoSansMalayalam-VariableFont_wdth,wght.ttf'

# Setting the font properties for matplotlib
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Noto Sans Malayalam']

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
    sns.countplot(data=data, x='Label', palette='viridis')
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(ticks=[0, 1, 2, 3, 4], labels=['FALSE', 'MOSTLY FALSE', 'HALF TRUE', 'PARTLY FALSE', 'MOSTLY TRUE'])
    plt.show()

# Function to generate and plot a word cloud
def plot_word_cloud(text, title, font_path):
    wordcloud = WordCloud(width=800, height=400, background_color='white', font_path=font_path).generate(text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Function to plot TF-IDF top terms
def plot_tfidf_top_terms(corpus, labels, class_label, n=20):
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(corpus)
    feature_names = tfidf.get_feature_names_out()
    
    class_docs = [corpus[i] for i in range(len(labels)) if labels[i] == class_label]
    class_tfidf = tfidf.transform(class_docs).mean(axis=0).A1
    top_terms = sorted(zip(feature_names, class_tfidf), key=lambda x: x[1], reverse=True)[:n]
    
    terms, scores = zip(*top_terms)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=scores, y=terms)
    plt.title(f'Top {n} TF-IDF Terms for Class {class_label}')
    plt.xlabel('TF-IDF Score')
    plt.ylabel('Term')
    plt.show()

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# Function to plot ROC curve
def plot_roc_curve(y_true, y_score, n_classes):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

# Function to plot Precision-Recall curve
def plot_precision_recall_curve(y_true, y_score, n_classes):
    precision = dict()
    recall = dict()
    
    plt.figure()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_score[:, i])
        plt.plot(recall[i], precision[i], lw=2, label=f'Class {i}')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="best")
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
        plot_word_cloud(class_text, f'Word Cloud for {list(label_mapping.keys())[list(label_mapping.values()).index(class_type)]}', font_path)

    # Plotting TF-IDF top terms for each class
    corpus = dataset['News/Comment'].fillna('').tolist()
    labels = dataset['Label'].tolist()
    for class_type in dataset['Label'].unique():
        plot_tfidf_top_terms(corpus, labels, class_type)
        
    # Assuming y_true and y_pred are available from your classification model
    # Plotting the confusion matrix
    y_true = [label_mapping[label] for label in dataset['Type']]
    y_pred = y_true  # Placeholder, replace with your model predictions
    plot_confusion_matrix(y_true, y_pred, list(label_mapping.keys()))

    # Assuming y_score is available from your classification model
    # Plotting ROC and Precision-Recall curves
    n_classes = len(label_mapping)
    y_true_binarized = label_binarize(y_true, classes=range(n_classes))
    y_score = y_true_binarized  # Placeholder, replace with your model scores
    
    plot_roc_curve(y_true_binarized, y_score, n_classes)
    plot_precision_recall_curve(y_true_binarized, y_score, n_classes)

# Example usage
file_path = '/Users/triahavijayekkumaran/Downloads/labelled_train_set.csv'
visualize_data(file_path)
