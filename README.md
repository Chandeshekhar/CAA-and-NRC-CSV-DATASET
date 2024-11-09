# CAA-and-NRC-CSV-DATASET
API Open source data from twitter malign information
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
# df = pd.read_csv('path_to_dataset.csv') # Replace with actual file path

# Separate true and false news
true_news = df[df['label'] == 'TRUE']
false_news = df[df['label'] == 'FAKE']

# Extract keywords using TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=50)
true_matrix = vectorizer.fit_transform(true_news['text'])
false_matrix = vectorizer.fit_transform(false_news['text'])

true_keywords = vectorizer.get_feature_names_out()
false_keywords = vectorizer.get_feature_names_out()

# Create a graph
G = nx.Graph()

# Add nodes for true and false keywords
for keyword in true_keywords:
    G.add_node(keyword, category='True News')
for keyword in false_keywords:
    G.add_node(keyword, category='False News')

# Add edges between common keywords
common_keywords = set(true_keywords).intersection(set(false_keywords))
for keyword in common_keywords:
    G.add_edge('True News', keyword)
    G.add_edge('False News', keyword)

# Draw the network
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10)
plt.title("Network Graph of True vs False News Keywords")
plt.show()
