import networkx as nx
from node2vec import Node2Vec
import matplotlib.pyplot as plt
import numpy as np

import scipy.io
mat_data = scipy.io.loadmat('POS.mat')

# Print the keys of the loaded data
print(mat.keys())


# Access a specific variable (replace 'variable_name' with the actual name)
variable = mat_data['network']
print(variable)
variable = mat_data['group']
print(variable)

edge_list = mat_data['network']  # Adjust this based on the structure of your .mat file
# Create the graph
G = nx.Graph(edge_list)
# Extra quick check to see if graph loaded properly
is_directed = nx.is_directed(G)
print(f"Is the graph directed? {is_directed}")

print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
plt.figure(figsize=(10,7))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True)
plt.show()
#print(f"Number of nodes: {len(G.nodes())}")
#print(f"Number of edges: {len(G.edges())}")
average_clustering = nx.average_clustering(G)
print(f"Average Clustering Coefficient: {average_clustering}")

# Set up Node2Vec with parameters p, q, walk length, and number of walks
node2vec = Node2Vec(G, dimensions=64, walk_length=80, num_walks=10, p=1, q=1, workers=1)
# Fit the model to generate embeddings
model = node2vec.fit(window=10, min_count=1, batch_words=4)
# Access node embeddings for all nodes
node_embeddings = model.wv  # or model.wv[model.wv.index_to_key] for vectors

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Assuming node embeddings are strings and you're using node indices as strings
X = np.array([node_embeddings[str(node)] for node in G.nodes()])
group = mat_data['group']  # Ensure this is correctly loaded
# Extract labels from 'group' variable (ensure correct indexing)
y = []
for node in G.nodes():
    label = group[node, 1]  # Adjust this depending on the structure of 'group'
    y.append(label)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Train a classifier (e.g., Logistic Regression)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
# Make predictions
y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
cm = confusion_matrix(y_test, y_pred)
print(cm)


#experiment with new variant
node2vec = Node2Vec(G, dimensions=64, walk_length=80, num_walks=10, p=0.5, q=2.0, workers=1)
model_p05_q20 = node2vec.fit(window=10, min_count=1, batch_words=4)


y = []
for node in G.nodes():
    label = group[node, 1]  # Adjust this depending on the structure of 'group'
    y.append(label)
# Convert the list of labels to a numpy array
y = np.array(y)
# Access node embeddings for all nodes
node_embeddings = model_p05_q20.wv
# Prepare feature matrix for classifier
X = np.array([node_embeddings[str(node)] for node in G.nodes()])  # Node embeddings
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Train classifier (e.g., Logistic Regression)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Evaluate the classifier
y_pred = clf.predict(X_test)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print results
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

#next variant
node2vec = Node2Vec(G, dimensions=64, walk_length=80, num_walks=10, p=2.0, q=0.5, workers=1)
model_p20_q05 = node2vec.fit(window=10, min_count=1, batch_words=4)


y = []
for node in G.nodes():
    label = group[node, 1]  # Adjust this depending on the structure of 'group'
    y.append(label)
# Convert the list of labels to a numpy array
y = np.array(y)
# Access node embeddings for all nodes
node_embeddings = model_p20_q05.wv
# Prepare feature matrix for classifier
X = np.array([node_embeddings[str(node)] for node in G.nodes()])  # Node embeddings
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Train classifier (e.g., Logistic Regression)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Evaluate the classifier
y_pred = clf.predict(X_test)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print results
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")