import networkx as nx
from node2vec import Node2Vec
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
mat_data = scipy.io.loadmat('POS.mat')
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

from gensim.models import Word2Vec
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

G=nx.read_edgelist("facebook_combined.txt", create_using=nx.Graph)
# Extra quick check to see if graph loaded properly:
nx.is_directed(G)

# Generate random walks for Node2Vec
def generate_walks(graph, num_walks, walk_length):
    walks = []
    nodes = list(graph.nodes())
    for _ in range(num_walks):
        np.random.shuffle(nodes)
        for node in nodes:
            walk = node2vec_walk(graph, start_node=node, walk_length=walk_length, p=1, q=1)
            walks.append([str(n) for n in walk])
    return walks

def node2vec_walk(graph, start_node, walk_length, p=1, q=1):
    walk = [start_node]
    while len(walk) < walk_length:
        cur = walk[-1]
        neighbors = list(graph.neighbors(cur))
        if neighbors:
            if len(walk) == 1:
                walk.append(np.random.choice(neighbors))
            else:
                prev = walk[-2]
                next_node = np.random.choice(neighbors)  # Simplified probability
                walk.append(next_node)
        else:
            break
    return walk

# Generate walks
walks = generate_walks(G, num_walks=10, walk_length=5)

# Train the Node2Vec embeddings
model = Word2Vec(walks, vector_size=64, window=5, min_count=0, sg=1, workers=2, epochs=10)

# Generate node embeddings
embeddings = {str(node): model.wv[str(node)] for node in G.nodes()}

# Prepare positive and negative samples

import random

# Function to create positive and negative samples
def create_edge_samples(graph, removal_fraction=0.5, num_negative=100):
    # --- Positive Samples: Remove 50% of edges randomly and ensure the graph stays connected ---
    edges = list(graph.edges())
    num_edges_to_remove = int(len(edges) * removal_fraction)

    # Randomly remove edges
    random.shuffle(edges)
    removed_edges = edges[:num_edges_to_remove]

    # Create a copy of the graph to test connectivity
    temp_graph = graph.copy()
    temp_graph.remove_edges_from(removed_edges)

    # Check connectivity
    if not nx.is_connected(temp_graph):
        # If disconnected, return original edges (can handle differently if needed)
        positive_samples = edges  # Returning all original edges as fallback
    else:
        positive_samples = edges[num_edges_to_remove:]  # Remaining edges after removal

    # --- Negative Samples: Randomly sample node pairs that are not connected ---
    all_nodes = list(graph.nodes())
    negative_samples = []
    while len(negative_samples) < num_negative:
        u, v = np.random.choice(all_nodes, 2)
        if not graph.has_edge(u, v) and u != v:
            negative_samples.append((u, v))

    return positive_samples, negative_samples

# Generate positive and negative samples
positive_samples, negative_samples = create_edge_samples(G)


# Generate edge embeddings
def edge_embedding(u, v, embeddings, method="average"):
    if method == "average":
        return (embeddings[str(u)] + embeddings[str(v)]) / 2
    elif method == "hadamard":
        return embeddings[str(u)] * embeddings[str(v)]
    elif method == "weighted-l1":
        return np.abs(embeddings[str(u)] - embeddings[str(v)])
    elif method == "weighted-l2":
        return (embeddings[str(u)] - embeddings[str(v)]) ** 2

X = [edge_embedding(u, v, embeddings) for u, v in positive_samples + negative_samples]
y = [1] * len(positive_samples) + [0] * len(negative_samples)
## y is ground truth or not. 1 or 0!

from sklearn.linear_model import LogisticRegression
# Train classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Evaluate the model
y_pred = classifier.predict(X_test)
y_pred_proba = classifier.predict_proba(X_test)[:, 1]  # Probability scores for ROC-AUC
auc_score = roc_auc_score(y_test, y_pred_proba)

print(f"AUC-ROC Score: {auc_score:.2f}")


# Variant 1: Generate walks with p=0.5, q=2

def generate_walks_variant1(graph, num_walks, walk_length):
    walks = []
    nodes = list(graph.nodes())
    for _ in range(num_walks):
        np.random.shuffle(nodes)
        for node in nodes:
            walk = node2vec_walk_variant1(graph, start_node=node, walk_length=walk_length, p=0.5, q=2)
            walks.append([str(n) for n in walk])
    return walks

def node2vec_walk_variant1(graph, start_node, walk_length, p=0.5, q=2):
    walk = [start_node]
    while len(walk) < walk_length:
        cur = walk[-1]
        neighbors = list(graph.neighbors(cur))
        if neighbors:
            if len(walk) == 1:
                walk.append(np.random.choice(neighbors))
            else:
                prev = walk[-2]
                next_node = np.random.choice(neighbors)  # Simplified probability
                walk.append(next_node)
        else:
            break
    return walk

# Generate walks using Variant 1 parameters
walks_variant1 = generate_walks_variant1(G, num_walks=10, walk_length=5)

# Train Node2Vec embeddings with Variant 1 walks
model_variant1 = Word2Vec(walks_variant1, vector_size=64, window=5, min_count=0, sg=1, workers=2, epochs=10)
embeddings_variant1 = {str(node): model_variant1.wv[str(node)] for node in G.nodes()}

# Prepare positive and negative samples (same as previous steps)
positive_samples, negative_samples = create_edge_samples(G)

# Generate edge embeddings for Variant 1
X_variant1 = [edge_embedding(u, v, embeddings_variant1) for u, v in positive_samples + negative_samples]
y_variant1 = [1] * len(positive_samples) + [0] * len(negative_samples)

# Train a classifier for Variant 1
X_train_v1, X_test_v1, y_train_v1, y_test_v1 = train_test_split(X_variant1, y_variant1, test_size=0.3, random_state=42)
classifier_variant1 = LogisticRegression()
classifier_variant1.fit(X_train_v1, y_train_v1)

# Evaluate the model for Variant 1
y_pred_v1 = classifier_variant1.predict(X_test_v1)
y_pred_proba_v1 = classifier_variant1.predict_proba(X_test_v1)[:, 1]  # Probability scores for ROC-AUC
auc_score_variant1 = roc_auc_score(y_test_v1, y_pred_proba_v1)

print(f"AUC Score for Variant 1 (p=0.5, q=2): {auc_score_variant1}")

# Variant 2: Generate walks with p=2.0, q=0.5

def generate_walks_variant2(graph, num_walks, walk_length):
    walks = []
    nodes = list(graph.nodes())
    for _ in range(num_walks):
        np.random.shuffle(nodes)
        for node in nodes:
            walk = node2vec_walk_variant2(graph, start_node=node, walk_length=walk_length, p=2.0, q=0.5)
            walks.append([str(n) for n in walk])
    return walks

def node2vec_walk_variant2(graph, start_node, walk_length, p=2.0, q=0.5):
    walk = [start_node]
    while len(walk) < walk_length:
        cur = walk[-1]
        neighbors = list(graph.neighbors(cur))
        if neighbors:
            if len(walk) == 1:
                walk.append(np.random.choice(neighbors))
            else:
                prev = walk[-2]
                next_node = np.random.choice(neighbors)  # Simplified probability
                walk.append(next_node)
        else:
            break
    return walk

# Generate walks using Variant 2 parameters
walks_variant2 = generate_walks_variant2(G, num_walks=10, walk_length=5)

# Train Node2Vec embeddings with Variant 2 walks
model_variant2 = Word2Vec(walks_variant2, vector_size=64, window=5, min_count=0, sg=1, workers=2, epochs=10)
embeddings_variant2 = {str(node): model_variant2.wv[str(node)] for node in G.nodes()}

# Prepare positive and negative samples (same as previous steps)
positive_samples, negative_samples = create_edge_samples(G)

# Generate edge embeddings for Variant 2
X_variant2 = [edge_embedding(u, v, embeddings_variant2) for u, v in positive_samples + negative_samples]
y_variant2 = [1] * len(positive_samples) + [0] * len(negative_samples)

# Train a classifier for Variant 2
X_train_v2, X_test_v2, y_train_v2, y_test_v2 = train_test_split(X_variant2, y_variant2, test_size=0.3, random_state=42)
classifier_variant2 = LogisticRegression()
classifier_variant2.fit(X_train_v2, y_train_v2)

# Evaluate the model for Variant 2
y_pred_v2 = classifier_variant2.predict(X_test_v2)
y_pred_proba_v2 = classifier_variant2.predict_proba(X_test_v2)[:, 1]  # Probability scores for ROC-AUC
auc_score_variant2 = roc_auc_score(y_test_v2, y_pred_proba_v2)

print(f"AUC Score for Variant 2 (p=2.0, q=0.5): {auc_score_variant2}")

