# graphlearning
Assignment for MSc Information Systems at NTU, for the module Graph Analysis, Learning and Applications, involving tasks such as node classification and link prediction as well as applying embedded methods like DeepWalk and Node2Vec, before evaluating model performance at the end.

The following datasets (https://snap.stanford.edu/node2vec/) were used for this project:
  •Wikipedia Graph (for Node Classification)
  •Facebook Social Network (for Link Prediction)
The WN18RR datasest, a subset of WordNet with reversed relations removed (commonly used for knowledge graph completion tasks) was 

1. Node Classification with the Wikipedia graph
   Goal: Classify nodes into their respective categories using node embeddings
  - Data preprocessing, splitting dataset into 80% training 20% testing sets
  - Trained classifier (logistic regression) using the node embeddings to classify the nodes into their respective labels.
  - Evaluated the classifier’s performance using metrics such as accuracy, precision, recall, and F1-score
  - Experimented with different Node2Vec hyperparameters (three variants: p=1, q=1; p=0.5, q=2.0; p=2.0, q=0.5) and compare their classification performance.

2. Link Prediction with the Facebook Social Network dataset
   Goal: Predict missing links between nodes in the graph.
   - Setup, data preprocessing etc. by following the experimental setting from the original Node2Vec paper (Grover & Leskovec, 2016)
     - To obtain positive examples, randomly removed 50% of the edges from the network while ensuring the graph remains connected.
     - For negative examples, randomly sampled an equal number of node pairs that are not connected by an edge.
   - Trained and evaluated the link prediction model using metrics like AUC (Area Under the ROC Curve).
     -Generated random walks for Node2Vec, trained Node2Vec embeddings via Word2Vec before generating node embeddings
     - Model evaluation done by looking at AUC score
     - Compared the results across different Node2Vec variants (three variants: p=1, q=1; p=0.5, q=2.0; p=2.0, q=0.5) for comparison
    
3. Knowledge Graph Completion with KGE, WN18RR dataset
    Goal: Predict missing triples in the knowledge graph, such as predicting the missing subject, relation, or object in triples like (subject, relation, ?), use the embeddings learned by TransE, TransR, and DistMult to predict missing links.
  - Trained TransE, TransR, and DistMult models on the WN18RR dataset.
  - Evaluated the models using the following metrics: Mean Rank, Hits@k, Mean Reciprocal Rank (MRR)
  - Compared performance of different knowledge graph embedding techniques

    
