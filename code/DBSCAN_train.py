# DBSCAN
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot
#from data_preparation import train_data_loader, data_pre_processing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, jaccard_score
from dbscan import DBSCAN
import networkx as nx
import operator
import sys
sys.path.insert(0,'..')
from utils.utils import train_data_loader, data_pre_processing, create_train_test, feature_selection_univ

## Load data

data = train_data_loader()

## Data Preprocessing

df = data_pre_processing(data)

# Keep only numeric features, divide features form target variable
Xtrain, ytrain = create_train_test(df)

# Drop features as per univariate selection
Xtrain = feature_selection_univ(Xtrain)

# Apply standard scaler to the data
X = StandardScaler().fit_transform(Xtrain)

eps = [0.2, 0.3, 0.5, 0.7]
min_samples = [10000, 5000, 1000, 500, 200, 100]

dbscan_performance = dict()
for e in eps:
    
    for n in min_samples:

       ## Fit the DBSCAN
       labels, core_samples_mask = DBSCAN(X,eps=e, min_samples=n)

       df_labels = pd.DataFrame(labels)

       df_core_samples_mask = pd.DataFrame(core_samples_mask)

       df_labels.to_csv(f'output/dbscan_out/dbscan_labels_{e}_{n}.csv')
       df_core_samples_mask.to_csv(f'output/dbscan_out/core_samples_mask_{e}_{n}.csv')

       df_labels = np.ravel(df_labels) 
       print(type(df_labels))
       print(type(ytrain))
       # Performance metrics

       ## Silouette average
       dkey = f'{e}_{n}_silhouette_avg'
       silhouette_avg = silhouette_score(X, df_labels)  
       dbscan_performance[dkey] = silhouette_avg

       ## Jaccard similarity
       dkey = f'{e}_{n}_jaccard_similarity'
       jaccard_similarity = jaccard_score(ytrain, df_labels, average='mirco')
       dbscan_performance[dkey] = jaccard_similarity

       dkey = f'{e}_{n}_normalized_mutual_info_score'
       nmi = normalized_mutual_info_score(ytrain, np.ravel(df_labels))
       dbscan_performance[dkey] = nmi

dbscan_performance.to_csv('output/dbscan_out/dbscan_performance.csv')

# Calculate the Silhouette Score
#silhouette_avg = silhouette_score(X, labels)

#print("Silhouette Score:", silhouette_avg)
# true_labels: true class labels for your data points
# predicted_labels: predicted cluster assignments

#jaccard_similarity = jaccard_score(true_labels, predicted_labels)
#print("Jaccard Similarity Coefficient:", jaccard_similarity)
# true_labels: true class labels for your data points
# predicted_labels: predicted cluster assignments

#nmi = normalized_mutual_info_score(true_labels, predicted_labels)

#print("Normalized Mutual Information:", nmi)