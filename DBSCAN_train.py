# DBSCAN

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot
from utils import create_date, drop_smart_contract, clean_up_row, drop_missing_data
from data_preparation import train_data_loader, data_pre_processing
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

import networkx as nx
import operator

## Load data

data = train_data_loader()

## Data Preprocessing

df = data_pre_processing(data)

# Drop target features (comments, flags)
train_data = df.drop(['comment_from_address_darklst','date_mal_trans_from', 'mal_trans_from', 'comment_to_address_darklst',
                    'date_mal_trans_to', 'mal_trans_to','transaction_flag1', 'transaction_flag2','attack_descr', 'attack_date'],axis=1)


# Keep only numeric features
train_data = train_data[['nonce', 'transaction_index',
       'value', 'gas', 'gas_price', 'receipt_cumulative_gas_used',
       'receipt_gas_used', 'block_number','receipt_effective_gas_price', 
       'dates', 'gas_price_unit','value_div_gas', 'from_address_count', 'to_address_count',
       'block_count', 'degree_centrality_from', 'degree_centrality_to',
       'in_degree_adr_to', 'out_degree_adr_to', 'in_degree_adr_from','out_degree_adr_from','transaction_flag']]

Xtrain = train_data.drop(['transaction_flag'],axis=1)
Xtrain.set_index('dates', inplace=True)

## Drop feature from univariate selection
Xtrain.drop(['block_number','gas_price','receipt_effective_gas_price','in_degree_adr_to','degree_centrality_to', 'out_degree_adr_to','degree_centrality_from','out_degree_adr_from'],axis=1,inplace=True)

ytrain = train_data[['transaction_flag','dates']]
ytrain.set_index('dates', inplace=True)

## Fit the DBSCAN

clustering = DBSCAN().fit(Xtrain)

labels = pd.Series(clustering.labels_)
labels.to_csv('DBSCAN_labels.csv')