import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import operator


#eth_transaction_2023 = pd.read_csv("https://drive.google.com/uc?export=download&id=1HpiWFIMescWkVHu90_9HJVEgaVXo0VVA")

eth_transaction_2023 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/bq-results-20230514-111950-1684063331543.csv")
#eth_transaction_2023 = pd.read_csv("C:/Users/erika/PycharmProjects/MScProject/ETH/bq-results-20230410-164146-1681144959832.csv")


# hitsotical
#eth_transaction_1 = pd.read_csv("C:/Users/erika/PycharmProjects/MScProject/ETH/bq-results-2015-08-20_2015-08-09.csv")
#eth_transaction_2 = pd.read_csv("C:/Users/erika/PycharmProjects/MScProject/ETH/bq-results-2020-12-31__20-08-2020.csv", low_memory=False)
#eth_transaction_3 = pd.read_csv("C:/Users/erika/PycharmProjects/MScProject/ETH/bq-results-20230514-2016-01-01_2016-06-30.csv", low_memory=False)
#eth_transaction_4 = pd.read_csv("C:/Users/erika/PycharmProjects/MScProject/ETH/bq-results-20230514-2016-06-30_2016-07.csv", low_memory=False)
#eth_transaction_5 = pd.read_csv("C:/Users/erika/PycharmProjects/MScProject/ETH/bq-results-31-07-2016.csv", low_memory=False)
#eth_transaction_6 = pd.read_csv("C:/Users/erika/PycharmProjects/MScProject/ETH/bq-results-2016-09-30.csv", low_memory=False)
#eth_transaction_7 = pd.read_csv("C:/Users/erika/PycharmProjects/MScProject/ETH/bq-results-2016-12-30.csv", low_memory=False)
#eth_transaction_8 = pd.read_csv("C:/Users/erika/PycharmProjects/MScProject/ETH/bq-results-2017-01-31.csv", low_memory=False)
#eth_transaction_9 = pd.read_csv("C:/Users/erika/PycharmProjects/MScProject/ETH/bq-results-2017-04-01.csv", low_memory=False)
#eth_transaction_10 = pd.read_csv("C:/Users/erika/PycharmProjects/MScProject/ETH/bq-results-2017-05-25.csv", low_memory=False)
#eth_transaction_11 = pd.read_csv("C:/Users/erika/PycharmProjects/MScProject/ETH/bq-results-20161001-20161031.csv", low_memory=False)
#eth_transaction_12 = pd.read_csv("C:/Users/erika/PycharmProjects/MScProject/ETH/bq-results-20160801-20160831.csv", low_memory=False)
#eth_transaction_13 = pd.read_csv("C:/Users/erika/PycharmProjects/MScProject/ETH/bq-results-20161101-20161130.csv", low_memory=False)
#eth_transaction_14 = pd.read_csv("C:/Users/erika/PycharmProjects/MScProject/ETH/bq-results-20161201-20161231.csv", low_memory=False)

#historical = pd.concat([eth_transaction_1,eth_transaction_5,eth_transaction_6,                        eth_transaction_7,eth_transaction_8,eth_transaction_9,                        eth_transaction_10,eth_transaction_11,eth_transaction_12,                        eth_transaction_13])


historical = eth_transaction_2023

print(historical['block_timestamp'].head())
print(historical['block_timestamp'].tail())

# Remove duplicated rows
historical_nodup = historical.drop_duplicates()
print(len(historical_nodup))
print(len(historical_nodup))
print(historical_nodup.head())


print('adj matrix')
# cross tab doesn't work with 400k info
#adj_matrix = pd.crosstab(historical_nodup.from_address,historical_nodup.to_address)
#print(adj_matrix.head(15))
#adj_matrix.to_csv('adj_matrix_ETH10.csv')

# try networkX
df_networkx = historical_nodup[['from_address','to_address']]
print('df length', len(df_networkx))

df_networkx_nodup = df_networkx.drop_duplicates()
print('df address only length',len(df_networkx_nodup))

transaction_address = list(zip(df_networkx.from_address,df_networkx.to_address))
print(len(transaction_address))

# DIRECTED GRAPH / MULTI-DIRECTED GRAPH

#G = nx.DiGraph()
G = nx.MultiDiGraph()
G.add_edges_from(transaction_address)
print(G)

#adj_matrix = nx.to_pandas_adjacency(G)

#print(adj_matrix.head(10))

# degree centrality
degree_centrality=nx.degree_centrality(G)
#between_centrality = nx.betweenness_centrality(G)

# Betweness centrality

# sort the degree centrality
degree_centrality_sort = dict(sorted(degree_centrality.items(), key=operator.itemgetter(1), reverse=True))
items = degree_centrality_sort.items()

degree_centrality_from = pd.DataFrame({'from_address': [i[0] for i in items], 'degree_centrality': [i[1] for i in items]})
degree_centrality_to = pd.DataFrame({'to_address': [i[0] for i in items], 'degree_centrality': [i[1] for i in items]})

in_degree_to = pd.DataFrame(G.in_degree, columns=['to_address','in_degree'])
out_degree_to = pd.DataFrame(G.out_degree,columns=['to_address','out_degree'])

in_degree_from = pd.DataFrame(G.in_degree, columns=['from_address','in_degree'])
out_degree_from = pd.DataFrame(G.out_degree,columns=['from_address','out_degree'])



# Merge to the transaction data

df = historical.merge(degree_centrality_from, how='left', on='from_address')
df = df.merge(degree_centrality_to, how='left', on='to_address')

df = df.merge(in_degree_to, how='left', on='to_address')
df = df.merge(out_degree_to, how='left', on='to_address')

df = df.merge(in_degree_from, how='left', on='from_address')
df = df.merge(out_degree_from, how='left', on='from_address')

print(df.columns)
# Betwness centrality
#between_centrality_sort = dict(sorted(between_centrality.items(), key=operator.itemgetter(1), reverse=True))


x = 0