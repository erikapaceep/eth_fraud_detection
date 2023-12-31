import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot
from utils import create_date, drop_smart_contract, clean_up_row, drop_missing_data

import networkx as nx
import operator

# July 2017 data

def train_data_loader():
    
    eth_20170701 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_trans_20170701.csv")


    data = eth_20170701
    return data

def test_data_loader():
    
    eth_06092017 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_06092017.csv")


    data = eth_06092017
    
    return data

def data_pre_processing(data):
    
    # Remove invalid transaction
    data = clean_up_row(data)

    # Remove smart contracts
    data_no_smart = drop_smart_contract(data)

    # Drop when from and to address is missing & 'block_timestamp' is missing
    data_no_miss = drop_missing_data(data_no_smart)

    # Remove duplicated rows
    data_no_smart_nodup_nomiss = data_no_miss.drop_duplicates()

    # Create dates 
    data = create_date(data_no_smart_nodup_nomiss)

    # transform attributes form 'object' dtype into numeric
    data['gas'] = data['gas'].astype('float')
    data['value'] = data['value'].astype('float')


    ## Feature engineering and Feature transformation

    #1. Compute the gas fees per unit: 'gas_price_unit'
    data['gas_price_unit'] = data['gas_price']/data['gas']

    #2. Compute the value of the transaction divided by the gas: 'value'
    data['value_div_gas'] = data['value']/data['gas']

    #3. Number of transactions per address from and to

    # count number of transaction for each 'address from' in July 2017
    data['from_address_count'] = data.groupby('from_address')['from_address'].transform('count')
    # count number of transaction for each 'address to' in July 2017
    data['to_address_count'] = data.groupby('to_address')['to_address'].transform('count')

    #4. Number of transactions per block
    data['block_count'] = data.groupby('block_number')['block_number'].transform('count')

    #5. Degree centrality
    # Create the graph using network x
    df_networkx = data[['from_address','to_address']]

    df_networkx_nodup = df_networkx.drop_duplicates()

    transaction_address = list(zip(df_networkx.from_address,df_networkx.to_address))

    # DIRECTED GRAPH / MULTI-DIRECTED GRAPH

    G = nx.MultiDiGraph()
    G.add_edges_from(transaction_address)

    degree_centrality=nx.degree_centrality(G)

    # sort the degree centrality
    degree_centrality_sort = dict(sorted(degree_centrality.items(), key=operator.itemgetter(1), reverse=True))
    items = degree_centrality_sort.items()

    degree_centrality_from = pd.DataFrame({'from_address': [i[0] for i in items], 'degree_centrality_from': [i[1] for i in items]})
    degree_centrality_to = pd.DataFrame({'to_address': [i[0] for i in items], 'degree_centrality_to': [i[1] for i in items]})

    in_degree_to = pd.DataFrame(G.in_degree, columns=['to_address','in_degree_adr_to'])
    out_degree_to = pd.DataFrame(G.out_degree,columns=['to_address','out_degree_adr_to'])

    in_degree_from = pd.DataFrame(G.in_degree, columns=['from_address','in_degree_adr_from'])
    out_degree_from = pd.DataFrame(G.out_degree,columns=['from_address','out_degree_adr_from'])

    # Merge the degree centrality to the transaction data

    df = data.merge(degree_centrality_from, how='left', on='from_address')
    df = df.merge(degree_centrality_to, how='left', on='to_address')

    df = df.merge(in_degree_to, how='left', on='to_address')
    df = df.merge(out_degree_to, how='left', on='to_address')

    df = df.merge(in_degree_from, how='left', on='from_address')
    df = df.merge(out_degree_from, how='left', on='from_address')

    ## Add labels : malicious addresses

    darklist = pd.read_csv('../darklist.csv',index_col=False, usecols=["address", "comment", "date;"])

    # Merge malicoous addresses on the 'from_address'
    df = df.merge(darklist, how='left', left_on='from_address', right_on='address')
    df.loc[df['address'].notna(), 'mal_trans_from'] = 1
    df.drop(['address',],axis=1,inplace=True)
    df.rename(columns = {'comment':'comment_from_address_darklst'}, inplace = True)

    # Merge malicoous addresses on the 'to_address'
    df = df.merge(darklist, how='left', left_on='to_address', right_on='address')
    df.loc[df['address'].notna(), 'mal_trans_to'] = 1
    df.drop(['address',],axis=1,inplace=True)
    df.rename(columns = {'comment':'comment_to_address_darklst'}, inplace = True)

    df.rename(columns = {'date;_x':'date_mal_trans_from','date;_y':'date_mal_trans_to'}, inplace = True)

    # Check how many times there is an overlap
    df['transaction_flag'] = 0
    df.loc[df['mal_trans_from'].notna(),'transaction_flag']=1
    df.loc[df['mal_trans_to'].notna(),'transaction_flag']=1

    df['transaction_flag1'] = 0
    df['transaction_flag2'] = 0

    df.loc[df['mal_trans_from'].notna(),'transaction_flag1']=1
    df.loc[df['mal_trans_to'].notna(),'transaction_flag2']=1

    df['attack_descr'] = df['comment_to_address_darklst'].where(pd.notnull, df['comment_from_address_darklst'])

    df['attack_date'] = df['date_mal_trans_to'].where(pd.notnull, df['date_mal_trans_from'])

    return df