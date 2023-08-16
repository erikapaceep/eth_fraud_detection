import pandas as pd
import numpy as np
import networkx as nx
import operator
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot

import os

def create_date(data_nodup):

    data_nodup['block_timestamp_str'] = data_nodup['block_timestamp'].astype(str)
    data_nodup['date'] = data_nodup['block_timestamp_str'].str.slice(start=0,stop=10)

    data_nodup['dates'] = pd.to_datetime(data_nodup['date'], format='%Y-%m-%d')

    data_nodup['year'] = data_nodup['block_timestamp_str'].str.slice(start=0,stop=4)


    data_nodup['month'] = data_nodup['block_timestamp_str'].str.slice(start=5,stop=7)
    data_nodup['month'] = pd.to_numeric(data_nodup['month'])

    data_nodup['day'] = data_nodup['block_timestamp_str'].str.slice(start=8,stop=10)
    data_nodup['day'] = pd.to_numeric(data_nodup['day'])

    #data_nodup.sort_values(by=['dates'],ascending=False, inplace=True)
    
    # sort by timestamp as there are mutliple transactions
    data_nodup.sort_values(by=['block_timestamp'],ascending=True, inplace=True)
    return data_nodup


def drop_smart_contract(data):

    "receipt_contract_address correspond to the The contract address created, if the transaction was a contract creation, otherwise null"
     
    data_no_smart = data[data['receipt_contract_address'].isnull()]

    return data_no_smart


def clean_up_row(data):
    " the has is supposed to start with 0x -> if that's not the case, drop it"
    
    data['check_hash'] = data['hash'].astype(str)
    data['check_hash'] = data['check_hash'].str[:2]
    data_drop = data.loc[data['check_hash'] == '0x']
    data_drop.drop(columns=['check_hash'], inplace=True)

    return data_drop

def drop_missing_data(data):

    data_no_missing_from = data.loc[data['from_address'].notna()]
    data_no_missing_to = data_no_missing_from.loc[data_no_missing_from['to_address'].notna()]
    data_no_missing_date = data_no_missing_to.loc[data_no_missing_to['block_timestamp'].notna()]

    # drop missing columns: 
    data_no_missing_date.drop(['receipt_status', 'max_fee_per_gas', 'max_priority_fee_per_gas', 'transaction_type','receipt_contract_address'],axis=1,inplace=True)     

    return data_no_missing_date



def train_data_loader():
       
    eth_20170701 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_trans_20170701.csv")
    eth_20170702 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_trans_20170702.csv")
    eth_20170703 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_trans_20170703.csv")
    eth_20170704 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_trans_20170704.csv")
    eth_20170705 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_trans_20170705.csv")
    eth_20170706 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_trans_20170706.csv")
    eth_20170707 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_trans_20170707.csv")
    eth_20170708 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_trans_20170708.csv")
    eth_20170709 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_trans_20170709.csv")
    eth_20170710 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_trans_20170710.csv")
    eth_20170711 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_trans_20170711.csv")
    eth_20170712 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_trans_20170712.csv")
    eth_20170713 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_trans_20170713.csv")
    eth_20170714 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_trans_20170714.csv")
    eth_20170715 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_trans_20170715.csv")
    eth_20170716 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_trans_20170716.csv")
    eth_20170717 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_trans_20170717.csv")
    eth_20170718 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_trans_20170718.csv")
    eth_20170719 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_trans_20170719.csv")
    eth_20170720 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_trans_20170720.csv")
    eth_20170721 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_trans_20170721.csv")
    eth_20170722 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_trans_20170722.csv")
    eth_20170723 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_trans_20170723.csv")
    eth_20170724 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_trans_20170724.csv")
    eth_20170725 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_trans_20170725.csv")
    eth_20170726 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_trans_20170726.csv")
    eth_20170727 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_trans_20170727.csv")
    eth_20170728 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_trans_20170728.csv")
    eth_20170729 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_trans_20170729.csv")
    eth_20170730 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_trans_20170730.csv")
    eth_20170731 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_trans_20170731.csv")

    data = pd.concat([eth_20170701,eth_20170702,eth_20170703,
                            eth_20170704,eth_20170705,eth_20170706,
                            eth_20170707,eth_20170708, eth_20170709, 
                            eth_20170710,eth_20170711,eth_20170712, eth_20170713,
                            eth_20170714,eth_20170715,eth_20170716, eth_20170717,
                            eth_20170718,eth_20170719,eth_20170720, eth_20170721,
                            eth_20170722,eth_20170723,
                            eth_20170724, 
                            eth_20170725,
                            eth_20170726,eth_20170727,eth_20170728, 
                            eth_20170729,
                            eth_20170730, eth_20170731])
    return data

def test_data_loader():
    
    eth_06092017 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_06092017.csv")
    eth_07092017 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_07092017.csv")
    eth_08092017 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_08092017.csv")
    eth_09092017 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_09092017.csv")
    eth_10092017 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_10092017.csv")
    eth_11092017 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_11092017.csv")
    eth_12092017 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_12092017.csv")
    eth_13092017 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_13092017.csv")
    eth_14092017 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_14092017.csv")
    eth_15092017 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_15092017.csv")

    data = pd.concat([eth_06092017,eth_07092017, eth_08092017,
                  eth_09092017,eth_10092017,eth_11092017,
                  eth_12092017,eth_13092017,eth_14092017,
                  eth_15092017])
    
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

def create_X_y(data):
    
    # drop traget feature variables (except the anomalous flag)
    data_X_y = data.drop(['comment_from_address_darklst','date_mal_trans_from', 'mal_trans_from', 'comment_to_address_darklst',
                    'date_mal_trans_to', 'mal_trans_to','transaction_flag1', 'transaction_flag2','attack_descr', 'attack_date'],axis=1)


    # Keep only numeric features
    data_X_y_numeric = data_X_y[['nonce', 'transaction_index',
        'value', 'gas', 'gas_price', 'receipt_cumulative_gas_used',
        'receipt_gas_used', 'block_number','receipt_effective_gas_price', 
        'block_timestamp_str', 'gas_price_unit','value_div_gas', 'from_address_count', 'to_address_count',
        'block_count', 'degree_centrality_from', 'degree_centrality_to',
        'in_degree_adr_to', 'out_degree_adr_to', 'in_degree_adr_from','out_degree_adr_from','transaction_flag']]

    #separate target (y) from features (X)
    X_data = data_X_y_numeric.drop(['transaction_flag'],axis=1)
    X_data = data_X_y_numeric.drop(['transaction_flag'],axis=1)
    X_data.set_index('block_timestamp_str', inplace=True)


    y_data = data_X_y_numeric[['transaction_flag','block_timestamp_str']]
    y_data.set_index('block_timestamp_str', inplace=True)

    return X_data, y_data


def feature_selection_univ(data):
   data.drop(['block_number','gas_price','receipt_effective_gas_price','in_degree_adr_to','degree_centrality_to', 'out_degree_adr_to','degree_centrality_from','out_degree_adr_from'],axis=1,inplace=True)
   return data