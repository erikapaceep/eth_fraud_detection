import pandas as pd
import numpy as np

def create_date(data_nodup):

    data_nodup['block_timestamp_str'] = data_nodup['block_timestamp'].astype(str)
    data_nodup['date'] = data_nodup['block_timestamp_str'].str.slice(start=0,stop=10)

    data_nodup['dates'] = pd.to_datetime(data_nodup['date'], format='%Y-%m-%d')

    data_nodup['year'] = data_nodup['block_timestamp_str'].str.slice(start=0,stop=4)


    data_nodup['month'] = data_nodup['block_timestamp_str'].str.slice(start=5,stop=7)
    data_nodup['month'] = pd.to_numeric(data_nodup['month'])

    data_nodup['day'] = data_nodup['block_timestamp_str'].str.slice(start=8,stop=10)
    data_nodup['day'] = pd.to_numeric(data_nodup['day'])

    data_nodup.sort_values(by=['dates'],ascending=False, inplace=True)
    
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