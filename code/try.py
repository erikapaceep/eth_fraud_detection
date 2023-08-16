
import pandas as pd
import datetime
import os
import seaborn as sns

darklist = pd.read_csv('../darklist.csv',index_col=False, usecols=["address", "comment", "date;"])
print(darklist.head())

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils.utils import train_data_loader, test_data_loader, create_date, drop_smart_contract, clean_up_row, drop_missing_data, create_X_y, feature_selection_univ, data_pre_processing

# July 2017 data

#train_data = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_trans_20170701.csv")
#test_data = eth_06092017 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_06092017.csv")
train_data = train_data_loader()
test_data = test_data_loader()
print(f'train data : {len(train_data)}')
print(f'test data : {len(test_data)}')

# drop smart contracts
print('drop smart contracts')
train_no_smart = drop_smart_contract(train_data)
test_no_smart = drop_smart_contract(test_data)
print(f'train data : {len(train_no_smart)}')
print(f'test data : {len(test_no_smart)}')

# remove invalid hash (not starting with 0x)
print('remove invalid hash (not starting with 0x)')
train_no_err = clean_up_row(train_no_smart)
test_no_err = clean_up_row(test_no_smart)
print(f'train data : {len(train_no_err)}')
print(f'test data : {len(test_no_err)}')

# drop missing address + missing block time stamp -> invalid transactions
print('drop missing address + missing block time stamp -> invalid transactions')
train_no_invalid = drop_missing_data(train_no_err)
test_no_invalid = drop_missing_data(test_no_err)
print(f'train data : {len(train_no_invalid)}')
print(f'test data : {len(test_no_invalid)}')

# drop duplicated rows
print('drop duplicates')
train_nodup = train_no_invalid.drop_duplicates()
test_nodup = test_no_invalid.drop_duplicates()
print(f'train data : {len(train_nodup)}')
print(f'test data : {len(test_nodup)}')

# data pre processing staring from data
train_data_prepro = data_pre_processing(train_data)
test_data_prepro = data_pre_processing(test_data)


# Data exploration 

print("Creating experiment folders..")
parent_folder="exploration_outputs"

os.makedirs( os.path.join(parent_folder,"charts"), exist_ok=True)

number_trans_day = train_data_prepro['dates'].value_counts().sort_index()
#number_trans_day.to_csv("number_trans_day.csv")
number_trans_day_df = pd.DataFrame(number_trans_day)
number_trans_day_train = number_trans_day_df.reset_index()

plt.figure(figsize=(12,7))
plt.xticks(rotation=90)
plt.hist(train_data_prepro['dates'], bins=len(number_trans_day_df), color='green')
plt.title("Ethereum transaction count per day - July 2017 (training data)")
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.3)
plt.savefig('exploration_outputs/charts/eth_trans_count_july2017.png')

plt.figure(figsize=(12,7))
plt.xticks(rotation=90)
plt.hist(test_data_prepro['dates'], bins=len(number_trans_day_df), color='green')
plt.title("Ethereum transaction count per day - Sept 2017 (training data)")
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.3)
plt.savefig('exploration_outputs/charts/eth_trans_count_sep2017.png')


# Number of trabsactions per block
train_trans_block= train_data_prepro.groupby('block_number')['hash'].count().rename('count_block')
test_trans_block= test_data_prepro.groupby('block_number')['hash'].count()
print(type(train_trans_block))
print('Number of blocks train', len(train_trans_block), 'average transactions per block', train_trans_block.mean())
print('Number of blocks test', len(test_trans_block), 'average transactions per block', test_trans_block.mean())


# Address - count number of addresses

# count number of 'address to' in July 2017 (train data)
train_from_address_count = train_data_prepro.groupby('from_address')['hash'].count().rename('address_count')
print('from address count,',len(train_from_address_count), 'average_transaction_per_address', train_from_address_count.mean())
test_from_address_count = test_data_prepro.groupby('from_address')['hash'].count().rename('address_count')
print('from address count,',len(test_from_address_count), 'average_transaction_per_address', test_from_address_count.mean())
train_to_address_count = train_data_prepro.groupby('to_address')['hash'].count().rename('address_count')
print('to address count,',len(train_to_address_count), 'average_transaction_per_address', train_to_address_count.mean())
test_to_address_count = test_data_prepro.groupby('to_address')['hash'].count().rename('address_count')
print('t0 address count,',len(test_to_address_count), 'average_transaction_per_address', test_to_address_count.mean())


fig, axes = plt.subplots(2,2, figsize=(12,8))
plt.subplots_adjust(wspace=0.3, hspace=0.3)
sns.kdeplot(data=train_from_address_count, fill=True, ax=axes[0,0], color='blue', label='address_from (train)')
sns.kdeplot(data=test_from_address_count, fill=True, ax=axes[0,1], color='red', label='address_from (test)')
sns.kdeplot(data=train_to_address_count, fill=True, ax=axes[1,0], color='blue', label='address_to (train)')
sns.kdeplot(data=test_to_address_count, fill=True, ax=axes[1,1], color='red', label='address_to (test)')


for row in axes:
    for ax in row:
        ax.set_xlabel("Transactions per address")
        ax.set_ylabel("Density")
        ax.legend()

axes[0, 0].set_title("Transactions per sending addresses distribution - train data")
axes[0, 1].set_title("Transactions per sending addresses distribution - test data")
axes[1, 0].set_title("Transactions per receving addresses distribution - train data")
axes[1, 1].set_title("Transactions per receving addresses distribution - train data")


#plt.title('transactions per address from distribution - train data ')
plt.savefig('exploration_outputs/charts/trans_address_from_train.png')

# count number of address from

from_address = pd.DataFrame(pd.unique(train_data_prepro['from_address']), columns=['from_address_count'])
print('in Jul 2017 the number of address from is:',len(from_address), 'which means on average :',len(train_data_prepro['from_address'])/len(from_address),' transctions per address')

to_address = pd.unique(train_data_prepro['to_address'])
print('in Jul 2017 the number of address to is:',len(to_address), 'which means on average :',len(train_data_prepro['to_address'])/len(to_address),' transctions per address')

# count number of 'address to' in Sept 2017 (test data)
from_address = pd.DataFrame(pd.unique(test_data_prepro['from_address']), columns=['from_address_count'])
print('in Sept 2017 the number of address from is:',len(from_address), 'which means on average :',len(test_data_prepro['from_address'])/len(from_address),' transctions per address')


# count number of address from
to_address = pd.unique(test_data_prepro['to_address'])
print('in Sept 2017 the number of address to is:',len(to_address), 'which means on average :',len(test_data_prepro['to_address'])/len(to_address),' transctions per address')

# Anomalies count
train_anomalies_count = train_data_prepro['transaction_flag'].value_counts()
test_anomalies_count = test_data_prepro['transaction_flag'].value_counts()
print('train anomalies count:', train_anomalies_count)
print('test anomalies count:', test_anomalies_count)

train_ano_from = train_data_prepro['mal_trans_from'].value_counts()
test_ano_from = test_data_prepro['mal_trans_from'].value_counts()
print('train anomalies from count:', train_ano_from)
print('test anomalies from count:', test_ano_from)

train_ano_to = train_data_prepro['mal_trans_to'].value_counts()
test_ano_to = test_data_prepro['mal_trans_to'].value_counts()
print('train anomalies to count:', train_ano_to)
print('test anomalies to count:', test_ano_to)