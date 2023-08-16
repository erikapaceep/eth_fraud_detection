import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils.utils import train_data_loader, test_data_loader, create_date, drop_smart_contract, clean_up_row, drop_missing_data, create_X_y, feature_selection_univ, data_pre_processing

# July 2017 data

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
number_trans_day = train_data_prepro['dates'].value_counts().sort_index()
#number_trans_day.to_csv("number_trans_day.csv")
number_trans_day_df = pd.DataFrame(number_trans_day)
number_trans_day_train = number_trans_day_df.reset_index()

plt.xticks(rotation=90)
plt.hist(train_data_prepro['dates'], bins=len(number_trans_day_df), color='green')
plt.title("Ethereum transaction count per day - July 2017 (training data)")
plt.show()


plt.xticks(rotation=90)
plt.hist(test_data_prepro['dates'], bins=len(number_trans_day_df), color='green')
plt.title("Ethereum transaction count per day - July 2017 (training data)")
plt.show()


# Number of trabsactions per block
train_trans_block= train_data_prepro.groupby(['block_number']).value_counts()
test_trans_block= test_data_prepro.groupby(['block_number']).value_counts()

# Address - count number of addresses

# count number of 'address to' in July 2017 (train data)
from_address = pd.DataFrame(pd.unique(train_data_prepro['from_address']), columns=['from_address_count'])
print('in Jul 2017 the number of address from is:',len(from_address), 'which means on average :',len(train_data_prepro['from_address'])/len(from_address),' transctions per address')
# count number of address from
to_address = pd.unique(train_data_prepro['to_address'])
print('in Jul 2017 the number of address to is:',len(to_address), 'which means on average :',len(train_data_prepro['to_address'])/len(to_address),' transctions per address')

# count number of 'address to' in Sept 2017 (test data)
from_address = pd.DataFrame(pd.unique(test_data_prepro['from_address']), columns=['from_address_count'])
print('in Jul 2017 the number of address from is:',len(from_address), 'which means on average :',len(test_data_prepro['from_address'])/len(from_address),' transctions per address')
# count number of address from
to_address = pd.unique(test_data_prepro['to_address'])
print('in Jul 2017 the number of address to is:',len(to_address), 'which means on average :',len(test_data_prepro['to_address'])/len(to_address),' transctions per address')

# Number of transactions flag as anomalies
