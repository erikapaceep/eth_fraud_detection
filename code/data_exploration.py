import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# July 2017 data

eth_20170701 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_trans_20170701.csv")
eth_20170702 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_trans_20170702.csv")
eth_20170703 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_trans_20170703.csv")
eth_20170704 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_trans_20170704.csv")
eth_20170705 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_trans_20170705.csv")
eth_20170706 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_trans_20170706.csv")
eth_20170707 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_trans_20170707.csv")
eth_20170708 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_trans_20170708.csv")
eth_20170713 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/eth_trans_20170713.csv")

historical = pd.concat([eth_20170701,eth_20170702,eth_20170703,
                        eth_20170704,eth_20170705,eth_20170706,
                        eth_20170707,eth_20170708,eth_20170713])

print(historical['block_timestamp'].head())
print(historical['block_timestamp'].tail())

# Remove duplicated rows
historical_clean = historical.drop_duplicates()
print(len(historical_clean))

historical['block_timestamp_str'] = historical['block_timestamp'].astype(str)

historical['date'] = historical['block_timestamp_str'].str.slice(start=0,stop=10)

historical['dates'] = pd.to_datetime(historical['date'], format='%Y-%m-%d')

historical['year'] = historical['block_timestamp_str'].str.slice(start=0,stop=4)
print(historical['year'])

historical['month'] = historical['block_timestamp_str'].str.slice(start=5,stop=7)
historical['month'] = pd.to_numeric(historical['month'])

historical['day'] = historical['block_timestamp_str'].str.slice(start=8,stop=10)
historical['day'] = pd.to_numeric(historical['day'])

historical.sort_values(by=['dates'],ascending=False, inplace=True)


# count transactions per date
number_trans_day = historical['dates'].value_counts().sort_index()
number_trans_day.to_csv("number_trans_day.csv")
number_trans_day_df = pd.DataFrame(number_trans_day)
number_trans_day_df = number_trans_day_df.reset_index()
number_trans_day_df['index'] = pd.to_datetime(number_trans_day_df['index'], format='%Y-%m-%d')
plt.xticks(number_trans_day_df['index'])
plt.hist(number_trans_day_df['dates'], bins=len(number_trans_day_df))


plt.show()

fig, ax = plt.subplots()
ax.bar(number_trans_day_df['index'], number_trans_day_df['dates'], edgecolor="k")
ax.set_xticks(number_trans_day_df['index'], rotation=45)
ax.set_xticklabels(number_trans_day_df['index'], rotation=45)

plt.show()
breakpoint()

print('number transaction per day')
print(number_trans_day)
# count transaction per months

# 2018
eth_transaction_2018 = pd.read_csv("C:/Users/erika/PycharmProjects/MScProject/ETH/bq-results-2018-07-26.csv", low_memory=False)

# 2019
eth_transaction_2019 = pd.read_csv("C:/Users/erika/PycharmProjects/MScProject/ETH/bq-results-2019-04_2019-06.csv", low_memory=False)

# 2020


# define data

eth_transaction = eth_transaction_2019


#print(eth_transaction.dtypes)
#eth_transaction['value'] = pd.to_numeric(eth_transaction['value']).astype('Int64')
#print(eth_transaction.dtypes)
print(eth_transaction.head())

print(eth_transaction.columns)

#eth_transaction['value_div_gas'] = eth_transaction['value']/eth_transaction['gas']
#plt.hist(eth_transaction['value_div_gas'])

#x = eth_transaction['value']
#print(x.columns)


# compute how many transaction per block
print()
# check the freq of the recepit status: which is a binary var 1 (success) 0 (failure)
print('Recepit status:')
print(eth_transaction['receipt_status'].value_counts())
#plt.hist(eth_transaction['receipt_status'])
#plt.show()

# Check how many blocks
print('Check how many block Number:')
unique_block = pd.unique(eth_transaction['block_number'])
print(len(unique_block))

# Check how many blocks in the dataset
print('Count how many transaction per block Number:')
transaction_in_block = eth_transaction['block_number'].value_counts()
print(transaction_in_block)
plt.hist(transaction_in_block)
plt.title('Transactions in a block (10 Apr 2023)')
plt.xlabel('Number of transaction')
plt.ylabel('Number of blocks')
plt.show()

print('Distribution of fees ...')

print('Average transaction per block')


# Check if there are different transaction types
print('Transaction type:')
print(eth_transaction['transaction_type'].value_counts())

# Check correlation between numerical variables
# Numerical variables: 'value','gas','gas_price','receipt_cumulative_gas_used','receipt_gas_used','max_fee_per_gas','max_priority_fee_per_gas','receipt_effective_gas_price'
# Categorical variables: 'nonce','transaction_index','receipt_status','block_number','transaction_type'

# Correlation between numerical values
numerical_feature = eth_transaction[['value','gas','gas_price','receipt_cumulative_gas_used','receipt_gas_used','max_fee_per_gas','max_priority_fee_per_gas','receipt_effective_gas_price']]
corr_num = numerical_feature.corr(numeric_only = True)
sns.heatmap(corr_num,vmin=-1, vmax=1, annot=True)
plt.show()

# Correlation between numerical + categorical values
numerical_feature = eth_transaction[['value','gas','gas_price','receipt_cumulative_gas_used','receipt_gas_used','max_fee_per_gas','max_priority_fee_per_gas','receipt_effective_gas_price','nonce','transaction_index','receipt_status','block_number','transaction_type']]
corr_num = numerical_feature.corr(numeric_only = True)
sns.heatmap(corr_num,vmin=-1, vmax=1, annot=True)
plt.show()


# count number of address to
from_address = pd.DataFrame(pd.unique(eth_transaction['from_address']), columns=['from_address_count'])

print('the number of address from is:',len(from_address), 'which means on average :',len(eth_transaction['from_address'])/len(from_address),' transctions per address')
# count number of address from
to_address = pd.unique(eth_transaction['to_address'])
print('the number of address to is:',len(to_address), 'which means on average :',len(eth_transaction['to_address'])/len(to_address),' transctions per address')

# Distribution of transactions per address 'from'
transaction_per_address_from = eth_transaction['from_address'].value_counts()
print('transaction per address from')
print(type(transaction_per_address_from))
print(transaction_per_address_from)


#res = sns.distplot(transaction_per_address_from)
#plt.show()

#plt.hist(transaction_per_address_from)
#plt.xticks(np.arange(0, len(from_address)+1, 100))
#plt.ylim(0, max(transaction_per_address_from))
#plt.title('Transactions per Address _from (10 Apr 2023)')
#plt.xlabel('Number of addresses')
#plt.ylabel('Number of transaction')
#plt.show()

# transform narray
print('Address from with 1 transaction:')
#from_address_pd = pd.DataFrame(transaction_per_address_from['from_address_count'], columns=['from_address'])
from_address_eq_one = transaction_per_address_from[transaction_per_address_from == 1]
print(len(from_address_eq_one))
print(len(from_address_eq_one)/len(transaction_per_address_from))

print('Address from with less than 10 transaction:')
from_address_less_ten = transaction_per_address_from[transaction_per_address_from <= 10]
print(len(from_address_less_ten))
print(len(from_address_less_ten)/len(transaction_per_address_from))

print('Address from with more than 10 transaction:',)
from_address_gt_ten = transaction_per_address_from[transaction_per_address_from > 10]
print(len(from_address_gt_ten))
print(len(from_address_gt_ten)/len(transaction_per_address_from))

# Distribution of transactions per address 'to'
transaction_per_address_to = eth_transaction['to_address'].value_counts()
print(transaction_per_address_to)
print(len(from_address_less_ten)/len(transaction_per_address_from))

plt.hist(transaction_per_address_to)
plt.title('Transactions per Address _to (10 Apr 2023)')
plt.xlabel('Number of transaction')
plt.ylabel('Number of addresses')
plt.show()

