import pandas as pd
import numpy as np
import seaborn as sns

import json, re, requests

# this are the malicious addresses detected, guthub

url = "https://raw.githubusercontent.com/MyEtherWallet/ethereum-lists/master/src/addresses/addresses-darklist.json"
resp = requests.get(url)
data = json.loads(resp.text)


darklist = pd.DataFrame(columns=['address','comment','date'])

for i in data:
    new_row = pd.DataFrame(i, index=[0])
    darklist = pd.concat([new_row, darklist.loc[:]]).reset_index(drop=True)

print(darklist.head(10))


# Data
eth_transaction_17 = pd.read_csv("C:/Users/erika/PycharmProjects/MScProject/ETH/bq-results-20180408_20180408.csv", low_memory=False)
eth_transaction_16 = pd.read_csv("C:/Users/erika/PycharmProjects/MScProject/ETH/bq-results-20180312_20180312.csv", low_memory=False)
eth_transaction_15 = pd.read_csv("C:/Users/erika/PycharmProjects/MScProject/ETH/bq-results-20170719-20170722.csv", low_memory=False)
eth_transaction_1 = pd.read_csv("C:/Users/erika/PycharmProjects/MScProject/ETH/bq-results-2015-08-20_2015-08-09.csv", low_memory=False)
eth_transaction_2 = pd.read_csv("C:/Users/erika/PycharmProjects/MScProject/ETH/bq-results-2020-12-31__20-08-2020.csv", low_memory=False)
eth_transaction_3 = pd.read_csv("C:/Users/erika/PycharmProjects/MScProject/ETH/bq-results-20230514-2016-01-01_2016-06-30.csv", low_memory=False)
eth_transaction_4 = pd.read_csv("C:/Users/erika/PycharmProjects/MScProject/ETH/bq-results-20230514-2016-06-30_2016-07.csv", low_memory=False)
eth_transaction_5 = pd.read_csv("C:/Users/erika/PycharmProjects/MScProject/ETH/bq-results-31-07-2016.csv", low_memory=False)
eth_transaction_6 = pd.read_csv("C:/Users/erika/PycharmProjects/MScProject/ETH/bq-results-2016-09-30.csv", low_memory=False)
eth_transaction_7 = pd.read_csv("C:/Users/erika/PycharmProjects/MScProject/ETH/bq-results-2016-12-30.csv", low_memory=False)
eth_transaction_8 = pd.read_csv("C:/Users/erika/PycharmProjects/MScProject/ETH/bq-results-2017-01-31.csv", low_memory=False)
eth_transaction_9 = pd.read_csv("C:/Users/erika/PycharmProjects/MScProject/ETH/bq-results-2017-04-01.csv", low_memory=False)
eth_transaction_10 = pd.read_csv("C:/Users/erika/PycharmProjects/MScProject/ETH/bq-results-2017-05-25.csv", low_memory=False)
eth_transaction_11 = pd.read_csv("C:/Users/erika/PycharmProjects/MScProject/ETH/bq-results-20161001-20161031.csv", low_memory=False)
eth_transaction_12 = pd.read_csv("C:/Users/erika/PycharmProjects/MScProject/ETH/bq-results-20160801-20160831.csv", low_memory=False)
eth_transaction_13 = pd.read_csv("C:/Users/erika/PycharmProjects/MScProject/ETH/bq-results-20161101-20161130.csv", low_memory=False)
eth_transaction_14 = pd.read_csv("C:/Users/erika/PycharmProjects/MScProject/ETH/bq-results-20161201-20161231.csv", low_memory=False)
eth_transaction_2023 = pd.read_csv("C:/Users/erika/PycharmProjects/MScProject/ETH/bq-results-20230410-164146-1681144959832.csv")


historical = pd.concat([eth_transaction_1,eth_transaction_2, eth_transaction_3,
                        eth_transaction_4,eth_transaction_5,eth_transaction_6,
                        eth_transaction_7,eth_transaction_8,eth_transaction_9,
                        eth_transaction_10,eth_transaction_11,eth_transaction_12,
                        eth_transaction_13, eth_transaction_14,eth_transaction_15,
                        eth_transaction_17, eth_transaction_16,eth_transaction_2023])


print(historical.head())
print(historical.tail())

# link historical transaction data to malicious address

#historical_malicious = historical.merge(darklist, how='left', left_on='from_address', right_on='address')

historical_malicious = historical.merge(darklist, how='left', left_on='to_address', right_on='address')

print(historical_malicious['address'].unique())
