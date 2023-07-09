
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

eth_transaction_2023 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/bq-results-20230702-to_address_malicious.csv")

print(eth_transaction_2023.head())

eth_2017_07_atk = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/bq-results-20170719-20170722.csv")

print(eth_2017_07_atk.head())

eth_2018_03 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/bq-results-20180309_20180311.csv")

print(eth_2018_03.head())

eth_2018_03_atk = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/bq-results-20180312_20180312.csv")

print(eth_2018_03_atk.head())

eth_2017_07 = pd.read_csv("https://s3.eu-central-2.wasabisys.com/ethblockchain/bq-results-2017-07_transaction.csv")

print(eth_2017_07.head())
