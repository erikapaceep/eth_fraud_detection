from matplotlib import pyplot as plt
from lpc_ad_libs import *
from lpc_ad_model import *
from  torch.utils.data import DataLoader
from data_preparation import *
from datetime import datetime
import os
 
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print('Using device:', device)

print("Creating experiment folders..")
parent_folder="experiments"
folder_name = datetime.now().strftime('%d_%m_%Y___%H_%M')
folder_path = os.path.join(parent_folder,folder_name)
 

os.makedirs( os.path.join(folder_path,"outputs"), exist_ok=True)
os.makedirs( os.path.join(folder_path,"state_dict","encoder"), exist_ok=True)
os.makedirs( os.path.join(folder_path,"state_dict","decoder"), exist_ok=True)
os.makedirs( os.path.join(folder_path,"state_dict","predictor"), exist_ok=True)

 
print("Loading train data..")
#train_data_path =  "OmniAnomaly-master\processed\machine-1-2_train.pkl"
#with open(train_data_path, "rb") as f:
#    train_data =  torch.from_numpy(pickle.load(f))

train_data_df = train_data_loader()
df = data_pre_processing(train_data_df)
df  = df [['nonce', 'transaction_index',
       'value', 'gas', 'gas_price', 'receipt_cumulative_gas_used',
       'receipt_gas_used', 'block_number','receipt_effective_gas_price', 
       'dates', 'gas_price_unit','value_div_gas', 'from_address_count', 'to_address_count',
       'block_count', 'degree_centrality_from', 'degree_centrality_to',
       'in_degree_adr_to', 'out_degree_adr_to', 'in_degree_adr_from','out_degree_adr_from','transaction_flag']]

print(df.index)
print(df.head())
df_np = df.to_numpy(dtype='float32', na_value=np.nan)
print(df_np)
train_data = torch.from_numpy(df_np)


x=0
#print("Loading test data..")
#test_data_path =  "OmniAnomaly-master\processed\machine-1-2_test.pkl"
#with open(test_data_path, "rb") as f:
#    test_data =  torch.from_numpy(pickle.load(f))
