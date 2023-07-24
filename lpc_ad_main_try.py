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
test_data = torch.from_numpy(df_np)
 

#M feature dimensione
#N embedding dimension
#K noise dimension
M = train_data.shape[-1]
N = 8
K = 1000

 
config_dict = { "EPOCHS":40 , "BATCH_SIZE" : 64 , "CRITERION":"MSE" , "PAST_LENGTH":10 , "FUTURE_LENGTH":2 ,
                "LR":0.005 , "FEATURES_DIMENSION":M ,"HIDDEN_SIZE_LSTM": int(M/2) , "HIDDEN_SIZE_FC": N  , "NOISE_DIM":K}

 
with open(os.path.join(folder_path,'config_dict.pickle'), 'wb') as handle:
    pickle.dump(config_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Creating train dataloader..")
train_dataset = TSDataset(mode="Training" , dataset=train_data , past_windows_length=config_dict["PAST_LENGTH"] , future_windows_length=config_dict["FUTURE_LENGTH"])
train_dataloader = DataLoader(train_dataset, batch_size=config_dict["BATCH_SIZE"],shuffle=True)
print(f"Total batches: {len(train_dataloader)}")

 
print("Creating test dataloader..")
test_dataset = TSDataset(mode="Test" , dataset=test_data , past_windows_length=config_dict["PAST_LENGTH"] , future_windows_length=config_dict["FUTURE_LENGTH"])
test_dataloader = DataLoader(test_dataset, batch_size=config_dict["BATCH_SIZE"],shuffle=True)
print(f"Total batches: {len(test_dataloader)}")

 
print("Loading models..")
encoder = Encoder_block(M , config_dict["HIDDEN_SIZE_LSTM"] , config_dict["HIDDEN_SIZE_FC"])
decoder = Decoder_block(config_dict["HIDDEN_SIZE_FC"] , config_dict["HIDDEN_SIZE_LSTM"] , M)
predictor = Predictor_block(config_dict["HIDDEN_SIZE_FC"])
print("Setting loss and optimizers..")

criterion = torch.nn.MSELoss(reduction = "mean")
encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=config_dict["LR"],weight_decay=1e-3)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=config_dict["LR"],weight_decay=1e-3)
predictor_optimizer = torch.optim.Adam(predictor.parameters(), lr=config_dict["LR"],weight_decay=1e-3)

print("Start training..")

train_loss_dict = {"redundancy_loss":[], "dependency_loss":[] , "general_loss":[]}
test_loss_dict = {"redundancy_loss":[], "dependency_loss":[] , "general_loss":[]}
for epoch in range(config_dict["EPOCHS"]):
    noise = torch.zeros(
        (config_dict["NOISE_DIM"], config_dict["FUTURE_LENGTH"], config_dict["HIDDEN_SIZE_FC"])).normal_(mean=0, std=1)
    redundancy_loss , dependency_loss , general_loss =  training_loop(epoch,
                      encoder, decoder, predictor,
                      encoder_optimizer, decoder_optimizer, predictor_optimizer,
                      criterion, train_dataloader,
                      config_dict, noise)

    train_loss_dict["redundancy_loss"].append(redundancy_loss)
    train_loss_dict["dependency_loss"].append(dependency_loss)
    train_loss_dict["general_loss"].append(general_loss)

    time.sleep(10)


    redundancy_loss , dependency_loss , general_loss = testing_loop(epoch ,
                                                      encoder, decoder, predictor,
                                                      criterion , test_dataloader , config_dict , noise)

    test_loss_dict["redundancy_loss"].append(redundancy_loss)
    test_loss_dict["dependency_loss"].append(dependency_loss)
    test_loss_dict["general_loss"].append(general_loss)

    time.sleep(10)


    if epoch % 5  == 0:

        print("Updating loss history..")
        plt.plot(range(len(train_loss_dict["redundancy_loss"])), train_loss_dict["redundancy_loss"] , label = "train_redundancy_loss" , color="red")
        plt.plot(range(len(train_loss_dict["dependency_loss"])), train_loss_dict["dependency_loss"], label="train_dependency_loss", color="orange")
        plt.plot(range(len(train_loss_dict["general_loss"])), train_loss_dict["general_loss"], label="train_general_loss", color="blue")
        plt.plot(range(len(test_loss_dict["redundancy_loss"])), test_loss_dict["redundancy_loss"] , label = "test_redundancy_loss" , color="purple")
        plt.plot(range(len(test_loss_dict["dependency_loss"])), test_loss_dict["dependency_loss"], label="test_dependency_loss", color="brown")
        plt.plot(range(len(test_loss_dict["general_loss"])), test_loss_dict["general_loss"], label="test_general_loss", color="green")


        plt.legend()
        plt.savefig(os.path.join(folder_path,"outputs","loss_history.png"))
        plt.close("all")

        with open(os.path.join(folder_path, 'outputs','train_loss_dict.pkl'), 'wb') as handle:
            pickle.dump(train_loss_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(folder_path, 'outputs','test_loss_dict.pkl'), 'wb') as handle:
            pickle.dump(test_loss_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("Saving models state dictionary..")
        torch.save(encoder.state_dict(), os.path.join(folder_path, "state_dict", "encoder",f"epoch_{epoch}"))
        torch.save(decoder.state_dict(), os.path.join(folder_path, "state_dict", "decoder", f"epoch_{epoch}"))
        torch.save(predictor.state_dict(), os.path.join(folder_path, "state_dict", "predictor", f"epoch_{epoch}"))

 
print("Training finished")

