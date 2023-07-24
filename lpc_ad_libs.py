import pickle
import time
import torch
from torch.utils.data import Dataset

 
def training_loop(epoch_idx ,
                  encoder_model , decoder_model , predictor_model  ,
                  encoder_optimizer, decoder_optimizer, predictor_optimizer ,
                  criterion , training_loader , config_dict , noise):

    encoder_model.train()
    decoder_model.train()
    predictor_model.train()

    past_length = config_dict["PAST_LENGTH"]
    future_length = config_dict["FUTURE_LENGTH"]
    overall_length = past_length+future_length
    emb_feature_dim = config_dict["HIDDEN_SIZE_FC"]
    feature_dim = config_dict["FEATURES_DIMENSION"]
 
    running_general_loss = []
    running_redundancy_loss = []
    running_dependency_loss = []
    batch_time_list = []

    for i, (data) in enumerate(training_loader):
        start_time = time.time()
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        predictor_optimizer.zero_grad()

        past = data[:, :past_length]
        future = data[:, past_length:]
        encoded_data = encoder_model(data)
        past_encoded = encoded_data[:,:past_length]
        future_encoded = encoded_data[:, past_length:]

        decoded_data = decoder_model(encoded_data)
        past_decoded = decoded_data[:,:past_length]
        future_decoded = decoded_data[:, past_length:]

        pred_future_encoded = predictor_model(past_encoded , future_length-1)
        # apply the perturbation
        pred_future_encoded_pertubated = (noise[:,None]*torch.abs((pred_future_encoded-future_encoded))).swapaxes(0,1)

        past_encoded_repeat = torch.repeat_interleave(past_encoded[:,None], repeats=config_dict["NOISE_DIM"], dim=1)
        pred_encoded_data_perturbated = torch.cat((past_encoded_repeat , pred_future_encoded_pertubated) , dim=2)
        pred_encoded_data_perturbated = pred_encoded_data_perturbated.reshape(shape=[-1,overall_length,emb_feature_dim])

        pred_decoded_data_perturbated = decoder_model(pred_encoded_data_perturbated)
        pred_decoded_data_perturbated = pred_decoded_data_perturbated.reshape(shape=[-1,config_dict["NOISE_DIM"],overall_length,feature_dim])

        future_decoded_perturbated = pred_decoded_data_perturbated[:,:, past_length:]
        future_repeat = torch.repeat_interleave(future[:, None], repeats=config_dict["NOISE_DIM"], dim=1)

        redundancy_loss = criterion(future_decoded , future ) + criterion(past_decoded , past )
        dependency_loss = criterion(future_decoded_perturbated , future_repeat)
        loss = redundancy_loss + dependency_loss

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()
        predictor_optimizer.step()

        running_redundancy_loss.append(redundancy_loss.item())
        running_dependency_loss.append(dependency_loss.item())
        running_general_loss.append(running_redundancy_loss[-1] + running_dependency_loss[-1])

        last_batch_time = time.time() - start_time
        batch_time_list.append(last_batch_time)

        if i % 100 == 0 :
            print(f"{time.asctime(time.localtime(time.time()))} -------- TRAIN Epoch {epoch_idx} - Batch idx:{i} - Average Batch processing time : {(sum(batch_time_list)/len(batch_time_list)):.6f} seconds")

    redundancy_loss_mean = sum(running_redundancy_loss)/len(running_redundancy_loss)
    dependency_loss_mean = sum(running_dependency_loss) / len(running_dependency_loss)
    general_loss_mean = sum(running_general_loss) / len(running_general_loss)
    if epoch_idx % 1 == 0:
        print(f"{time.asctime(time.localtime(time.time()))} -------- TRAIN Epoch {epoch_idx} - Redundancy loss: {redundancy_loss_mean:.6f} - "
              f"Dependency loss: {dependency_loss_mean:.6f} - Loss :{general_loss_mean:.6f}")

    return redundancy_loss_mean ,dependency_loss_mean , general_loss_mean

 

def testing_loop(epoch_idx ,
                  encoder_model , decoder_model , predictor_model  ,
                  criterion , testing_loader , config_dict , noise):

    encoder_model.train()
    decoder_model.train()
    predictor_model.train()


    past_length = config_dict["PAST_LENGTH"]
    future_length = config_dict["FUTURE_LENGTH"]
    overall_length = past_length+future_length
    emb_feature_dim = config_dict["HIDDEN_SIZE_FC"]
    feature_dim = config_dict["FEATURES_DIMENSION"]

    running_general_loss = []
    running_redundancy_loss = []
    running_dependency_loss = []
    batch_time_list = []
    with torch.no_grad():
        for i, (data) in enumerate(testing_loader):
            start_time = time.time()
            past = data[:, :past_length]
            future = data[:, past_length:]
            encoded_data = encoder_model(data)
            past_encoded = encoded_data[:,:past_length]
            future_encoded = encoded_data[:, past_length:]

            decoded_data = decoder_model(encoded_data)
            past_decoded = decoded_data[:,:past_length]
            future_decoded = decoded_data[:, past_length:]

            pred_future_encoded = predictor_model(past_encoded , future_length-1)
            pred_future_encoded_pertubated = (noise[:,None]*torch.abs((pred_future_encoded-future_encoded))).swapaxes(0,1)

            past_encoded_repeat = torch.repeat_interleave(past_encoded[:,None], repeats=config_dict["NOISE_DIM"], dim=1)
            pred_encoded_data_perturbated = torch.cat((past_encoded_repeat , pred_future_encoded_pertubated) , dim=2)
            pred_encoded_data_perturbated = pred_encoded_data_perturbated.reshape(shape=[-1,overall_length,emb_feature_dim])

            pred_decoded_data_perturbated = decoder_model(pred_encoded_data_perturbated)
            pred_decoded_data_perturbated = pred_decoded_data_perturbated.reshape(shape=[-1,config_dict["NOISE_DIM"],overall_length,feature_dim])

            future_decoded_perturbated = pred_decoded_data_perturbated[:,:, past_length:]
            future_repeat = torch.repeat_interleave(future[:, None], repeats=config_dict["NOISE_DIM"], dim=1)

            redundancy_loss = criterion(future_decoded , future ) + criterion(past_decoded , past )
            dependency_loss = criterion(future_decoded_perturbated , future_repeat)

            running_redundancy_loss.append(redundancy_loss.item())
            running_dependency_loss.append(dependency_loss.item())
            running_general_loss.append(running_redundancy_loss[-1] + running_dependency_loss[-1])

            last_batch_time = time.time() - start_time
            batch_time_list.append(last_batch_time)

            if i % 100 == 0 :
                print(f"{time.asctime(time.localtime(time.time()))} -------- TEST Epoch {epoch_idx} - Batch idx:{i} - Average Batch processing time : {(sum(batch_time_list)/len(batch_time_list)):.5f} seconds")

        redundancy_loss_mean = sum(running_redundancy_loss)/len(running_redundancy_loss)
        dependency_loss_mean = sum(running_dependency_loss) / len(running_dependency_loss)
        general_loss_mean = sum(running_general_loss) / len(running_general_loss)
        if epoch_idx % 1 == 0:
            print(f"{time.asctime(time.localtime(time.time()))} -------- TEST Epoch {epoch_idx} - Redundancy loss: {redundancy_loss_mean:.6f} - "
                  f"Dependency loss: {dependency_loss_mean:.6} - Loss :{general_loss_mean:.6f}")
            
    return redundancy_loss_mean ,dependency_loss_mean , general_loss_mean

 

def load_data(data_path):
    with open(data_path , "rb") as f:
        dataset = pickle.load(f)
        return dataset

class TSDataset(Dataset):
    def __init__(self, mode, dataset, past_windows_length, future_windows_length):

        self.mode = mode
        self.dataset = dataset
        self.past_windows_length = past_windows_length
        self.future_windows_length = future_windows_length
        self.n_records = dataset.shape[0]
        self.n_features = dataset.shape[1]
        self.n_samples = self.n_records - self.past_windows_length - 1
        print(f"{self.mode} dataset has {self.n_records} records and {self.n_features} for each record")

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        start = idx
        stop = idx + self.past_windows_length + self.future_windows_length
        sample = self.dataset[start:stop]
        return sample
    
class TSDataset_txextracted(Dataset):
    def __init__(self, mode, dataset):

        self.mode = mode
        self.dataset = dataset
 
        self.n_samples = dataset.shape[0]
        self.n_features = dataset.shape[1]
        print(f"{self.mode} dataset has {self.n_records} records and {self.n_features} feature for each record")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        return sample


def evaluate_reconstruction_error(dataloader , criterion,  encoder,decoder,predictor ,config_dict):

    past_length = config_dict["PAST_LENGTH"]
    future_length = config_dict["FUTURE_LENGTH"]
    error = 0
    with torch.no_grad():
        for i,data in enumerate (dataloader):
            past = data[:,:past_length]
            future = data[:,-future_length:]

            past_encoded = encoder(past)
            pred_future_encoded = predictor(past_encoded , future_length-1)
            data_encoded = torch.cat((past_encoded,pred_future_encoded),dim=1)
            data_decoded = decoder(data_encoded)
            future_decoded = data_decoded[:,-future_length:]
            batch_error = criterion(future , future_decoded)
            if i == 0 :
                error = batch_error
            else:
                error = torch.cat((error,batch_error))

    return error.detach()

 

def calculate_threshold ( errors , z_vector):
    feature_dim = errors.shape[-1]
    #Error posses a window of 2 future prediction: how to report errors to a 1-D vector per feature dimension ?

    avg_error_values = (errors[:-1,1]+errors[1:,0])/2
    errors = torch.cat((errors[[0],0],avg_error_values ,errors[[-1],1]) , dim=0)

    epsilon_matrix = torch.zeros(size=[feature_dim ,len(z_vector) ])
    epsilon_matrix_processed = torch.zeros(size=[feature_dim ,len(z_vector) ])

    for i_dim in range(feature_dim):
        mean = torch.mean(errors[:, i_dim])
        std = torch.std(errors[:, i_dim])
        for i_z, z in enumerate(z_vector):
            epsilon_matrix[i_dim , i_z] = mean + z*std


    for i_dim in range(feature_dim):
        mean = torch.mean(errors[:, i_dim])
        std = torch.std(errors[:, i_dim])
        for i_z, z in enumerate(z_vector):
            eps = epsilon_matrix[i_dim , i_z]

            below_eps_list = torch.as_tensor( [x for x in errors[:, i_dim] if x<=eps] )
            above_eps_list = torch.as_tensor( [x for x in errors[:, i_dim] if x>eps] )

            delta_mean = torch.mean(below_eps_list)
            delta_std = torch.std(below_eps_list)
            e_above = torch.linalg.norm(above_eps_list)
            eps_processed = ( delta_mean/mean + delta_std/std ) / e_above
            epsilon_matrix_processed[i_dim , i_z] = eps_processed

    epsilon_idx = torch.argmax(epsilon_matrix_processed , dim = 1 )

    epsilon = []
    for i_dim , argmax_eps in enumerate(epsilon_idx):
        epsilon.append(epsilon_matrix[i_dim,argmax_eps])

    return epsilon

