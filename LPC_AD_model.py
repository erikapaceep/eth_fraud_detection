import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
import seaborn as sns


class Encoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size,embedding_size, num_layers=1, dropout=0, bidirectional=False):

        super(Encoder,self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.embedding_size)

    def forward(self, x):
        h_0 = torch.zeros(size=(1,x.shape[0], self.hidden_size))
        c_0 = torch.zeros(size=(1,x.shape[0], self.hidden_size))

        lstm_out, (h_n,c_n) = self.lstm(x, (h_0,c_0))
        output = self.linear(lstm_out)

        return lstm_out, (h_n,c_n), output

class Decoder(torch.nn.Module):
    def __init__(self,output_size, hidden_size, embedding_size, num_layers=1, dropout=0, bidirectional=False):

        super(Decoder,self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self,x):

        h_0 = torch.zeros(size=(1,x.shape[0],self.hidden_size))
        c_0 = torch.zeros(size=(1, x.shape[0], self.hidden_size))

        lstm_output, (h_n, c_n) = self.lstm(x, (h_0,c_0))
        output = self.linear(lstm_output)

        return lstm_output, (h_n, c_n), output


class Predictor(nn.Module):

    def __init__(self,input_size, output_size):
        super(Predictor,self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.lstm = nn.LSTM(self.input_size, self.output_size, batch_first=True)

    def forward(self,x,HORIZON=2):

        output=[]
        h_0 = torch.zeros(size=(1,x.shape[0],self.output_size))
        c_0 = torch.zeros(size=(1, x.shape[0], self.output_size))

        for i in range(HORIZON):
            if i == 0:
                x, (h,c) = self.lstm(x, (h_0,c_0))
                output = h.swapaxes(0,1)
            else:
                x, (h, c) = self.lstm(x, (h, c))
                output = torch.cat((output, h.swapaxes(0,1)), dim=1)

        return output, (h,c)




train = pd.read_csv('C:/Users/erika/PycharmProjects/pythonProject3/machine-1-1.txt', header = None)
test = pd.read_csv('C:/Users/erika/PycharmProjects/pythonProject3/machine-1-1_test.txt', header = None)

print(train.shape)

scaler = StandardScaler()
scaler = scaler.fit(train)
data_scaled = scaler.transform(train)
test_data_scaled = scaler.transform(test)

#print(data_scaled[0:5,:])

train_X = []
train_Y = []
test_X = []

hist_window = 12
future_window = 2
M = 16

for i in range(hist_window,len(data_scaled)-future_window+1):
    train_X.append(data_scaled[i-hist_window:i,0:data_scaled.shape[1]])
    train_Y.append(data_scaled[i:i+future_window,0:data_scaled.shape[1]])

for i in range(hist_window, len(test_data_scaled) - future_window + 1):
    test_X.append(test_data_scaled[i - hist_window:i, 0:test_data_scaled.shape[1]])

train_X = np.array(train_X)
train_Y = np.array(train_Y)
test_X = np.array(test_X)

X_train_torch = torch.from_numpy(train_X).type(torch.Tensor)
Y_train_torch = torch.from_numpy(train_Y).type(torch.Tensor)
X_test_torch = torch.from_numpy(test_X).type(torch.Tensor)

print('X train torch shape:', X_train_torch.shape)
print('Y train torch shape:', Y_train_torch.shape)
print('X test torch shape :', X_test_torch.shape)

# training loop
encoder = Encoder(input_size=X_train_torch.shape[2], hidden_size=M, embedding_size=int(M/2))
decoder = Decoder(output_size= X_train_torch.shape[2], embedding_size=int(M/2), hidden_size=M)
predictor = Predictor(input_size=int(M/2),output_size=int(M/2))


# Set up a loss function
loss_fn = torch.nn.MSELoss()
# Set up an optimizer
encoder_optimizer = torch.optim.Adam(params=encoder.parameters(), lr=0.1)
decoder_optimizer = torch.optim.Adam(params=decoder.parameters(), lr=0.1)
predictor_optimizer = torch.optim.Adam(params=predictor.parameters(), lr=0.1)

train_loss_seq = []
test_loss_seq = []

# Training loop
epochs = 10
WINDOW = 10
for epoch in range(epochs):

    # set the model to training mode
    encoder.train()
    decoder.train()
    predictor.train()

    # 1. Forward pass (perform the forward propagation
    # Encoder
    lstm_out, (h_n,c_n), latent = encoder(X_train_torch)

    # Predictor + Perturbation
    # Prediction
    latent_window = latent[:,:WINDOW,:]
    latent_horizon = latent[:,WINDOW:,:]
    prediction, (h_n, c_n) = predictor(latent_window)

    #Perturbation
    error = torch.normal(0,1,size=latent_horizon.shape)
    prediction_perturb = (error*torch.abs((prediction - latent_horizon))) + latent_horizon

    latent_perturb = torch.cat((latent_window,prediction_perturb), dim=1)

    # Decoder
    lstm_output, (h_n, c_n), decoder_output = decoder(latent)
    lstm_output_perturb, (h_n_pert, c_n_pert), decoder_out_perturb = decoder(latent_perturb)

    # 2. Calculate the loss
    # Option A
    loss_past = loss_fn(decoder_out_perturb[:,:WINDOW,:],X_train_torch[:,:WINDOW,:])
    loss_future = loss_fn(decoder_out_perturb[:,WINDOW:,:],X_train_torch[:,WINDOW:,:])
    # check this
    loss_perturbated = loss_fn(decoder_out_perturb[:,WINDOW:,:],X_train_torch[:,WINDOW:,:])
    loss = loss_past + loss_future + loss_perturbated

    # Option B
    loss = loss_fn(decoder_out_perturb,X_train_torch)
    train_loss_seq.append(loss.detach().numpy())
    print(f'Training Loss for epoch {epoch}: {loss}')

    # 3. Optimizer zero grad
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    predictor_optimizer.zero_grad()

    # 4. Perform backpropagation
    loss.backward()

    # 5. Set up optimizer (perfrom Adam optimization)
    encoder_optimizer.step()
    decoder_optimizer.step()
    predictor_optimizer.step()

    if epoch % 10 == 0:
        # Compute the distance between the horizon window and the reconstructed
        horizon_window = X_train_torch[:,WINDOW:,:]
        anomaly_score = torch.cdist(decoder_out_perturb[:,WINDOW:,:], horizon_window, p=2)
        anomaly_score_mean = torch.mean(torch.mean(anomaly_score, dim=2), dim=1)
        dist = torch.zeros(size=(horizon_window.shape[0],horizon_window.shape[1],1))

        anomlay_score_np = anomaly_score_mean.detach().numpy()
        anomaly_df = pd.DataFrame({'Anomaly score':anomlay_score_np})

        sns.displot(anomaly_df['Anomaly score'], kind="kde")

        import matplotlib.pyplot as plt
        plt.show()

# testing loop
    ### Testing
    encoder.eval()  # put the model in evaluation mode for testing (inference)
    decoder.eval()
    predictor.eval()

    with torch.inference_mode():

        # Encoder
        lstm_out, (h_n, c_n), latent = encoder(X_test_torch)

        # Predictor + Perturbation
        # Prediction
        latent_window = latent[:, :WINDOW, :]
        latent_horizon = latent[:, WINDOW:, :]
        prediction, (h_n, c_n) = predictor(latent_window)

        # Perturbation
        error = torch.normal(0, 1, size=latent_horizon.shape)
        prediction_perturb = (error * torch.abs((prediction - latent_horizon))) + latent_horizon

        latent_perturb = torch.cat((latent_window, prediction_perturb), dim=1)

        # Decoder
        lstm_output, (h_n, c_n), decoder_output = decoder(latent)
        lstm_output_perturb, (h_n_pert, c_n_pert), decoder_out_per_test = decoder(latent_perturb)

        # 2. Calculate the loss
        test_loss = loss_fn(decoder_out_per_test, X_test_torch)
        test_loss_seq.append(test_loss.detach().numpy())

    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Train loss: {loss} | Test loss: {test_loss}")


plt.plot(train_loss_seq, label='train loss')
plt.plot(test_loss_seq, label='test loss')
plt.legend()
plt.title('Train-test loss')
plt.show()
