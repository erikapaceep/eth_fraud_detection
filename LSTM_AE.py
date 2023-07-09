import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


hist_window = 10
future_window = 2
M = 16 # this represent the number of hidden features # tried (64,32,16,8)

E = 1 # this represent the number of features
epochs = 30
learning_rate = 0.05

class Encoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size,embedding_size, num_layers=1, dropout=0.2, bidirectional=False):

        super(Encoder,self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.embedding_size)

    def forward(self, x, horizon=2):
        h_0 = torch.zeros(size=(1,x.shape[0], self.hidden_size))
        c_0 = torch.zeros(size=(1,x.shape[0], self.hidden_size))


        for i in range(horizon):
            if i == 0:
                lstm_out, (h_n,c_n) = self.lstm(x, (h_0,c_0))
                out = h_n.swapaxes(0,1)
            else:
                lstm_out, (h_n,c_n) = self.lstm(x, (h_n,c_n))
                out = torch.cat((out, h_n.swapaxes(0,1)), dim=1)

        output = self.linear(out)

        return out, (h_n,c_n), output



data = pd.read_csv('C:/Users/erika/PycharmProjects/MScProject/SPX.csv').set_index('Date')


close = pd.DataFrame(data['Close'])
volume = pd.DataFrame(data['Volume'])

data = close
# split into train test split
split = 0.8
train = data[:int(split*len(data))]
test = data[int(split*len(data)):]

scaler = StandardScaler()
scaler = scaler.fit(train)
train_scaled = scaler.transform(train)
test_scaled = scaler.transform(test)

#print(data_scaled[0:5,:])

train_X = []
train_Y = []
test_X = []
test_Y = []



for i in range(hist_window,len(train_scaled)-future_window+1):
    train_X.append(train_scaled[i-hist_window:i,0:train_scaled.shape[1]])
    train_Y.append(train_scaled[i:i+future_window,0:train_scaled.shape[1]])

for i in range(hist_window, len(test_scaled) - future_window + 1):
    test_X.append(test_scaled[i - hist_window:i, 0:test_scaled.shape[1]])
    test_Y.append(test_scaled[i:i+future_window,0:test.shape[1]])

train_X = np.array(train_X)
train_Y = np.array(train_Y)
test_X = np.array(test_X)
test_Y = np.array(test_Y)

X_train_torch = torch.from_numpy(train_X).type(torch.Tensor)
Y_train_torch = torch.from_numpy(train_Y).type(torch.Tensor)
X_test_torch = torch.from_numpy(test_X).type(torch.Tensor)
Y_test_torch = torch.from_numpy(test_Y).type(torch.Tensor)

print('X train torch shape:', X_train_torch.shape)
print('Y train torch shape:', Y_train_torch.shape)
print('X test torch shape :', X_test_torch.shape)

# training loop
encoder = Encoder(input_size=X_train_torch.shape[2], hidden_size=M, embedding_size=E)

# Set up a loss function
loss_fn = torch.nn.L1Loss()
# Set up an optimizer
encoder_optimizer = torch.optim.Adam(params=encoder.parameters(), lr=learning_rate)


train_loss_seq = []
test_loss_seq = []

# Training loop

for epoch in range(epochs):

    # set the model to training mode
    encoder.train()

    # 1. Forward pass (perform the forward propagation
    # Encoder
    lstm_out, (h_n,c_n), latent = encoder(X_train_torch)

    # 2. Calculate the loss
    loss = loss_fn(latent,Y_train_torch)
    train_loss_seq.append(loss.detach().numpy())
    print(f'Training Loss for epoch {epoch}: {loss}')

    # 3. Optimizer zero grad
    encoder_optimizer.zero_grad()


    # 4. Perform backpropagation
    loss.backward()

    # 5. Set up optimizer (perfrom Adam optimization)
    encoder_optimizer.step()

# testing loop
    ### Testing
    encoder.eval()  # put the model in evaluation mode for testing (inference)

    with torch.inference_mode():

        # Encoder
        lstm_out, (h_n, c_n), latent = encoder(X_test_torch)

        # 2. Calculate the loss
        test_loss = loss_fn(latent, Y_test_torch)
        test_loss_seq.append(test_loss.detach().numpy())

    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Train loss: {loss} | Test loss: {test_loss}")


plt.plot(train_loss_seq, label='train loss')
plt.plot(test_loss_seq, label='test loss')
plt.legend()
plt.title('Train-test loss')
plt.show()
