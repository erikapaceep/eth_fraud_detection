import torch
import torch.nn as nn

class LSTM_block(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM_block, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, batch_first=True)

    def forward(self,x):
        h_0 = torch.zeros(size = (1,x.shape[0], self.hidden_size))
        c_0 = torch.zeros(size = (1, x.shape[0], self.hidden_size))
        self.output, (self.h, self.c) = self.lstm(x, (h_0,c_0))
        return self.output, (self.h, self.c)


class Linear_block(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(Linear_block, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.lin = nn.Linear(self.input_size, self.output_size)
    
    def forward(self,x):
        self.output = self.lin(x)
        return self.output
    
class Encoder_block(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder_block,self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm_block = LSTM_block(self.input_size, self.hidden_size)
        self.lin_block = Linear_block(self.hidden_size, self.output_size)

        # self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        output, (_,_) = self.lstm_block(x)
        #output = self.dropout(output)
        output = self.lin_block(output)
        return output

class Decoder_block(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Decoder_block, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm_block = LSTM_block(self.input_size, self.hidden_size)
        self.lin_block = Linear_block(self.hidden_size, self.output_size)

        # self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):
        output, (_,_) = self.lstm_block(x)
        #output = self.dropout(output)
        output = self.lin_block(output)
        return output

class Predictor_block(torch.nn.Module):
    def __init__(self, input_size):
        super(Predictor_block,self).__init__()

        self.input_size = input_size
        self.hidden_size = input_size
        
        self.lstm_predictor = self.lstm = nn.LSTM(self.input_size, self.hidden_size, batch_first=True)

    def forward(self, x_0, n_steps):

        h_0 = torch.zeros(size = (1, x_0.shape[0], self.hidden_size))
        c_0 = torch.zeros(size=(1, x_0.shape[0], self.hidden_size))

        x, (h,c) = self.lstm_predictor(x_0, (h_0,c_0))

        self.prediction = h.swapaxes(0,1)
        next_h = h
        next_c = c

        for _ in range(n_steps):
            _, (next_h, next_c) = self.lstm_predictor(next_h.swapaxes(0,1), (next_h, next_c))
            self.prediction = torch.cat((self.prediction, next_h.swapaxes(0,1)), dim=1)
        return self.prediction