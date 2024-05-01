import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size = 1, hidden_size = 8, num_layers = 2, batch_first = True) #输入是 batch, time_step(seq), input_size(fea)
        self.linear = nn.Linear(8,1) 
        self.bn = nn.BatchNorm1d(num_features=49)
        self.relu = nn.ReLU()

    
    def forward(self, X):
        X, _ = self.lstm(X) # batch, seq, 8
        # X = self.bn(X)
        X = self.relu(X)
     
        X = self.linear(X) # batch, seq, 1
        # X = self.bn(X)
        X = self.relu(X)
     
        X = X[:, -1, :]
        X = X.squeeze(-1)
        return X # batch