import torch.nn as nn
import numpy as np

class framework(nn.Module):
    def __init__(self, FeatureNumber):
        super(framework, self).__init__()
       
        self.cnn = nn.Sequential()
        self.cnn.add_module('Conv1',nn.Conv1d(1, 32, 2, stride=2, padding=0))
        self.cnn.add_module('ReLU1', nn.ReLU())
        self.cnn.add_module('Conv2',nn.Conv1d(32, 64, 2, stride=2, padding=0))
        self.cnn.add_module('ReLU2', nn.ReLU())
        self.cnn.add_module('Conv3',nn.Conv1d(64, 128, 4, stride=1, padding=0))
        self.cnn.add_module('ReLU3', nn.ReLU())

        self.predictor = nn.Sequential()
        self.predictor.add_module('pre_Fc1', nn.Linear(FeatureNumber, 50))
        # self.predictor.add_module('pre_ReLU1', nn.ReLU())
        self.predictor.add_module('pre_Sigm1', nn.Sigmoid())
        
        self.reconstructor = nn.Sequential()
        self.reconstructor.add_module('recon_Fc1', nn.Linear(FeatureNumber, FeatureNumber))
        self.reconstructor.add_module('recon_ReLU1', nn.ReLU())

    def forward(self,X_src):

        fs0=self.cnn(X_src)
        # print(fs0.shape)
        a=fs0.shape[0]
        fs=fs0.view(a,1,-1)

        Pre_src=self.predictor(fs)  
        # print(Pre_src.shape)      
        
        return np.squeeze(Pre_src)
   

      
class RGHead(nn.Module):
    def __init__(self):
        super(RGHead, self).__init__()
        self.rghead = nn.Linear(50, 1)

    def forward(self, x):
        x = self.rghead(x)
        return np.squeeze(x)
    
    
class SSHead(nn.Module):
    def __init__(self):
        super(SSHead, self).__init__()
        self.sshead = nn.Linear(50, 49)

    def forward(self, x):
        x = self.sshead(x)
        return x
    

# class RGHead(nn.Module):
#     def __init__(self, nf=50, target_window=1, head_dropout=0):
#         super().__init__()
#         self.flatten = nn.Flatten(start_dim=-2)
#         self.linear = nn.Linear(nf, target_window)
#         self.dropout = nn.Dropout(head_dropout)

#     def forward(self, x):
#         x = self.flatten(x)
#         x = self.linear(x)
#         x = self.dropout(x)
#         return x