import torch
import torch.nn as nn

class GaussianProcessRegression(nn.Module):
    def __init__(self, input_dim):
        super(GaussianProcessRegression, self).__init__()

        self.input_dim = input_dim

        self.gpr_layer = nn.Sequential(
            nn.Linear(self.input_dim, 512),  
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        # x: [batch, 1, 49]
        x = x.view(-1, self.input_dim) 
        return x

input_dim = 49
gpr_model = GaussianProcessRegression(input_dim)

print(gpr_model)
