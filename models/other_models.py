import torch
import torch.nn as nn

import torch
import torch.nn as nn

class MLP_SOH(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(MLP_SOH, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # x = x.view(x.size(0), -1)  # Flatten the input
        return self.mlp(x).mean(dim=1)


from sklearn.ensemble import RandomForestRegressor
import numpy as np

class RandomForestSOH:
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        
    def fit(self, X_train, y_train):
        X_train_flat = X_train.reshape(X_train.shape[0], -1)  # Flatten the input
        self.model.fit(X_train_flat, y_train)

    def predict(self, X):
        X_flat = X.reshape(X.shape[0], -1)  # Flatten the input
        return self.model.predict(X_flat)


import lightgbm as lgb

class LightGBMSOH:
    def __init__(self, num_leaves=31, learning_rate=0.05, n_estimators=100):
        self.model = lgb.LGBMRegressor(num_leaves=num_leaves, learning_rate=learning_rate, n_estimators=n_estimators)

    def fit(self, X_train, y_train):
        X_train_flat = X_train.reshape(X_train.shape[0], -1)  # Flatten the input
        self.model.fit(X_train_flat, y_train)

    def predict(self, X):
        X_flat = X.reshape(X.shape[0], -1)  # Flatten the input
        return self.model.predict(X_flat)

