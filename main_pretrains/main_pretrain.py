import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import random


def mask_input(src, mask_ratio=0.15):
    """
    Randomly mask input sequence.
    Args:
        src: input sequence of shape [batch_size, seq_len, feature_dim]
        mask_ratio: percentage of the sequence to be masked
    """
    src = src.clone()
    batch_size, seq_len, feature_dim = src.size()
    mask = torch.rand(batch_size, seq_len) < mask_ratio
    src[mask] = 0
    return src

def main_pretrain(train_loader, test_loader, model, optimizer, device, mask_ratio=0.15, num_epochs=10):
    model.train()
    criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        train_loss = 0
        test_loss = 0
        for i, batch in enumerate(train_loader):
            src = batch[0].to(device)
            tgt = src.clone()
            src = mask_input(src, mask_ratio).to(device)
            optimizer.zero_grad()
            output, _ = model(src)
            loss = criterion(output, tgt)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        for i, batch in  enumerate(test_loader):
            src = batch[0].to(device)
            tgt = src.clone()
            src = mask_input(src, mask_ratio).to(device)
            output, _ = model(src)
            loss = criterion(output, tgt)
            test_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Reconstruction (train) Loss: {train_loss / len(train_loader)}, Reconstruction (test) Loss: {test_loss / len(test_loader)}")


# # model parameters
# input_dim = 1
# embed_dim = 128
# num_heads = 4
# num_layers = 2

# # training parameters
# mask_ratrio = 0.3
# num_epochs = 30
# LR = 1e-3

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
# f_g = TransformerSOH(input_dim, embed_dim, num_heads, num_layers).to(device)
# optimizer = optim.AdamW(f_g.parameters(), lr=LR)
# set_seed(0)
# main_pretrain(model=f_g, optimizer=optimizer, device=device,mask_ratio=mask_ratrio, num_epochs=num_epochs)