import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
# from main_gpt2 import mask_input

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

def main_tta(target_loader, model, regressor, optimizer, device, mask_ratio=0.15, num_epochs=5):
    model.train()
    regressor.eval()
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        total_loss = 0
        for i, batch in enumerate(target_loader):
            src = batch[0].to(device)
            tgt = src.clone()
            src = mask_input(src, mask_ratio).to(device)
            optimizer.zero_grad()
            output, _ = model(src)
            loss = criterion(output, tgt)
            # print(loss)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], TTA Reconstruction Loss: {total_loss / len(target_loader)}")

        # Evaluate on target loader
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for i, batch in enumerate(target_loader):
                src = batch[0].to(device)
                labels = batch[1].to(device)

                _, features = model(src)
                
                output = regressor(features)
                all_preds.append(output.squeeze(-1).cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        mae = mean_absolute_error(all_labels, all_preds)
        rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
        print(f"Target Loader - MAE: {mae}, RMSE: {rmse}")