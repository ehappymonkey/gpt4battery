import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from main_pretrains.main_pretrain import mask_input

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
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], TTA Reconstruction Loss: {total_loss / len(target_loader)}")

        # Evaluate on target loader
        model.eval()
        regressor.eval()
        all_preds = []                                                                   
        all_labels = []
        with torch.no_grad():
            for i, batch in enumerate(target_loader):
                src = batch[0].to(device)
                labels = batch[1].to(device)
                # if model_name = transformer
                # features = model.transformer_encoder(model.input_linear(src))
                # if model_name = gpt2
                # features = model.gpt2(inputs_embeds=model.input_linear(src)).last_hidden_state
                # features = features.mean(dim=1)
                _, features = model(src)
                output = regressor(features)
                all_preds.append(output.squeeze(-1).cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        mae = mean_absolute_error(all_labels, all_preds)
        rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
        print(f"Target Loader - MAE: {mae}, RMSE: {rmse}")
        model.train()
