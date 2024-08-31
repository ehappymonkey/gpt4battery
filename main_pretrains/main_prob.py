import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def main_prob(source_loader, target_loader, combined_training, model, regressor, optimizer, device, num_epochs=10):
    model.eval()  # Freeze encoder parameters
    regressor.train()
    criterion = nn.MSELoss()
    
    # Extract first batch from target_loader
    target_data_iter = iter(target_loader)
    target_batch = next(target_data_iter)
    src_tar = target_batch[0].to(device)
    labels_tar = target_batch[1].to(device)
    labels_tar = torch.ones_like(labels_tar)

    for epoch in range(num_epochs):
        model.eval()
        regressor.train()
        total_loss = 0
        
        for i, batch in enumerate(source_loader):
            src = batch[0].to(device)
            labels = batch[1].to(device)

            if combined_training:
                # Concatenate source and target data
                src_combined = torch.cat((src, src_tar), dim=0)
                labels_combined = torch.cat((labels, labels_tar), dim=0)
                with torch.no_grad():
                    _, features = model(src_combined)
                # Train the regressor
                output = regressor(features)
                loss = criterion(output.squeeze(-1), labels_combined)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            else:
                with torch.no_grad():
                    _, features = model(src)
                # Train the regressor
                output = regressor(features)
                loss = criterion(output.squeeze(-1), labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Source Loader Loss: {total_loss / len(source_loader)}")
            
        
        # Evaluate on target loader
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in target_loader:
                src = batch[0].to(device)
                labels = batch[1].to(device)
                
                # Extract features using the fixed encoder
                # features = model.transformer_encoder(model.input_linear(src)).mean(dim=1)
                _, features = model(src)
                output = regressor(features)
                
                all_preds.append(output.squeeze(-1).cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        mae = mean_absolute_error(all_labels, all_preds)
        rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
        print(f"Target Loader - MAE: {mae}, RMSE: {rmse}")
