import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def evaluate(model, regressor, target_loader, model_name, device):
    model.eval()
    regressor.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for i, batch in enumerate(target_loader):
            src = batch[0].to(device)
            labels = batch[1].to(device)

            if model_name == 'transformer' or 'lstm' or 'gpt2' or 'gru':
                _, features = model(src)

            # if model_name == 'gpt2':
            #     # Instance normalization on src
            #     means = src.mean(1, keepdim=True).detach()
            #     src = src - means
            #     stdev = torch.sqrt(torch.var(src, dim=1, keepdim=True, unbiased=False) + 1e-5)
            #     src /= stdev

            #     # Linear transformation and positional encoding
            #     src = model.input_linear(src)
            #     src = model.pos_encoder(src)

            #     # Extract features using the fixed encoder (GPT-2)
            #     if model.PPA:
            #         # Add the soft prompt to the beginning of the input sequence
            #         batch_size = src.size(0)
            #         soft_prompt_expanded = model.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)
            #         src_with_prompt = torch.cat([soft_prompt_expanded, src], dim=1)
            #         # Use GPT-2 model
            #         features = model.gpt2(inputs_embeds=src_with_prompt).last_hidden_state[:, model.soft_prompt.size(0):, :].mean(dim=1)
            #     else:
            #         features = model.gpt2(inputs_embeds=src).last_hidden_state.mean(dim=1)
            #     features = features * stdev.squeeze(1) + means.squeeze(1)
            output = regressor(features)   
            all_preds.append(output.squeeze(-1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    mae = mean_absolute_error(all_labels, all_preds)
    rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
    
    return mae, rmse
