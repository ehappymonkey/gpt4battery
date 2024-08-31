import matplotlib.pyplot as plt
import numpy as np
import torch


def drawDegradation(target_loader_test, model, regressor, device, model_name = None):
        labels = []
        preds = []
        for i, data in enumerate(target_loader_test):
                src, label = data[0], data[1]
                src = src.float().to(device)
                label = label.float().to(device)

                if model_name =='transformer' or 'lstm' or 'gpt2' or 'gru':
                     _, features = model(src)

                # if model_name =='gpt2':
                #     means = src.mean(1, keepdim=True).detach()
                #     src = src - means
                #     stdev = torch.sqrt(torch.var(src, dim=1, keepdim=True, unbiased=False) + 1e-5)
                #     src /= stdev
                #     # Linear transformation and positional encoding
                #     src = model.input_linear(src)
                #     src = model.pos_encoder(src)
                #     if model.PPA:
                #         # Add the soft prompt to the beginning of the input sequence
                #         batch_size = src.size(0)
                #         soft_prompt_expanded = model.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)
                #         src_with_prompt = torch.cat([soft_prompt_expanded, src], dim=1)
                #         # Use GPT-2 model
                #         features = model.gpt2(inputs_embeds=src_with_prompt).last_hidden_state[:, model.soft_prompt.size(0):, :].mean(dim=1)
                #     else:
                #         # Extract features using the fixed GPT-2 encoder
                #         features = model.gpt2(inputs_embeds=src).last_hidden_state.mean(dim=1)  
                #     features = features * stdev.squeeze(1) + means.squeeze(1)

                output = regressor(features)
                labels.append(label.detach().cpu().numpy())
                preds.append(output.squeeze(-1).detach().cpu().numpy())

        labels =  [i for ii in labels for i in ii]
        preds =  [i for ii in preds for i in ii]
        labels = np.array(labels)
        preds = np.array(preds)
        print(labels.shape)
        print(preds.shape)
        
        x = np.arange(labels.shape[0])
        fig, ax = plt.subplots()

        scale = 200.0 * np.random.rand(1419)
        ax.scatter(x, preds,  label='Pred', 
                alpha=0.2, edgecolors='none')

        ax.scatter(x, labels, label='True',
                alpha=0.2, edgecolors='none')

        ax.set_xlabel('Time')
        ax.set_ylabel('SOH')
        # ax.set_title('GOTION')
        ax.legend()
        ax.grid(True) 