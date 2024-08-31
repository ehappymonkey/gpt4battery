import numpy as np
from torch.utils.data import Dataset, DataLoader

from LoadBatteryData import * 

class Data(Dataset):
    def __init__(self, name, flag):  
        self.flag = flag
        if name == 'CALCE':
            self.label_valid = label_scr_CALCE[number_CALCE[-1]:-1]
            self.data_valid = data_scr_CALCE[number_CALCE[-1]:-1,:]
            self.label_train = label_scr_CALCE[1:(number_CALCE[-1]-1)]
            self.data_train = data_scr_CALCE[1:(number_CALCE[-1]-1),:]

        if name =='SANYO':
            self.label_valid = label_scr_SANYO[number_SANYO[-1]:-1]
            self.data_valid = data_scr_SANYO[number_SANYO[-1]:-1,:]
            self.label_train = label_scr_SANYO[1:(number_SANYO[-1]-1)]
            self.data_train = data_scr_SANYO[1:(number_SANYO[-1]-1),:]
        
        if name =='PANASONIC':
            self.label_valid = label_scr_PANASONIC[number_PANASONIC[-1]:-1]
            self.data_valid = data_scr_PANASONIC[number_PANASONIC[-1]:-1,:]
            self.label_train = label_scr_PANASONIC[1:(number_PANASONIC[-1]-1)]
            self.data_train = data_scr_PANASONIC[1:(number_PANASONIC[-1]-1),:]
            
        if name =='KOKAM':
            self.label_valid = label_scr_KOKAM[number_KOKAM[-1]:-1]
            self.data_valid = data_scr_KOKAM[number_KOKAM[-1]:-1,:]
            self.label_train = label_scr_KOKAM[1:(number_KOKAM[-1]-1)]
            self.data_train = data_scr_KOKAM[1:(number_KOKAM[-1]-1),:]
            
        if name =='GOTION':
            self.label_valid = label_scr_GOTION[number_GOTION[-1]:-1]
            self.data_valid = data_scr_GOTION[number_GOTION[-1]:-1,:]
            self.label_train = label_scr_GOTION[1:(number_GOTION[-1]-1)]
            self.data_train = data_scr_GOTION[1:(number_GOTION[-1]-1),:]


    def __getitem__(self, index):
        if self.flag == 'train':
            seq_x = self.data_train[index]
            seq_y = self.label_train[index]
            seq_x = seq_x.reshape([len(seq_x), 1])

        if self.flag == 'test':
            seq_x = self.data_valid[index]
            seq_y = self.label_valid[index]
            seq_x = seq_x.reshape([len(seq_x), 1])
        return seq_x, seq_y
        
    def __len__(self):
        if self.flag == 'train':
            return len(self.data_train)

        if self.flag =='test':
            return len(self.data_valid)

    

def data_provider(name, flag, shuffle_flag, batch_size): # flag = train/test
    data_set = Data(name=name, flag = flag)
    print(name , flag, len(data_set))
    data_loader = DataLoader(
        data_set, 
        batch_size = batch_size,
        shuffle = shuffle_flag,
    )
    return data_set, data_loader
