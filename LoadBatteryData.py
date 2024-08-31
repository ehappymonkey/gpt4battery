# -*- coding: utf-8 -*-
"""
Created on September 2022
@author: Jiahuan Lu
Import battery data
"""

import scipy.io as scio
import numpy as np



BatteryDatabase=scio.loadmat('BatteryDatabase.mat')

data_scr_CALCE=BatteryDatabase['data_scr_CALCE']
data_scr_GOTION=BatteryDatabase['data_scr_GOTION']
data_scr_PANASONIC=BatteryDatabase['data_scr_PANASONIC']
data_scr_KOKAM=BatteryDatabase['data_scr_KOKAM']
data_scr_SANYO=BatteryDatabase['data_scr_SANYO']

label_scr_CALCE=BatteryDatabase['label_scr_CALCE']
label_scr_GOTION=BatteryDatabase['label_scr_GOTION']
label_scr_PANASONIC=BatteryDatabase['label_scr_PANASONIC']
label_scr_KOKAM=BatteryDatabase['label_scr_KOKAM']
label_scr_SANYO=BatteryDatabase['label_scr_SANYO']

label_scr_CALCE=np.squeeze(label_scr_CALCE)
label_scr_GOTION=np.squeeze(label_scr_GOTION)
label_scr_PANASONIC=np.squeeze(label_scr_PANASONIC)
label_scr_KOKAM=np.squeeze(label_scr_KOKAM)
label_scr_SANYO=np.squeeze(label_scr_SANYO)

number_CALCE=BatteryDatabase['number_CALCE']
number_GOTION=BatteryDatabase['number_GOTION']
number_PANASONIC=BatteryDatabase['number_PANASONIC']
number_KOKAM=BatteryDatabase['number_KOKAM']
number_SANYO=BatteryDatabase['number_SANYO']

number_CALCE=np.squeeze(number_CALCE)
number_GOTION=np.squeeze(number_GOTION)
number_PANASONIC=np.squeeze(number_PANASONIC)
number_KOKAM=np.squeeze(number_KOKAM)
number_SANYO=np.squeeze(number_SANYO)

del BatteryDatabase








