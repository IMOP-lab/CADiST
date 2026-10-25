### script for test model parameters and FLOPs

import torch
import time
import os
from model import get3dmodel
from thop import profile, clever_format
import numpy as np


os.environ['CUDA_VISIBLE_DEVICES'] = '0'  #'0,1,2,3,4,5,6,7'

network_name = 'CADiST'

device = torch.device("cuda:0")
model = get3dmodel(network_name, 1, 5).to(device)  # Make sure this line initializes your model correctly

model.eval()
print('network_name',network_name)



dummy_input = torch.randn(1, 1, 96, 96, 96, device=device)
flops, params = profile(model, (dummy_input,))
with torch.no_grad():
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
# flops, params = clever_format([flops, params], "%.2f")
print('params: ', params, 'flops: ', flops)
print('params: %.2f M, flops: %.2f G' % (params / 1000000.0, flops / 1000000000.0))



