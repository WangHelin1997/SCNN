from network import *
import torch.utils.data as Data
import torchvision
import torch.nn.functional as F
from data_process import *
import os

model = WaveMsNet_fixed_logmel(phase=2)
model = torch.load('../model/WaveMsNet_fixed_logmel_phase1_fold0_epoch2.pkl')
model2 = torch.load('../model/WaveMsNet_fixed_logmel_phase2_fold0_epoch2.pkl')
params=model.state_dict()
params2=model2.state_dict()
for k,v in params.items():
    print(k)    #打印网络中的变量名
    print(v)
pas = list(model.named_parameters())
(name, para) = pas[2]
print(name)
for i in range(0, 40):
    (name, para) = pas[i]
    print(name)
# for param in list(model.parameters())[:24]:
#     print(param)
# print('conv3.weight:'+str(params['conv3.weight']))
# print('conv1_1.bias:'+str(params['conv1_1.bias']))
# print('conv2_3.weight:'+str(params['conv2_3.weight']))
# print('conv2_3.weight:'+str(params2['conv2_3.weight']))
# print('conv2_3.bias:'+str(params['conv2_3.bias']))
# print('conv2_3.bias:'+str(params2['conv2_3.bias']))
# print('bn2_3.weight:'+str(params['bn2_3.weight']))
# print('bn2_3.weight:'+str(params2['bn2_3.weight']))
# print('bn2_3.bias:'+str(params['bn2_3.bias']))
# print('bn2_3.bias:'+str(params2['bn2_3.bias']))
# print('bn2_3.num_batches_tracked:'+str(params['bn2_3.num_batches_tracked']))
# print('bn2_3.num_batches_tracked:'+str(params2['bn2_3.num_batches_tracked']))
#
#
# print('bn2_3.running_mean:'+str(params['bn2_3.running_mean']))
# print('bn2_3.running_mean:'+str(params2['bn2_3.running_mean']))
# print('bn2_3.running_var:'+str(params['bn2_3.running_var']))
# print('bn2_3.running_var:'+str(params2['bn2_3.running_var']))
