"""
    Draw time and Logmel spectrogram for audio through single filter.

"""

import os
import pickle
import matplotlib.pyplot as plt
import librosa
import  librosa.display
import torch
import numpy as np
from util import *
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math
import scipy.io as io
# y,sr = librosa.load("F:/TUT-urban-acoustic-scenes-2018-development/audio/street_pedestrian-stockholm-157-4779-a.wav", sr=None)
y,sr = librosa.load("F:/audio/1-94036-A-22.wav", sr=None)
print(y)
model = torch.load('WaveMsNet_fixed_logmel_phase1_fold0_best.pkl', map_location='cpu')
params=model.state_dict()
for name, param in model.named_parameters():
    if name == 'sincnet_1.low_hz_':
        filt_b1 = param
    if name == 'sincnet_1.band_hz_':
        filt_band = param

out_channels = 80
kernel_size = 1001
# kernel_size = 201
sample_rate = 16000
min_low_hz = 1
min_band_hz = 1
low_hz = 1
high_hz = 8000 - (min_low_hz + min_band_hz)
low_hz_ = filt_b1
band_hz_ = filt_band
n_lin = torch.linspace(0, (kernel_size / 2) - 1,steps=int((kernel_size / 2)))  # computing only half of the window
window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / kernel_size)
n = (kernel_size - 1) / 2.0
n_ = 2 * math.pi * torch.arange(-n, 0).view(1, -1) / sample_rate  # Due to symmetry, I only need half of the time axes
low = min_low_hz + torch.abs(low_hz_)
high = low + min_band_hz + torch.abs(band_hz_)
band = (high - low)[:, 0]

f_times_t_low = torch.matmul(low, n_)
f_times_t_high = torch.matmul(high, n_)

band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (
        n_ / 2)) * window_  # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations.
band_pass_center = 2 * band.view(-1, 1)
band_pass_right = torch.flip(band_pass_left, dims=[1])

band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
band_pass = band_pass / (2 * band[:, None])
filters = (band_pass).view(out_channels, 1, kernel_size)
sincnet = F.conv1d(torch.tensor(y).reshape(1,1,220500), filters, stride=1,padding=500, dilation=1, bias=None, groups=1)
print(sincnet[0][70].detach().numpy())
y = sincnet[0][70].detach().numpy()
io.savemat('test', {'name': y})
librosa.display.waveplot(y,sr)
plt.title('Time Wave After Filter 70')
plt.show()
melspec = librosa.feature.melspectrogram(y,sr,n_fft=2048,hop_length=512,n_mels=128)
logmelspec = librosa.power_to_db(melspec)
plt.figure()
librosa.display.specshow(logmelspec,sr=sr,x_axis='time',y_axis='mel')
plt.title('Melspectrogram After Filter 70')
plt.show()

import wave
# import matplotlib.pyplot as plt
import numpy as np
import os
import struct

# wav文件读取
filepath = "../src_DIY_cnn/"  # 添加路径
filename = os.listdir(filepath)  # 得到文件夹下的所有文件名称
f = wave.open('F:/audio/1-60997-A-20.wav', 'rb')
params = f.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]
strData = f.readframes(nframes)  # 读取音频，字符串格式
waveData = np.fromstring(strData, dtype=np.int16)  # 将字符串转化为int
waveData = y
waveData = waveData * 1.0 / (max(abs(waveData)))  # wave幅值归一化
f.close()
# wav文件写入
outData = waveData  # 待写入wav的数据，这里仍然取waveData数据
print(outData)
outfile = filepath + 'filter70.wav'
outwave = wave.open(outfile, 'wb')  # 定义存储路径以及文件名
nchannels = 1
sampwidth = 2
fs = 44100
data_size = len(outData)
framerate = int(fs)
nframes = data_size
comptype = "NONE"
compname = "not compressed"
outwave.setparams((nchannels, sampwidth, framerate, nframes,
                   comptype, compname))

for v in outData:
    outwave.writeframes(struct.pack('h', int(v * 64000 / 2)))  # outData:16位，-32767~32767，注意不要溢出
outwave.close()
