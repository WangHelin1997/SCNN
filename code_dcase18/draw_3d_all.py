from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from data_process import *
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import math

def get_normal_random_number(loc, scale):
	"""
	:param loc: 正态分布的均值
	:param scale: 正态分布的标准差
	:return:从正态分布中产生的随机数
	"""
	# 正态分布中的随机数生成
	number = np.random.normal(loc=loc, scale=scale)
	# 返回值
	return number

def to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)
def to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)
def frequency_response(filt_b1, filt_band):
    out_channels = 80
    kernel_size = 201
    # Forcing the filters to be odd (i.e, perfectly symmetrics)
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1
    sample_rate = 16000
    min_low_hz = 1
    min_band_hz = 20
    # initialize filterbanks such that they are equally spaced in Mel scale
    low_hz = 1
    high_hz = 8000 - (min_low_hz + min_band_hz)
    hz_high = 8000
    mel = np.linspace(to_mel(low_hz),
                      to_mel(high_hz),
                      out_channels + 1)
    hz = to_hz(mel)
    # filter lower frequency (out_channels, 1)
    low_hz_ = filt_b1

    # filter frequency band (out_channels, 1)
    band_hz_ = filt_band

    # Hamming window
    n_lin = torch.linspace(0, (kernel_size / 2) - 1,
                           steps=int((kernel_size / 2)))  # computing only half of the window
    window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / kernel_size);
    # (kernel_size, 1)
    n = (kernel_size - 1) / 2.0
    n_ = 2 * math.pi * torch.arange(-n, 0).view(1, -1) / sample_rate  # Due to symmetry, I only need half of the time axes
    low = min_low_hz + torch.abs(low_hz_)
    high = torch.clamp(low + min_band_hz + torch.abs(band_hz_), min_low_hz, hz_high)
    band = (high - low)[:, 0]

    f_times_t_low = torch.matmul(low, n_)
    f_times_t_high = torch.matmul(high, n_)

    band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (
            n_ / 2)) * window_  # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations.
    band_pass_center = 2 * band.view(-1, 1)
    band_pass_right = torch.flip(band_pass_left, dims=[1])

    band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
    band_pass = band_pass / (2 * band[:, None])
    filters = (band_pass).view(
        out_channels, 1, kernel_size)

    return filters

def draw_frequency_response(data , draw_choice):
    a = data
    f = 16000
    n = len(a)
    k = np.arange(n)
    T = n / f
    f = k / T
    t = np.arange(0, n, 1)
    dft=  abs(np.fft.fft(a))
    dft_a = dft / sum(dft)
    dft_a = dft_a[range(int(n / 2))]
    dft_a = np.append(dft_a, 0)
    dd = dft_a
    dd[0]=0
    dd[1:int(n/2)+1]=dft_a[0:int(n/2)]
    f1 = f[range(int(n / 2))]
    f1=np.append(f1,0)
    ff = f1
    ff[0] = 0
    ff[1:int(n / 2)+1] = f1[0:int(n / 2)]
    if draw_choice == 'true':
        # plt.ion()
        plt.title('Filter 75 in the time domain')
        plt.plot(t, a, 'b')
        plt.xlim(-1, n)
        # plt.ylabel('Frequency')
        # plt.xlabel('sample')
        plt.xticks([])
        plt.yticks([])
        # plt.pause(0.5)
        # plt.close()
        plt.show()
        # plt.ion()
        plt.title('Filter 75 frequency response')
        plt.plot(f1, dft_a, 'r')
        plt.xlim(0, )
        plt.ylim(0, )
        plt.xlabel('Frequency[Hz] (6476.72~ 6705.23)')
        plt.yticks([])
        # plt.xlabel('')
        # plt.pause(0.5)
        # plt.close()
        plt.show()
        # data = np.fft.ifft(np.log(dft))
        # b,a = signal.butter(8, 0.08, 'lowpass')
        # filt_data = signal.filtfilt(b,a, data)
        # plt.plot(f, filt_data)
        # plt.show()
        return dd, ff
    else:
        return dd, ff

def draw_cumulative_frequency_response(dft_a, f1):
    for i in range(len(dft_a)):
        dft_a[i] = dft_a[i] + get_normal_random_number(0, 0.5)/200
    dft_a = dft_a/sum(dft_a)
    plt.title('Cumulative frequency response of SincConv on DCASE2018')
    plt.plot(f1, dft_a, 'r')
    plt.xlim(0, )
    plt.ylim(0, )
    plt.ylabel('Normalized Filter Sum')
    plt.xlabel('Frequency[Hz]')
    plt.show()


X = np.zeros((10,101))
Y = np.zeros((10,101))
Z = np.zeros((10,101))
for x in range(10):
    model = torch.load('model/WaveMsNet_fixed_logmel_epoch'+str(x*100+100)+'.pkl', map_location='cpu')
    # model = torch.load('WaveMsNet_fixed_logmel_phase1_best.pkl', map_location='cpu')
    params=model.state_dict()
    for name, param in model.named_parameters():
        # print(name)
        if name == 'sincnet_1.low_hz_':
            filt_b1 = param
        if name == 'sincnet_1.band_hz_':
            filt_band = param

    a = frequency_response(filt_b1, filt_band)
    for i in range(80):
        data = a[i][0].detach().numpy()
        dft_a , f1 = draw_frequency_response(data, draw_choice='rue')
        dlen = len(dft_a)
        if i==0:
            dft_all = dft_a
        else:
            dft_all = dft_all + dft_a
    for j in range(len(dft_a)):
        X[x][j] = f1[j]
        Z[x][j] = dft_all[j]
        Y[x][j] = x*100



# draw_cumulative_frequency_response(dft_all, f1)
fig, ax1= plt.subplots(1, 1, subplot_kw={'projection': '3d'})

# Get the test data
# X, Y, Z = axes3d.get_test_data(0.05)
# print(type(X))
# print(Y[0])
# print(Z[0])
# Give the first plot only wireframes of the type y = c
ax1.plot_wireframe(X, Y, Z, rstride=1, cstride=0)
ax1.set_title("Cumulative Frequency Response of epochs on DCASE2018")
ax1.set_xlabel('Frequency[Hz]')
ax1.set_ylabel('epoch')
ax1.set_zlabel('Cumulative Frequency Response')
# ax1.set_xlim(-50,50)
# cset = ax1.contour(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
# cset = ax1.contour(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
# cset = ax1.contour(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)
# Give the second plot only wireframes of the type x = c
# ax2.plot_wireframe(X, Y, Z, rstride=0, cstride=8)
# ax2.set_title("Row (y) stride set to 0")

plt.tight_layout()
plt.show()