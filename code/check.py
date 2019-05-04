""" Display waveform and logmel-spectrogram figures.
    usage:
    python check.py
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
# y,sr = librosa.load("F:/TUT-urban-acoustic-scenes-2018-development/audio/street_pedestrian-stockholm-157-4779-a.wav", sr=None)
y,sr = librosa.load("F:/audio/1-94036-A-22.wav", sr=None)
print(y)
plt.figure()
librosa.display.waveplot(y,sr)
plt.title('ESC50_1-94036-A-22.wav')
plt.show()
melspec = librosa.feature.melspectrogram(y,sr,n_fft=2048,hop_length=512,n_mels=128)
logmelspec = librosa.power_to_db(melspec)
plt.figure()
librosa.display.specshow(logmelspec,sr=sr,x_axis='time',y_axis='mel')
plt.title('ESC50_1-94036-A-22.wav')
plt.show()


