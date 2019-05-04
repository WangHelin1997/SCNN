import os
import pickle

data = pickle.load(open("../data_wave_ESC50_44100/fold0_train.cPickle","rb"))
print (data)