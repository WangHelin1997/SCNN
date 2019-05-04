
from util import *
import os
import random
import numpy as np



def get_fold_wavelist(fold_wavelist):
    f = open(fold_wavelist, 'r')
    waveList = []
    for line in f.readlines():
        filePath = os.path.join('../ESC-50', line.split('\t')[0])
        waveList.append(filePath)
    return waveList


def get_pkl(fs):
    """
    :return: [{'key', 'data', 'label'}, {}, {}... {}]
    """

    wav_len = fs * 5

    for fold_num in range(5):

        print('get pkl on fold ', str(fold_num))
        trainWaveName = '../cross_folds/evaluate-setup-ESC50/fold' + str(fold_num) + '_train.txt'
        validWaveName = '../cross_folds/evaluate-setup-ESC50/fold' + str(fold_num) + '_valid.txt'
        testWaveName = '../cross_folds/evaluate-setup-ESC50/fold' + str(fold_num) + '_test.txt'
        trainWaveList = get_fold_wavelist(trainWaveName)
        validWaveList = get_fold_wavelist(validWaveName)
        testWaveList = get_fold_wavelist(testWaveName)

        # waveLists = [trainWaveList, validWaveList, testWaveList]
        waveLists = [trainWaveList, testWaveList]

        data = []
        item = {}

        for idx, wavelist in enumerate(waveLists):
            for f in wavelist:
               # print(f)
                cls_id = f.split('/')[1].split(' ')[0].split('\\')[1]
               # print(cls_id)
                cls_id = num_to_id_ESC50(cls_id)
                # cls_id = num_to_id_ESC10(cls_id)

                audio_data, _ = librosa.load(f, fs)

                # make each audio exactly 5s.
                audio_data = audio_data[: wav_len]
               # print (audio_data)
                audio_data = audio_data * 1.0 / np.max(abs(audio_data))
               # print (audio_data)
                if len(audio_data) < wav_len:
                    audio_data = np.r_[audio_data, np.zeros(wav_len - len(audio_data))]

                item['label'] = int(cls_id)
                item['key'] = f.split('/')[-1].split('.')[0]
                item['data'] = audio_data

                data.append(item)
                item = {}

            if idx == 0:
                random.shuffle(data)
                # if not os.path.exists('../data_wave_ESC50_' + str(fs)):
                #     os.makedir('../data_wave_ESC50_' + str(fs))

                save_data('../data_wave_ESC50_'+ str(fs) +'/fold'+str(fold_num)+'_train.cPickle', data)
            elif idx == 1:
                save_data('../data_wave_'+ str(fs) +'/fold' + str(fold_num) + '_valid.cPickle', data)
            # elif idx == 1:
            #
            #     save_data('../data_wave_ESC50_'+ str(fs) +'/fold' + str(fold_num) + '_test.cPickle', data)

            data=[]


def get_spec():
    """
    :return: [{'key', 'data', 'label'}, {}, {}... {}]
    """

    win_size = 66150
    stride = int(44100 * 0.2)

    for fold_num in range(5):
        trainPkl = '../data_wave_44100/fold' + str(fold_num) + '_train.cPickle'
        # validPkl = '../data_wave_44100/fold' + str(fold_num) + '_valid.cPickle'
        sampleSet = load_data(trainPkl)

        segs = []

        for item in sampleSet:
            print(item['label'], item['key'])
            record_data = item['data']

            for j in range(0, len(record_data) - win_size + 1, stride):

                seg = {}
                win_data = record_data[j: j+win_size]
                # Continue if cropped region is silent

                maxamp = np.max(np.abs(win_data))
                if maxamp < 0.005:
                    continue
                melspec = librosa.feature.melspectrogram(win_data, 44100, n_fft=2048, hop_length=150, n_mels=64)  # (40, 442)
                # logmel = librosa.logamplitude(melspec)[:,:441]  # (40, 441)
                logmel = librosa.amplitude_to_db(melspec)[:, :441]  # (40, 441)
                delta = librosa.feature.delta(logmel)

                feat = np.stack((logmel, delta))

                seg['label'] = item['label']
                seg['key'] = item['key']
                seg['data'] = feat

                segs.append(seg)

        save_data('../segments_logmel/fold' + str(fold_num) + '_train.cPickle', segs)

    for fold_num in range(5):
        validPkl = '../data_wave_44100/fold' + str(fold_num) + '_valid.cPickle'
        sampleSet = load_data(validPkl)

        segs = []

        for item in sampleSet:
            print(item['label'], item['key'])
            record_data = item['data']

            for j in range(0, len(record_data) - win_size + 1, stride):

                seg = {}
                win_data = record_data[j: j+win_size]
                # Continue if cropped region is silent

                maxamp = np.max(np.abs(win_data))
                if maxamp < 0.005:
                    continue
                melspec = librosa.feature.melspectrogram(win_data, 44100, n_fft=2048, hop_length=150, n_mels=64)  # (40, 442)
                # logmel = librosa.logamplitude(melspec)[:,:441]  # (40, 441)
                logmel = librosa.amplitude_to_db(melspec)[:, :441]  # (40, 441)
                delta = librosa.feature.delta(logmel)

                feat = np.stack((logmel, delta))

                seg['label'] = item['label']
                seg['key'] = item['key']
                seg['data'] = feat

                segs.append(seg)

        save_data('../segments_logmel/fold' + str(fold_num) + '_valid.cPickle', segs)
    #save_data('../segments_logmel/fold0_valid.cPickle', segs)


if __name__ == '__main__':


    # get_pkl(fs=44100)


    # get_spec()
    data = load_data('../segments_logmel/fold1_train.cPickle')
    print("data num: ", len(data))
    print(data[1])
    print(len(data[1]['data'][1]))
