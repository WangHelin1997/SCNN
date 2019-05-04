

# import cPickle # for python2
import pickle  # for python3
import librosa
import torch
from torch.autograd import Variable


def save_data(filename, data):
    """Save variable into a pickle file

    Parameters
    ----------
    filename: str
        Path to file

    data: list or dict
        Data to be saved.

    Returns
    -------
    nothing

    """
    pickle.dump(data, open(filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    # pickle.dump(data, open(filename, 'w'))


def load_data(filename):
    """Load data from pickle file

    Parameters
    ----------
    filename: str
        Path to file

    Returns
    -------
    data: list or dict
        Loaded file.

    """

    return pickle.load(open(filename, "rb"), encoding='latin1')


def to_np(x):
    return x.data.cpu().numpy()


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def dic_ESC10():
    return {0: 'Fire crackling', 1: 'Dog bark', 2: 'Rain', 3: 'Sea waves', 4: 'Baby cry', 5: 'Clock tick',
            6: 'Person sneeze', 7: 'Helicopter', 8: 'Chainsaw', 9: 'Rooster'}


def dic_ESC50():
    return {0: 'hand_saw', 1: 'dog', 2: 'rooster', 3: 'pig', 4: 'cow', 5: 'frog', 6: 'cat', 7: 'hen',
            8: 'insects', 9: 'sheep', 10: 'crow', 11: 'rain', 12: 'sea_waves',
            13: 'crackling_fire', 14: 'crickets', 15: 'chirping_birds', 16: 'water_drops',
            17: 'wind', 18: 'pouring water', 19: 'toilet flush', 20: 'thunderstorm',
            21: 'crying_baby', 22: 'sneezing', 23: 'clapping', 24: 'breathing', 25: 'coughing',
            26: 'footsteps', 27: 'laughing', 28: 'brushing_teeth', 29: 'snoring',
            30: 'drinking_sipping', 31: 'door_wood_knock', 32: 'mouse_click', 33: 'keyboard_typing',
            34: 'door_wood_creaks', 35: 'can_opening', 36: 'washing_machine',
            37: 'vacuum_cleaner', 38: 'clock_alarm', 39: 'clock_tick', 40: 'glass_breaking',
            41: 'helicopter', 42: 'chainsaw', 43: 'siren', 44: 'car_horn', 45: 'engine',
            46: 'train', 47: 'church_bells', 48: 'airplane', 49: 'fireworks'}


def num_to_id_ESC50(num):
    dic = {'049':0, '000':1, '001':2, '002':3, '003':4, '004':5, '005':6, '006':7, '007':8,
           '008':9, '009':10, '010':11, '011':12, '012':13, '013':14, '014':15, '015':16,
           '016':17, '017':18, '018':19, '019':20, '020':21, '021':22, '022':23, '023':24,
           '024':25, '025':26, '026':27, '027':28, '028':29, '029':30, '030':31, '031':32,
           '032':33, '033':34, '034':35, '035':36, '036':37, '037':38, '038':39, '039':40,
           '040':41, '041':42, '042':43, '043':44, '044':45, '045':46, '046':47, '047':48,
           '048':49}
    return dic[num]


def num_to_id_ESC10(num):
    dic = {'010':0, '001':1, '002':2, '003':3, '004':4, '005':5, '006':6, '007':7, '008':8, '009':9}
    return dic[num]


def id_to_lb(id, dataSet='ESC-50'):
    if dataSet == 'ESC-10':
        dic = dic_ESC10()
    elif dataSet == 'ESC-50':
        dic = dic_ESC50()
    else:
        raise ValueError
    return dic[id]


def lb_to_id(lb, dataSet='ESC-50'):
    if dataSet == 'ESC-10':
        dic = dic_ESC10()
    elif dataSet == 'ESC-50':
        dic = dic_ESC50()
    else:
        raise ValueError
    re_dic = {v: k for k, v in dic.items()}
    return re_dic[lb]
