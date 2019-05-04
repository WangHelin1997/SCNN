""" usage:
    python main.py
    python main.py --network=WaveMsNet --mode=test --model='../model/dnn_mix.pkl'

"""
import argparse
import time
from network import *
import torch.utils.data as Data
import torchvision
import torch.nn.functional as F
from data_process import *
import os

# Training settings
from torch.optim import lr_scheduler
from net_ESC50 import WaveMsNet as WNN
from tensorboardX import SummaryWriter
writer = SummaryWriter('./res/')

parser = argparse.ArgumentParser(description='pytorch model')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                            help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=5, metavar='N',
                            help='input batch size for testing (default: 5)')
parser.add_argument('--epochs', type=int,  metavar='N', default=700,
                            help='number of epochs to train')
parser.add_argument('--lr', type=float,  metavar='LR', default=0.01,
                            help='initial learning rate')
parser.add_argument('--momentum', type=float,  metavar='M', default=0.9,
                            help='SGD momentum')
parser.add_argument('--weight_decay', type=float, metavar='M', default=0.0001,
                            help='weight decay')
parser.add_argument('--no-cuda', action='store_true', default=False,
                            help='disables CUDA training')
parser.add_argument('--gpu', type=list, default=[0,1,2,3],
                            help='gpu device number')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                            help='how many batches to wait before logging training status')
parser.add_argument('--model_save_interval', type=int, default=4, metavar='N',
                            help='how many epochs to wait before saving the model.')
parser.add_argument('--network', type=str, default = 'WaveMsNet_fixed_logmel',
                            help='WaveMsNet or WaveMsNet_Logmel')
parser.add_argument('--mode', type=str, default='train',
                            help='train or test')
parser.add_argument('--model', type=str, default='../model/WaveMsNet_fold0_v2_epoch400.pkl',
                            help='trained model path')
parser.add_argument('--train_slices', type=int, default=1,
                            help='slices number of one record divide into.')
parser.add_argument('--test_slices_interval', type=int, default=0.2,
                            help='slices number of one record divide into.')
parser.add_argument('--output', type=str, default = '../output_second_phase.txt',
                            help='output file.')
parser.add_argument('--fs', type=int)
parser.add_argument('--phase', type=int, default=1,
                            help='phase.')
parser.add_argument('--lr_list1', type=int, default=150,
                            help='list1 of lr.')
parser.add_argument('--lr_list2', type=int, default=360,
                            help='list2 of lr.')
parser.add_argument('--lr_list3', type=int, default=420,
                            help='list3 of lr.')
parser.add_argument('--cuda_num', type=str, default="1",
                            help='cuda device num.')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_num
args.cuda = not args.no_cuda and torch.cuda.is_available()

#  torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    #  torch.cuda.set_device(2)



#[windowing]
fs=16000
cw_len=1500
cw_shift=10


def str_to_bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise ValueError

def train(model_WNN, optimizer_WNN, train_loader, epoch, phase, fold_num):
#{{{
    model_WNN.train()
    start = time.time()

    running_loss = 0
    running_correct = 0
    if phase == 1:
        for idx, (data, label) in enumerate(train_loader):

            #  reshape to torch.LongTensor of size 64
            label = label.resize_(label.size()[0])

            if args.cuda:
                data, label = data.cuda(), label.cuda()
            data, label = Variable(data), Variable(label)

            optimizer_WNN.zero_grad()
            # print data.size()
            output = model_WNN(data) # (batch, 50L)
            # print(label)
            # print output.size()
            # exit(0)
            loss = F.cross_entropy(output, label)
            #  loss = F.nll_loss(output, label)

            loss.backward()

            optimizer_WNN.step()
            _, pred = torch.max(output.data, 1)  # get the index of the max log-probability

            # statistics
            running_loss += loss.item()
            running_correct += torch.sum(pred == label.data.view_as(pred))

        epoch_loss = running_loss / float(len(train_loader))
        epoch_acc = 100. * float(running_correct) / float(len(train_loader.dataset))

        elapse = time.time() - start

        f = open(args.output, 'a')
        print('Epoch:{} ({:.1f}s) lr:{:.4g}  '
              'samples:{}  Loss:{:.3f}  TrainAcc:{:.2f}%'.format(
            epoch, elapse, optimizer_WNN.param_groups[0]['lr'],
            len(train_loader.dataset), epoch_loss, epoch_acc))
        f.write('Epoch:{} ({:.1f}s) lr:{:.4g}  '
                'samples:{}  Loss:{:.3f}  TrainAcc:{:.2f}%'.format(
            epoch, elapse, optimizer_WNN.param_groups[0]['lr'],
            len(train_loader.dataset), epoch_loss, epoch_acc) + '\n')
        f.close()
        writer.add_scalar('scalar/Train_Loss'+str(fold_num), epoch_loss , epoch)
        writer.add_scalar('scalar/Train_Acc'+str(fold_num), epoch_acc, epoch)
        return epoch_acc


    else:
        for idx, (data, feat, label) in enumerate(train_loader):

            #  reshape to torch.LongTensor of size 64
            label = label.resize_(label.size()[0])

            if args.cuda:
                data, feat, label = data.cuda(), feat.cuda(), label.cuda()
            data, feat, label = Variable(data), Variable(feat), Variable(label)

            optimizer.zero_grad()

            # print data.size()
            output = model.forward(x=data, feats=feat) # (batch, 50L)
            # print(label)
            # print output.size()
            # exit(0)
            loss = F.cross_entropy(output, label)
            #  loss = F.nll_loss(output, label)

            loss.backward()

            optimizer.step()
            _, pred = torch.max(output.data, 1)  # get the index of the max log-probability

            # statistics
            running_loss += float(loss.item())
            running_correct += torch.sum(pred == label.data.view_as(pred))

        epoch_loss = running_loss / float(len(train_loader))
        epoch_acc = 100. * float(running_correct) / float(len(train_loader.dataset))

        elapse = time.time() - start

        f = open(args.output, 'a')
        print('Epoch:{} ({:.1f}s) lr:{:.4g}  '
              'samples:{}  Loss:{:.3f}  TrainAcc:{:.2f}%'.format(
            epoch, elapse, optimizer.param_groups[0]['lr'],
            len(train_loader.dataset), epoch_loss, epoch_acc))
        f.write('Epoch:{} ({:.1f}s) lr:{:.4g}  '
              'samples:{}  Loss:{:.3f}  TrainAcc:{:.2f}%'.format(
            epoch, elapse, optimizer.param_groups[0]['lr'],
            len(train_loader.dataset), epoch_loss, epoch_acc)+'\n')
        f.close()


def test( model_WNN, test_pkl, phase, epoch, fold_num):
#{{{
    model_WNN.eval()

    start = time.time()

    test_loss = 0
    correct = 0
    y_pred = []
    y_true = []

    win_size = 24000
    stride = int(16000 * args.test_slices_interval)
    sampleSet = load_data(test_pkl)
    label_all = []
    pred_all = []
    if phase == 1:
        for item in sampleSet:
            label = item['label']
            record_data = item['data']
            wins_data = []
            for j in range(0, len(record_data) - win_size + 1, stride):

                win_data = record_data[j: j + win_size]
                # Continue if cropped region is silent

                maxamp = np.max(np.abs(win_data))
                if maxamp < 0.005:
                    continue
                wins_data.append(win_data)

            if len(wins_data) == 0:
                print(item['key'])

            wins_data = np.array(wins_data)

            wins_data = wins_data[:, np.newaxis, :]
            # print wins_data.shape

            data = torch.from_numpy(wins_data).type(torch.FloatTensor)  # (N, 1L, 24002L)
            label = torch.LongTensor([label])

            if args.cuda:
                data, label = data.cuda(), label.cuda()
            data, label = Variable(data), Variable(label)

            # print data.size()
            output = model_WNN(data)
            output = torch.sum(output, dim=0, keepdim=True)
            # print output
            # print(label.cpu().detach().numpy())
            label_all.append(label[0].cpu().detach().numpy().tolist())
            # print(output.shape)
            test_loss += float(F.cross_entropy(output, label).item())  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            # print(pred.cpu().detach().numpy())#[0]
            pred_all.append(pred[0][0].cpu().detach().numpy().tolist())
            correct += pred.eq(label.data.view_as(pred)).sum()

        label_all = np.array(label_all)
        pred_all = np.array(pred_all)
        np.save('label_all.npy',label_all)
        np.save('pred_all.npy', pred_all)
        test_loss /= len(sampleSet)
        test_acc = 100. * float(correct) / float(len(sampleSet))

        elapse = time.time() - start

        file = args.output
        f = open(file, 'a')
        print('\nTest set: Average loss: {:.3f} ({:.1f}s), TestACC: {}/{} {:.2f}%\n'.format(
            test_loss, elapse, correct, len(sampleSet), test_acc))
        f.write('\nTest set: Average loss: {:.3f} ({:.1f}s), TestACC: {}/{} {:.2f}%\n'.format(
            test_loss, elapse, correct, len(sampleSet), test_acc) + '\n')
        f.close()
        writer.add_scalar('scalar/Test_Loss'+str(fold_num), test_loss, epoch)
        writer.add_scalar('scalar/Test_Acc'+str(fold_num), test_acc, epoch)
        for name, param in model_WNN.named_parameters():
            # print(name)
            if name == 'sincnet_1.low_hz_':
                filt_b1 = param.cpu().detach().numpy()
                # print(filt_b1)
            if name == 'sincnet_1.band_hz_':
                filt_band = param.cpu().detach().numpy()
                # print(filt_band)
        for i in range(len(filt_b1)):
            if i == 0:
                if filt_band[i]>0:
                    filt = list([filt_b1[i], filt_b1[i] + filt_band[i]+20])
                else:
                    filt = list([filt_b1[i]])
                # filt = list(np.arange(filt_b1[i], filt_b1[i] + filt_band[i] + 800, 800))
                # print(filt)
            else:
                if filt_band[i]>0:
                    filt_list = list([filt_b1[i], filt_b1[i] + filt_band[i]+20])
                else:
                    filt_list = list([filt_b1[i]])
                # filt_list = list(np.arange(filt_b1[i], filt_b1[i] + filt_band[i] + 800, 800))
                # print(filt_list)
                filt.extend(filt_list)
        filt = np.array(filt)

        writer.add_histogram('filt', torch.from_numpy(filt), epoch)
    else:
        for item in sampleSet:
            label = item['label']
            record_data = item['data']
            wins_data = []
            feats_data = []
            for j in range(0, len(record_data) - win_size + 1, stride):

                win_data = record_data[j: j+win_size]
                feat_data = record_data[j: j+win_size]
                # Continue if cropped region is silent
                #win_data = torch.FloatTensor(win_data)
                #feat_data = torch.FloatTensor(feat_data)
                #win_data , feat_data = win_data.cuda(), feat_data.cuda()
                #win_data, feat_data = Variable(win_data), Variable(feat_data)

                maxamp = np.max(np.abs(win_data))
                #maxamp = abs(win_data).max()
                if maxamp < 0.005:
                    continue
                #MFCC
                melspec = librosa.feature.melspectrogram(feat_data, 16000, n_fft=2048, hop_length=150, n_mels=96)  # (40, 442)
                logmel = librosa.amplitude_to_db(melspec)[:, :441]  # (40, 441)
                feats_data.append(logmel)
                #WaveMsNet
                wins_data.append(win_data)


            if len(wins_data) == 0:
                print(item['key'])

            wins_data = np.array(wins_data)
            wins_data = wins_data[:, np.newaxis, :]
            # print wins_data.shape
            if len(feats_data) == 0:
                print(item['key'])
            feats_data = np.array(feats_data)

            feats_data = feats_data[:, np.newaxis, :]
            # print feats_data.shape

            data = torch.from_numpy(wins_data).type(torch.FloatTensor) # (N, 1L, 24002L)
            feats = torch.from_numpy(feats_data).type(torch.FloatTensor) # (N, 1L, 24002L)
            label = torch.LongTensor([label])

            if args.cuda:
                data, feats, label = data.cuda(), feats.cuda(), label.cuda()
            data, feats, label = Variable(data), Variable(feats), Variable(label)

            # print data.size()
            output = model(x=data, feats=feats)
            output = torch.sum(output, dim=0, keepdim=True)
            # print output

            test_loss += float(F.cross_entropy(output, label).item()) # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(label.data.view_as(pred)).sum()

        test_loss /= len(sampleSet)
        test_acc = 100. * float(correct) / float(len(sampleSet))

        elapse = time.time() - start

        file = args.output
        f = open(file,'a')
        print('\nTest set: Average loss: {:.3f} ({:.1f}s), TestACC: {}/{} {:.2f}%\n'.format(
            test_loss, elapse, correct, len(sampleSet), test_acc))
        f.write('\nTest set: Average loss: {:.3f} ({:.1f}s), TestACC: {}/{} {:.2f}%\n'.format(
            test_loss, elapse, correct, len(sampleSet), test_acc)+'\n')
        f.close()

    return test_acc

lr = args.lr
def adjust_learning_rate(optimizer, epoch):
    adj_lr = lr * (0.98 ** (epoch // 4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = adj_lr

def main_on_fold(foldNum, trainPkl, validPkl):
    WNN_net = WNN()
    WNN_net.cuda()
    model_WNN=WNN_net
    x = torch.autograd.Variable(torch.rand(32,1,24000))
    writer.add_graph(model_WNN, x.cuda(), verbose=True)
    if args.phase == 1:
        optimizer_WNN = optim.SGD(model_WNN.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        for param in list(model.parameters())[:24]:
            param.requires_grad = False
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    exp_lr_scheduler_WNN = lr_scheduler.MultiStepLR(optimizer_WNN, milestones=[args.lr_list1, args.lr_list2, args.lr_list3], gamma=0.05)
    # exp_lr_scheduler_WNN = lr_scheduler.ExponentialLR(optimizer_WNN, gamma=0.99)
    trainDataset = FusionDataset(trainPkl, window_size=24000, train_slices=args.train_slices, phase=args.phase)
    train_loader = DataLoader(trainDataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    best_acc = 0.0
    test_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        # exp_lr_scheduler_WNN.step()
        adjust_learning_rate(optimizer_WNN, epoch)
        auc = train(model_WNN, optimizer_WNN, train_loader, epoch, args.phase, foldNum)
        #  test and save the best model
        # if epoch % args.model_save_interval == 0:
        if auc > 95.5 :
            if epoch % args.model_save_interval == 0:
                test_acc = test(model_WNN, validPkl, args.phase, epoch, foldNum)
                model_name = 'model/' + args.network + '_epoch' + str(epoch) + '.pkl'
                torch.save(model_WNN, model_name)
            # if test_acc > best_acc:
            #     best_acc = test_acc
            #     # best_model_wts = model.state_dict()
            #
            #     model_name = '../model/' + args.network + '_phase'+ str(args.phase)+ '_fold' + str(foldNum) + '_epoch' + str(epoch) + '.pkl'
            #     best_model_name = '../model/' + args.network + '_phase'+ str(args.phase)+ '_fold' + str(foldNum) + '_best.pkl'
            #     torch.save(model_WNN, best_model_name)



def main():
    print(args.network)
    print(args.phase)
    for fold_num in range(1):
        trainPkl = '../data_wave_44100/fold' + str(fold_num) + '_train.cPickle'
        validPkl = '../data_wave_44100/fold' + str(fold_num) + '_valid.cPickle'
        start = time.time()
        main_on_fold(fold_num, trainPkl, validPkl)

        file = args.output
        f = open(file, 'a')
        print('time on fold: %fs' % (time.time() - start))
        f.write('time on fold: %fs' % (time.time() - start)+'\n')
        f.close()
    writer.close()

if __name__ == "__main__":
    main()

