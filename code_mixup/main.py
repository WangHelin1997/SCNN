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
from util import mixup_data, mixup_criterion
# Training settings
from torch.optim import lr_scheduler

# from src.network import WaveMsNet_fixed_logmel, WaveMsNet_lrf_fixed_logmel, WaveMsNet_mrf_fixed_logmel, \
#     WaveMsNet_srf_fixed_logmel, WaveMsNet
# from src.network import *


parser = argparse.ArgumentParser(description='pytorch model')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                            help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=5, metavar='N',
                            help='input batch size for testing (default: 5)')
parser.add_argument('--epochs', type=int,  metavar='N', default=400,
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
parser.add_argument('--model_save_interval', type=int, default=40, metavar='N',
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
parser.add_argument('--lr_list1', type=int, default=80,
                            help='list1 of lr.')
parser.add_argument('--lr_list2', type=int, default=140,
                            help='list2 of lr.')
parser.add_argument('--lr_list3', type=int, default=200,
                            help='list3 of lr.')
parser.add_argument('--alpha', default=1., type=float, help='interpolation strength (uniform=1., ERM=0.)')
parser.add_argument('--cuda_num', type=str, default="0",
                            help='cuda device num.')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_num
args.cuda = not args.no_cuda and torch.cuda.is_available()

#  torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    #  torch.cuda.set_device(2)

def train(model, optimizer, train_loader, epoch, phase, criterion):
#{{{
    model.train()
    start = time.time()
    total = 0
    running_loss = 0
    running_correct = 0
    if phase == 1:
        for idx, (data, label) in enumerate(train_loader):

            #  reshape to torch.LongTensor of size 64
            label = label.resize_(label.size()[0])

            if args.cuda:
                data, label = data.cuda(), label.cuda()
            data, label = Variable(data), Variable(label)
            inputs, label_a, label_b, lam = mixup_data(data, label, args.alpha)
            optimizer.zero_grad()
            inputs, label_a, label_b = Variable(inputs), Variable(label_a), Variable(label_b)
            # print data.size()
            output = model(inputs) # (batch, 50L)
            # print(label)
            # print output.size()
            # exit(0)
            loss_func = mixup_criterion(label_a, label_b, lam)
            # loss = F.cross_entropy(output, label)
            loss = loss_func(criterion, output)
            #  loss = F.nll_loss(output, label)

            loss.backward()

            optimizer.step()
            _, pred = torch.max(output.data, 1)  # get the index of the max log-probability
            total += label.size(0)
            # statistics
            running_loss += loss.item()
            running_correct += lam*float(torch.sum(pred == label_a.data.view_as(pred)))+ (1-lam)*float(torch.sum(pred == label_b.data.view_as(pred)))

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
            len(train_loader.dataset), epoch_loss, epoch_acc) + '\n')
        f.close()


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


def test(model, test_pkl, phase):
#{{{
    model.eval()

    start = time.time()

    test_loss = 0
    correct = 0
    y_pred = []
    y_true = []

    win_size = 66150
    stride = int(44100 * args.test_slices_interval)
    sampleSet = load_data(test_pkl)

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
            output = model(data)
            output = torch.sum(output, dim=0, keepdim=True)
            # print output

            test_loss += float(F.cross_entropy(output, label).item())  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(label.data.view_as(pred)).sum()

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
                melspec = librosa.feature.melspectrogram(feat_data, 44100, n_fft=2048, hop_length=150, n_mels=96)  # (40, 442)
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


def main_on_fold(foldNum, trainPkl, validPkl):
    model = WaveMsNet_fixed_logmel(phase=args.phase)
    if args.network == 'WaveMsNet':
        model = WaveMsNet()
    elif args.network == 'WaveMsNet_LogMel':
        model = WaveMsNet_Logmel()
    elif args.network == 'WaveMsNet_srf_fixed_logmel':
        model = WaveMsNet_srf_fixed_logmel()
    elif args.network == 'WaveMsNet_mrf_fixed_logmel':
        model = WaveMsNet_mrf_fixed_logmel()
    elif args.network == 'WaveMsNet_lrf_fixed_logmel':
        model = WaveMsNet_lrf_fixed_logmel()
    elif args.network == 'WaveMsNet_fixed_logmel':
        model = WaveMsNet_fixed_logmel(phase=args.phase)
        if args.phase == 2:
            model = torch.load('../model/' + args.network + '_phase1_fold' + str(foldNum) + '_best.pkl')

    if args.cuda:
        model.cuda()

    if args.phase == 1:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        for param in list(model.parameters())[:24]:
            param.requires_grad = False
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    #  optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[args.lr_list1, args.lr_list2, args.lr_list3], gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    #trainDataset = WaveformDataset(trainPkl, window_size=66150, train_slices=args.train_slices, transform=ToTensor(), add_logmel= True)
    #trainDataset = MFCCDataset(trainPkl, window_size=66150, train_slices=args.train_slices, transform=ToTensor(), add_logmel= True, fs= 44100)
    trainDataset = FusionDataset(trainPkl, window_size=66150, train_slices=args.train_slices, phase=args.phase)
    train_loader = DataLoader(trainDataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
    # for epoch in range(1, 2):
        exp_lr_scheduler.step()

        train(model, optimizer, train_loader, epoch, args.phase, criterion)

        #  test and save the best model
        if epoch % args.model_save_interval == 0:
            test_acc = test(model, validPkl, args.phase)
            if test_acc > best_acc:
                best_acc = test_acc
                # best_model_wts = model.state_dict()

                model_name = '../model/' + args.network + '_phase'+ str(args.phase)+ '_fold' + str(foldNum) + '_epoch' + str(epoch) + '.pkl'
                best_model_name = '../model/' + args.network + '_phase'+ str(args.phase)+ '_fold' + str(foldNum) + '_best.pkl'
                torch.save(model, model_name)

                file = args.output
                f = open(file, 'a')
                print('model has been saved as: ' + model_name)
                f.write('model has been saved as: ' + model_name+'\n')

                if args.phase == 1:
                    torch.save(model, best_model_name)
                    print('Till now, best model has been saved as: ' + best_model_name)
                    f.write('Till now, best model has been saved as: ' + best_model_name + '\n')
                f.close()


def main():
    print(args.network)
    print(args.phase)
    for fold_num in range(5):
        trainPkl = '../data_wave_44100/fold' + str(fold_num) + '_train.cPickle'
        validPkl = '../data_wave_44100/fold' + str(fold_num) + '_valid.cPickle'
        start = time.time()
        main_on_fold(fold_num, trainPkl, validPkl)

        file = args.output
        f = open(file, 'a')
        print('time on fold: %fs' % (time.time() - start))
        f.write('time on fold: %fs' % (time.time() - start)+'\n')
        f.close()

if __name__ == "__main__":
    main()

