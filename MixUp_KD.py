#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.
from __future__ import print_function

import argparse
import csv
import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import network
from mixup_utils import progress_bar
from data_list import ImageList_idx, ImageList, ImageList_MixUp
from torch.utils.data import DataLoader
import ml_collections
import wandb

def init_src_model_load(args):
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net, se=args.se, nl=args.nl).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()
    elif args.net == 'vit':
        netF = network.ViT().cuda()
    elif args.net == 'deit_s':
        netF = torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224', pretrained=True).cuda()
        netF.in_features = 1000

    netB = network.feat_bootleneck(type='bn', feature_dim=netF.in_features,bottleneck_dim=256).cuda()
    netC = network.feat_classifier(type='wn', class_num=65, bottleneck_dim=256).cuda()

    return netF, netB, netC


def image_train(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def data_load(batch_size=64,txt_path='data/office-home'):
    ## prepare data

    def image_train(resize_size=256, crop_size=224, alexnet=False):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        return transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            normalize
        ])

    dsets = {}
    dset_loaders = {}
    train_bs = batch_size
    
    
    txt_files = {'clipart': f'{txt_path}/Clipart.txt',
                'art': f'{txt_path}/Art.txt', 
                'product':  f'{txt_path}/Product.txt', 
                'realworld': f'{txt_path}/RealWorld.txt'}

    for domain, paths in txt_files.items(): 
        # txt_tar = open(paths).readlines()
        txt_tar = open('Art.txt').readlines()#!@
        dsets[domain] = ImageList_MixUp(txt_tar, transform=image_train()) #!@
        dset_loaders[domain] = DataLoader(dsets[domain], batch_size=batch_size, shuffle=True,drop_last=False)
        break #!@
    return dset_loaders
    
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--net', default="deit_s", type=str,
                    help='model type (default: ResNet18)')
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--suffix', default='0', type=str, help='wandb name of run')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batch-size', default=32, type=int, help='batch size')
parser.add_argument('--epoch', default=200, type=int,
                    help='total epochs to run')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='use standard augmentation (default: True)')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--alpha', default=1., type=float,
                    help='mixup interpolation coefficient (default: 1)')
parser.add_argument('--wandb', type=int, default=0)

args = parser.parse_args()

use_cuda = torch.cuda.is_available()

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if args.seed != 0:
    torch.manual_seed(args.seed)

# Data
print('==> Preparing data..')
if args.augment:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
else:
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# trainset = datasets.CIFAR10(root='~/data', train=True, download=True,
#                             transform=transform_train)
# trainloader = torch.utils.data.DataLoader(trainset,
#                                           batch_size=args.batch_size,
#                                           shuffle=True, num_workers=8)
all_loader = data_load(batch_size=args.batch_size)
trainloader = all_loader['clipart']
testloader =  all_loader['clipart']

# testset = datasets.CIFAR10(root='~/data', train=False, download=False,
#                            transform=transform_test)
# testloader = torch.utils.data.DataLoader(testset, batch_size=100,
#                                          shuffle=False, num_workers=8)


# Model
# if args.resume:
#     # Load checkpoint.
#     print('==> Resuming from checkpoint..')
#     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
#     checkpoint = torch.load('./checkpoint/ckpt.t7' + args.name + '_'
#                             + str(args.seed))
#     net = checkpoint['net']
#     best_acc = checkpoint['acc']
#     start_epoch = checkpoint['epoch'] + 1
#     rng_state = checkpoint['rng_state']
#     torch.set_rng_state(rng_state)

if True:# else:
    print('==> Building model..')
    netF, netB, netC = init_src_model_load(args)

if not os.path.isdir('results'):
    os.mkdir('results')


logname = ('results/log' + '.csv')

if use_cuda:
    netF.cuda()
    netB.cuda()
    netC.cuda()
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    print('Using CUDA..')

criterion = nn.CrossEntropyLoss()
param_group = []
for k, v in netF.named_parameters():
    param_group += [{'params': v, 'lr': args.lr*0.1}]
for k, v in netB.named_parameters():
    param_group += [{'params': v, 'lr': args.lr}]
for k, v in netC.named_parameters():
    param_group += [{'params': v, 'lr': args.lr}]   

optimizer = optim.SGD(param_group, lr=args.lr, momentum=0.9,
                      weight_decay=args.decay)

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    netF.train()
    netB.train()
    netC.train()
    train_loss = 0
    reg_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, pseudo_lbl, targets, domain) in enumerate(trainloader):
        # print(inputs.shape, targets, pseudo_lbl, domain)
        if use_cuda:
            inputs, targets, pseudo_lbl, domain = inputs.cuda(), targets.cuda(),  pseudo_lbl.cuda(), domain.cuda()

        inputs, targets_a, targets_b, lam = mixup_data(inputs, pseudo_lbl,
                                                       args.alpha, use_cuda)
        inputs, targets_a, targets_b = map(Variable, (inputs,
                                                      targets_a, targets_b))
        outputs = netC(netB(netF(inputs)))
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += pseudo_lbl.size(0)
        correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar(batch_idx, len(trainloader),
                     'Loss: %.3f | Reg: %.5f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), reg_loss/(batch_idx+1),
                        100.*correct/total, correct, total))
        wandb.log({'train_loss': train_loss/(batch_idx+1),
                'reg_loss': reg_loss/(batch_idx+1),
                'train_acc': 100.*correct/total, 
            })
    return (train_loss/batch_idx, reg_loss/batch_idx, 100.*correct/total)


def test(epoch):
    global best_acc
    netF.eval()
    netB.eval()
    netC.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, pseudo_lbl, targets, domain) in enumerate(testloader):
        if use_cuda:
            inputs, pseudo_lbl, targets, domain = inputs.cuda(), pseudo_lbl.cuda(), targets.cuda(), domain.cuda()
        inputs, pseudo_lbl = Variable(inputs, volatile=True), Variable(pseudo_lbl)
        outputs = netC(netB(netF(inputs)))
        loss = criterion(outputs, pseudo_lbl)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += pseudo_lbl.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader),
                     'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss/(batch_idx+1), 100.*correct/total,
                        correct, total))
    acc = 100.*correct/total
    if epoch == start_epoch + args.epoch - 1 or acc > best_acc:
        checkpoint(acc, epoch)
    if acc > best_acc:
        best_acc = acc
    return (test_loss/batch_idx, 100.*correct/total)


def checkpoint(acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'netF': netF,
        'netB': netB,
        'netC': netC,
        'acc': acc,
        'epoch': epoch,
        # 'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt.t7' + args.name + '_'
               + str(args.seed))


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = args.lr
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if not os.path.exists(logname):
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['epoch', 'train loss', 'reg loss', 'train acc',
                            'test loss', 'test acc'])

mode = 'online' if args.wandb else 'disabled'
wandb.init(project='MixUp KD', entity='vclab', name=f'A20'+args.suffix, mode=mode)

for epoch in range(start_epoch, args.epoch):
    train_loss, reg_loss, train_acc = train(epoch)
    if epoch % 10 == 0:
        test_loss, test_acc = test(epoch)
        wandb.log({ 'test_loss': test_loss,  
                    'test_acc': test_acc,
                    })

    # adjust_learning_rate(optimizer, epoch)
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch, train_loss, reg_loss, train_acc, test_loss,
                            test_acc])