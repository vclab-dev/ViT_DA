from sklearn import svm
import argparse
import os
import os.path as osp
import torchvision
import numpy as np
import torch
from torchvision import transforms
from tqdm.std import tqdm
from wandb.sdk.lib import disabled
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_dom_dis
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    print(s)
    return s

def load_model(args):
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net, se=False, nl=False).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()
    elif args.net == 'vit':
        netF = network.ViT().cuda()
    elif args.net[0:4] == 'deit':
        if args.net == 'deit_s':
            netF = torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224', pretrained=True).cuda()
        elif args.net == 'deit_b':
            netF = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True).cuda()
        netF.in_features = 1000
    
    # summary(netF, (3, 224, 224))
        
    netB = network.feat_bootleneck(type="bn", feature_dim=netF.in_features,
                                   bottleneck_dim=256).cuda()
   

    modelpath = f'{args.weights_path}/{args.dset}/{names[args.s][0].upper()}{names[args.t][0].upper()}/target_F_par_0.2.pt'
    netF.load_state_dict(torch.load(modelpath))
    # print('Model Loaded from', modelpath)

    netF.eval()
    netB.eval()
    # netC.eval()
    return netF,netB

def data_load(args):

    def image_train(resize_size=256, crop_size=224, alexnet=False):
        if not alexnet:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        return  transforms.Compose([
                transforms.Resize((resize_size, resize_size)),
                transforms.RandomCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])

    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()
    txt_src = open(args.t_dset_path).readlines()

    dsets["source"] = ImageList_dom_dis(txt_src, transform=image_train(), strong_aug=0)
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, shuffle=True,drop_last=False)

    dsets["target"] = ImageList_dom_dis(txt_src, transform=image_train(), strong_aug=1)
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True,drop_last=False)

    return dsets, dset_loaders

def feature_extractor(loader, netF,netB):
    start_test = True
    features = []
    label =  []

    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            
            if start_test:
                all_fea = feas.float().cpu()
                all_label = labels.int()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    return all_fea, all_label


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--s', type=int, default=0, help="source")
    # parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--dset', type=str, default='office', help="dset")
    parser.add_argument('--net', type=str, default='deit_s', help="vgg16, resnet50, resnet101")
    parser.add_argument('--weights_path', type=str, default='weights/STDA_wt_fbnm_rlccsoft/STDA', help="vgg16, resnet50, resnet101")
    parser.add_argument('--txt_path', type=str, default='data/', help="vgg16, resnet50, resnet101")
    parser.add_argument('--gpu_id', type=str, nargs='?', default='1', help="device id to run")
    parser.add_argument('--batch_size', type=int, default='32', help="batch size")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
    
    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65
    
    for i in range(len(names)):
        args.t = i
        if i == args.s:
            continue

        args.s_dset_path = args.txt_path + args.dset + '/' + names[args.s] + '.txt'
        args.t_dset_path = args.txt_path + args.dset + '/' + names[args.t] + '.txt'

        dsets, dset_loaders = data_load(args)
        netF,netB = load_model(args)

        concatenated_dsets = torch.utils.data.ConcatDataset([dsets['source'],dsets['target']])

        train_size = int(0.8 * len(concatenated_dsets))
        test_size = len(concatenated_dsets) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(concatenated_dsets, [train_size, test_size])
        trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,drop_last=False)
        testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,drop_last=False)


        all_feas_train, all_label_train = feature_extractor(trainloader, netF,netB)
        all_feas_test, all_label_test = feature_extractor(testloader, netF,netB)

        clf = svm.SVC()
        clf.fit(all_feas_train, all_label_train)
        pred_lables = clf.predict(all_feas_test)

        acc = accuracy_score(all_label_test, pred_lables)
        print(f'\nSVM Acc for {names[args.s]} 2 {names[args.t]} with {args.net}:', acc*100)
