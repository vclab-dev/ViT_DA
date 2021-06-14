import argparse
import os, sys
import os.path as osp
from numpy.core.fromnumeric import shape
from torch.nn.modules.activation import Softmax
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from loss import KnowledgeDistillationLoss
import wandb
from tqdm import tqdm


def create_teachers_student(s2t, source): 

    '''
        Create ensemble of teachers and one student model 
        teachers : Dict of all possible combination for models for a particular source 
    '''

    teachers = {}
    
    for dom_adapts in s2t[source]:

        print('Loading weights for ', dom_adapts)

        netF = network.ViT().cuda()
        netB = network.feat_bootleneck(type='bn', feature_dim=netF.in_features,bottleneck_dim=256).cuda()
        netC = network.feat_classifier(type='wn', class_num=65, bottleneck_dim=256).cuda()

        modelpathF = f'ckps/target/uda/office-home/{dom_adapts}/target_F_par_0.3.pt'
        netF.load_state_dict(torch.load(modelpathF))

        modelpathB = f'ckps/target/uda/office-home/{dom_adapts}/target_B_par_0.3.pt'
        netB.load_state_dict(torch.load(modelpathB))

        modelpathC = f'ckps/target/uda/office-home/{dom_adapts}/target_C_par_0.3.pt'
        netC.load_state_dict(torch.load(modelpathC))
        
        netF.eval()
        netB.eval()
        netC.eval()

        teachers[dom_adapts] =  [netF,netB, netC]
    
    print('Teachers made Successfully !')

    netF = network.ViT().cuda()
    netB = network.feat_bootleneck(type='bn', feature_dim=netF.in_features,bottleneck_dim=256).cuda()
    netC = network.feat_classifier(type='wn', class_num=65, bottleneck_dim=256).cuda()
    
    netF.train()
    netB.train()
    netC.train()

    student = [netF,netB, netC]
    return teachers, student

def office_home_dataloaders(path, batch_size=32):
    
    domains = ['Art', 'Clipart', 'Product', 'RealWorld']
   
   # Transformation

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    trans = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    domain_dataloaders = {}

    for dom in domains:
        dataset = torchvision.datasets.ImageFolder(f'{path}/{dom}', transform=trans)
        domain_dataloaders[dom] = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return domain_dataloaders

def test_model(model_list, dataloader):
    # print(len(dataloader))
    netF = model_list[0]
    netB = model_list[1]
    netC = model_list[2]

    correct = 0
    total = 0
    
    with torch.no_grad():
        print('Started Testing')
        iter_test = iter(dataloader)
        for _ in tqdm(range(len(dataloader))):
            data = iter_test.next()
            inputs = data[0].to('cuda')
            labels = data[1].to('cuda')

            outputs = netC(netB(netF(inputs)))
            # print(outputs)
            
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            

    print('Accuracy of the network on the test images: {}'.format(100 * correct / total))

def data_load(batch_size=32,txt_path='data/office-home'):
    ## prepare data

    def image_train(resize_size=256, crop_size=224, alexnet=False):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        return transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

    dsets = {}
    dset_loaders = {}
    train_bs = batch_size
    
    txt_files = {'Art' : f'{txt_path}/Art_list.txt', 
                'Clipart': f'{txt_path}/Clipart_list.txt', 
                'Product': f'{txt_path}/Product_list.txt',
                'RealWorld': f'{txt_path}/RealWorld_list.txt'}
                
    for domain, paths in txt_files.items(): 
        txt_tar = open(paths).readlines()

        dsets[domain] = ImageList_idx(txt_tar, transform=image_train())
        dset_loaders[domain] = DataLoader(dsets[domain], batch_size=train_bs, shuffle=True,drop_last=False)

    return dset_loaders

def train_sequential_KD(student_model_list, teacher_model_list, dataloader, temp_coeff=0.1):
    # print(len(dataloader))
    netF_student = student_model_list[0]
    netB_student = student_model_list[1]
    netC_student = student_model_list[2]

    netF_teacher = teacher_model_list[0]
    netB_teacher = teacher_model_list[1]
    netC_teacher = teacher_model_list[2]

    print('Started Training')

    iter_test = iter(dataloader)
    soft = nn.Softmax(dim=1)

    for _ in range(len(dataloader)):
        with torch.no_grad():
            data = iter_test.next()
            inputs = data[0].to('cuda')
            labels = data[1].to('cuda')

            # Teacher output
            teach_outputs = netC_teacher(netB_teacher(netF_teacher(inputs)))
            soft_op = soft(teach_outputs/temp_coeff)
        
        
        # Student Train

        student_outputs = netC_student(netB_student(netF_student(inputs)))
        
        
if __name__ == '__main__':
    
    source = 'Clipart'

    s2t = {'Art' : ['AC', 'AP', 'AR'], 'Clipart': ['CA', 'CP', 'CR'], 
            'Product': ['PA','PC','PR'], 'RealWorld': ['RA', 'RP', 'RC']}

    teachers, student = create_teachers_student(s2t, source)
    dom_dataloaders = data_load() #office_home_dataloaders('data/office-home')

    # test_model(teachers['CA'], dom_dataloaders['Art'])
    # test_model(teachers['CP'], dom_dataloaders['Product'])
    # test_model(teachers['CR'], dom_dataloaders['RealWorld'])
    
    train_sequential_KD(student, teachers['CA'], dom_dataloaders['Art'], temp_coeff=0.1)
