import argparse
from genericpath import exists
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

def test_model(model_list, dataloader, dataset_name=None):
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
            
    accuracy = 100 * correct / total
    print('Accuracy of the network on the {} images: {}'.format(dataset_name, accuracy))
    return accuracy, correct, total

def data_load(batch_size=64,txt_path='data/office-home'):
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

def dist_loss(t, s, T=0.1):
    soft = nn.Softmax(dim=1)

    prob_t = soft(t/T)
    log_prob_s = nn.LogSoftmax( dim=1)(s)
    dist_loss = -(prob_t*log_prob_s).sum(dim=1).mean()
    return dist_loss

def train_sequential_KD(student_model_list, teacher_model_list, dataloader, temp_coeff=0.1, num_epoch=1, log_interval=2):
    # print(len(dataloader))

    '''
        take a teacher model (already apated), take a dataloader and distil the knowledge to student 
    '''
    netF_student = student_model_list[0]
    netB_student = student_model_list[1]
    netC_student = student_model_list[2]

    netF_teacher = teacher_model_list[0]
    netB_teacher = teacher_model_list[1]
    netC_teacher = teacher_model_list[2]
    
    for param in netF_teacher.parameters():
        param.requires_grad = False

    for param in netB_teacher.parameters():
        param.requires_grad = False   
    
    for param in netC_teacher.parameters():
        param.requires_grad = False

    print('Started Training')

    soft = nn.Softmax(dim=1)
    list_of_params = list(netF_student.parameters()) + list(netB_student.parameters()) + list(netC_student.parameters()) 
    optimizer = optim.SGD(list_of_params, lr=0.001, momentum=0.9)
    epoch_loss=  0.0
    for epoch in range(num_epoch):
        running_loss = 0.0
        iter_test = iter(dataloader)

        for i in range(len(dataloader)):
    
            with torch.no_grad():
                data = iter_test.next()
                inputs = data[0].to('cuda')
                labels = data[1].to('cuda')
                # Teacher output
                teach_outputs = netC_teacher(netB_teacher(netF_teacher(inputs)))
                # soft_teach_op = soft(teach_outputs/temp_coeff)
            
            # Student Train
            optimizer.zero_grad()
            student_outputs = netC_student(netB_student(netF_student(inputs)))
            # soft_student_op = soft(student_outputs)
            if epoch <=5:
                loss = nn.MSELoss()(teach_outputs,student_outputs)
            else:
                loss = dist_loss(teach_outputs,student_outputs)
            # loss = nn.MSELoss()(teach_outputs,student_outputs)#dist_loss(soft_teach_op,soft_student_op T=temp_coeff)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            epoch_loss+=running_loss
            if i % log_interval == 0 : 

                print('[%d / %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / log_interval))
                running_loss = 0.0
        print("Epoch loss:",epoch_loss/len(dataloader))
        wandb.log({"Epoch Loss":epoch_loss/len(dataloader)})
        epoch_loss=0.0
    print('Finished Training')

    return [netF_student, netB_student, netC_student]

def mixed_data_load(batch_size=64,txt_path='data/office_home_mixed'):
    '''
        Returns a dict of dataloaders on mixed set.
        dataloader[source] will have all the target except source.  
    '''

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
    
    txt_files = {'RealWorld' : f'{txt_path}/Art_Clipart_Product.txt', 
                'Procuct': f'{txt_path}/Art_Clipart_Real_World.txt', 
                'Clipart': f'{txt_path}/Art_Product_Real_World.txt',
                'Art': f'{txt_path}/Clipart_Product_Real_World.txt'}
                
    for domain, paths in txt_files.items(): 
        txt_tar = open(paths).readlines()

        dsets[domain] = ImageList_idx(txt_tar, transform=image_train())
        dset_loaders[domain] = DataLoader(dsets[domain], batch_size=train_bs, shuffle=True,drop_last=False)

    return dset_loaders

def multi_domain_avg_acc(student, test_on=None):

    '''
        Given a student model and a set of domain, this func returns the avg accuracy

        Input:
            student : Student model
            test_on : List of domain to be tested on eg ['RealWorld', 'Clipart', 'Art']
        
        Return:
            Average accuracy of all domains
    '''

    if test_on is not None:
        correct,total = [], []

        for sample in test_on:
            print(f'Testing Acc on {sample}')
            _,corr,tot = test_model(student, dom_dataloaders[sample], dataset_name=sample)
            correct.append(corr)
            total.append(tot)

        avg_acc = 100*sum(correct)/sum(total)
        
        combined_name = '_'.join(test_on)
        print(f'\n\n Average Accuracy on {combined_name}: {avg_acc} \n\n')
        return avg_acc
    
    else:
        raise ValueError

        

if __name__ == '__main__':
    
    source = 'Art'
    wandb.init(project='cycle_MTDA_ViT', entity='vclab',name=f"{source} to others Cycle")

    save_weight_dir = 'ckps/office-home'

    save_path = f'{save_weight_dir}/{source}_to_others'
    os.makedirs(save_path,exist_ok=True)
    s2t = {'Art' : ['AC', 'AP', 'AR'], 'Clipart': ['CA', 'CP', 'CR'], 
            'Product': ['PA','PC','PR'], 'RealWorld': ['RA', 'RP', 'RC']}

    teachers, student = create_teachers_student(s2t, source)
    dom_dataloaders = data_load(batch_size=24) #office_home_dataloaders('data/office-home')
    # mix_dataloaders = mixed_data_load(batch_size=64)

    # test_model(teachers['CA'], dom_dataloaders['Art'])
    # test_model(teachers['CP'], dom_dataloaders['Product'])
    # test_model(teachers['CR'], dom_dataloaders['RealWorld'])
    # test_model(teachers['CR'], mix_dataloaders['Clipart']) 
    
    total_epoch = 100
    for i in range(total_epoch):
        print(f'\n\n#### CYCLE {i} #####\n\n')

        student = train_sequential_KD(student, teachers['AR'], dom_dataloaders['RealWorld'], temp_coeff=0.1, num_epoch=1, log_interval=5)
        # print('Testing Acc on Art')
        # test_model(student, dom_dataloaders['Art'])

        student = train_sequential_KD(student, teachers['AC'], dom_dataloaders['Clipart'], temp_coeff=0.1, num_epoch=1, log_interval=5)
        # print('Testing Acc on Clipart')
        # test_model(student, dom_dataloaders['Clipart'])   
        # print('Testing Acc on Art')
        # test_model(student, dom_dataloaders['Art'])

        student = train_sequential_KD(student, teachers['AP'], dom_dataloaders['Product'], temp_coeff=0.1, num_epoch=1, log_interval=5)
        if i % 10 ==0:
            test_domains = ['Product', 'Clipart', 'RealWorld']
            avg_acc = multi_domain_avg_acc(student,test_on=test_domains)
            
            wandb.log({"avg_acc":avg_acc, 'cycle': i})
    
            torch.save(student[0].state_dict(), osp.join(save_path, f"target_F_{source}.pt"))
            torch.save(student[1].state_dict(), osp.join(save_path, f"target_B_{source}.pt"))
            torch.save(student[2].state_dict(), osp.join(save_path, f"target_C_{source}.pt"))