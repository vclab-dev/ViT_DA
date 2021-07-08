import argparse
from genericpath import exists
from logging import disable
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
from data import ImageList, ImageList_idx
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from loss import KnowledgeDistillationLoss
import wandb
from tqdm import tqdm
import random
from image_target import obtain_label

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

def create_teachers_student(s2t, source, student_arch='rn50'): 

    '''
        Create ensemble of teachers and one student model 
        teachers : Dict of all possible combination for models for a particular source 
    '''

    teachers = {}
    
    for dom_adapts in s2t[source]:

        print('Loading weights for ', dom_adapts)

        netF = network.ViT().cuda()
        netB = network.feat_bootleneck(type='bn', feature_dim=netF.in_features,bottleneck_dim=256).cuda()
        netC = network.feat_classifier(type='wn', class_num=num_classes, bottleneck_dim=256).cuda()

        modelpathF = f'{args.adapted_wt_dir}/{dom_adapts}/target_F_par_0.3.pt'
        #print(modelpathF)
        # print(netF.state_dict().keys())
        # print('\n \n \n \n')
        # exit(0)
        netF.load_state_dict(torch.load(modelpathF))
        

        modelpathB = f'{args.adapted_wt_dir}/{dom_adapts}/target_B_par_0.3.pt'
        netB.load_state_dict(torch.load(modelpathB))

        modelpathC = f'{args.adapted_wt_dir}/{dom_adapts}/target_C_par_0.3.pt'
        netC.load_state_dict(torch.load(modelpathC))
        
        netF.eval()
        netB.eval()
        netC.eval()

        teachers[dom_adapts] =  [netF,netB, netC]
    
    print('Teachers made Successfully !')
    if student_arch == 'rn50':
        netF = network.ResBase(res_name='resnet50').cuda()
    if student_arch == 'vit':
        netF = network.ViT().cuda()
    netB = network.feat_bootleneck(type='bn', feature_dim=netF.in_features,bottleneck_dim=1024).cuda()
    netC = network.feat_classifier(type='wn', class_num=num_classes, bottleneck_dim=1024).cuda()
    
    print(f'Created {student_arch} based student')
    netF.train()
    netB.train()
    netC.train()

    student = [netF,netB, netC]
    return teachers, student

def test_model(model, dataloader, dataset_name=None):
    # print(len(dataloader))
    netF = model[0]
    netB = model[1]
    netC = model[2]

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

def data_load(batch_size=64,txt_path='data/office', dset=None):
    ## prepare data

    def image(resize_size=256, crop_size=224, alexnet=False):
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
    
    dset = {}
    train_bs = batch_size
    
    if args.dset == 'office':
        txt_files = {'amazon' : f'{txt_path}/amazon.txt', 
                    'webcam': f'{txt_path}/webcam.txt', 
                    'dslr': f'{txt_path}/dslr.txt'}
    if args.dset == 'office-home':
        txt_files = {'Art' : f'{txt_path}/Art.txt', 
                'Clipart': f'{txt_path}/Clipart.txt', 
                'Product': f'{txt_path}/Product.txt',
                'RealWorld': f'{txt_path}/RealWorld.txt'}

    if args.dset == 'pacs':
        txt_files = {'art_painting' : f'{txt_path}/art_painting.txt', 
                'cartoon': f'{txt_path}/cartoon.txt', 
                'photo': f'{txt_path}/photo.txt',
                'sketch': f'{txt_path}/sketch.txt'}
    
    if args.dset == 'domain_net':
        txt_files = {'clipart': f'{txt_path}/clipart.txt',
                'infograph': f'{txt_path}/infograph.txt', 
                'painting':  f'{txt_path}/painting.txt', 
                'quickdraw': f'{txt_path}/quickdraw.txt', 
                'sketch':    f'{txt_path}/sketch.txt', 
                'real':      f'{txt_path}/real.txt'}

                
    for domain, paths in txt_files.items(): 
        txt_tar = open(paths).readlines()

        dsets[domain] = ImageList_idx(txt_tar, transform=image())
        dset_loaders[domain] = DataLoader(dsets[domain], batch_size=train_bs, shuffle=True,drop_last=True)


    return dset_loaders

def dist_loss(t, s, T=0.1):
    soft = nn.Softmax(dim=1)

    prob_t = soft(t/T)
    log_prob_s = nn.LogSoftmax( dim=1)(s)
    dist_loss = -(prob_t*log_prob_s).sum(dim=1).mean()
    return dist_loss

def train_sequential_KD(student_model, teacher_model, dataloader, curr_cycle= 0, max_cycles=100, temp_coeff=0.1, num_epoch=1, log_interval=2):
    # print(len(dataloader))

    '''
        take a teacher model (already apated), take a dataloader and distil the knowledge to student 
    '''
    netF_student = student_model[0]
    netB_student = student_model[1]
    netC_student = student_model[2]

    netF_teacher = teacher_model[0]
    netB_teacher = teacher_model[1]
    netC_teacher = teacher_model[2]
    
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

    
    max_iter = len(dataloader)
    for epoch in range(num_epoch):
        running_loss = 0.0
        iter_test = iter(dataloader)

        for i in range(len(dataloader)):
    
            with torch.no_grad():
                data = iter_test.next()
                inputs = data[0].to('cuda')
                labels = data[1].to('cuda')
                #tar_idx = data[2]
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
                loss = dist_loss(teach_outputs,student_outputs,T=temp_coeff)
            # loss = nn.MSELoss()(teach_outputs,student_outputs)#dist_loss(soft_teach_op,soft_student_op T=temp_coeff)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            epoch_loss+=running_loss
            if i % log_interval == log_interval-1 : 
                print_loss = running_loss / log_interval
                print(f'Cycle: {curr_cycle+1}/{max_cycles} Iter: {i + 1}/{max_iter} loss: {print_loss:.4f}')
                running_loss = 0.0
        print("Epoch loss:",epoch_loss/len(dataloader))
        wandb.log({"Epoch Loss":epoch_loss/len(dataloader)})
        epoch_loss=0.0
    print('Finished Training')

    return [netF_student, netB_student, netC_student]

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

    parser = argparse.ArgumentParser(description='Args parser for KD_MTDA')

    # training parameters
    parser.add_argument('-b', '--batch_size', default=32, type=int,help='mini-batch size (default: 32)')
    #parser.add_argument('--gpu_id', type=str, nargs='?', default='1', help="device id to run")
    parser.add_argument('-s', '--source', type=str,help='Select the source [amazon, dslr, webcam]')
    parser.add_argument('-e', '--epochs', default=100, type=int,help='select number of cycles')
    parser.add_argument('-w', '--use_wandb', default=0, type=int,help='Log to wandb or not [0 - dont use | 1 - use]')
    parser.add_argument('-a', '--arch', default='rn50', type=str,help='Select student vit or rn50 based (default: rn50)')

    parser.add_argument('-l', '--adapted_wt_dir', default='san/uda/office', type=str,help='Load 1S1T  adapted wts')
    parser.add_argument('-t','--txt', default='data/office', type=str,help='Load the dataset txt file')
    parser.add_argument('--save', default='save_wt_KTMTDA', type=str,help='Save directory')

    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--dset', type=str, default='office', choices=['VISDA-C', 'office', 'office-home', 'office-caltech', 'domain_net'])
    

    
    args = parser.parse_args()

    
    source = args.source
    max_cycles = args.epochs
    batch_size = args.batch_size
    save_weight_dir = args.save

    mode = 'online' if args.use_wandb else 'disabled'
    wandb.init(project='KD_MTDA_BMVC', entity='vclab',name=f"{source} to others {args.arch}", mode=mode)
    
    save_path = f'{save_weight_dir}/{args.dset}/{source}_to_others'
    os.makedirs(save_path,exist_ok=True)

    if args.dset == 'office':
        s2t = {'amazon' : ['AD', 'AW'], 'webcam': ['WA', 'WD'], 
            'dslr': ['DA','DW']}
        num_classes = 31
    
    if args.dset == 'office-home':
        s2t = {'Art' : ['AC', 'AP', 'AR'], 'Clipart': ['CA', 'CP', 'CR'], 
            'Product': ['PA','PC','PR'], 'RealWorld': ['RA', 'RP', 'RC']}

        num_classes = 65

    if args.dset == 'pacs':
        s2t = {'art_painting' : ['AC', 'AP', 'AS'], 'cartoon': ['CA', 'CP', 'CS'], 
        'photo': ['PA','PC', 'PS'], 'sketch': ['SA', 'SC', 'SP']}
        num_classes = 7
    
    if args.dset == 'domain_net':
        s2t = {'clipart': ['CI','CP','CQ','CR','CS'],
         'infograph': ['IC','IP','IQ','IR','IS'], 
         'painting':  ['PC','PI','PQ','PS','PR'], 
         'quickdraw': ['QC','QI','QP','QS','QR'], 
         'sketch':    ['SC','SI','SP','SQ','SR'], 
         'real':      ['RC','RI','RP','RQ','RS']}

        num_classes = 345

    teachers, student = create_teachers_student(s2t, source, student_arch=args.arch)
    dom_dataloaders = data_load(batch_size=batch_size, dset = args.dset, txt_path=args.txt) 

    if args.dset == 'office':                     
        sequential_select = { 'amazon': [['AD', 'dslr'], ['AW', 'webcam']],
                                    'webcam': [['WA', 'amazon'], ['WD', 'dslr']],
                                    'dslr': [['DA', 'amazon'], ['DW', 'webcam']]
                                    }
    if args.dset == 'office-home':
        sequential_select = { 'Art': [['AC', 'Clipart'], ['AP', 'Product'], ['AR', 'RealWorld']],
                                'Clipart': [['CA', 'Art'], ['CP', 'Product'], ['CR', 'RealWorld']],
                                'Product': [['PA', 'Art'], ['PC', 'Clipart'], ['PR', 'RealWorld']],
                                'RealWorld': [['RA' , 'Art'], ['RC', 'Clipart'], ['RP', 'Product']],
                                }

    if args.dset == 'pacs':
        sequential_select = { 'art_painting': [['AC', 'cartoon'], ['AP', 'photo'], ['AS', 'sketch']],
                                'cartoon': [['CA', 'art_painting'], ['CP', 'photo'], ['CS', 'sketch']],
                                'photo': [['PA', 'art_painting'], ['PC', 'cartoon'], ['PS', 'sketch']],
                                'sketch': [['SA', 'art_painting'], ['SC', 'cartoon'], ['SP', 'photo']]
                                }
    if args.dset == 'domain_net':
        sequential_select = { 'clipart':   [['CI','infograph'],['CP','painting'],['CQ','quickdraw'],['CR','real'],['CS','sketch']], 
                                'infograph': [['IC','clipart'],['IP','painting'],['IQ','quickdraw'],['IR','real'],['IS','sketch']], 
                                'painting':  [['PC','clipart'],['PI','infograph'],['PQ','quickdraw'],['PS','sketch'],['PR','real']], 
                                'quickdraw': [['QC','clipart'],['QI','infograph'],['QP','painting'],['QS','sketch'],['QR','real']], 
                                'sketch':    [['SC','clipart'],['SI','infograph'],['SP','painting'],['SQ','quickdraw'],['SR','real']], 
                                'real':      [['RC','clipart'],['RI','infograph'],['RP','painting'],['RQ','quickdraw'],['RS','sketch']]
                            }
    
    test_domains = list(sequential_select.keys())
    test_domains.remove(source)

    for i in range(max_cycles):
        print(f'\n\n#### CYCLE {i+1} #####\n\n')
        for adap_module, domain_sel in sequential_select[source]:
            print(f'Started Distillation from Teacher {adap_module} -> to student using {domain_sel} images\n')
            student = train_sequential_KD(student, teachers[adap_module], dom_dataloaders[domain_sel], 
                          curr_cycle=i, max_cycles=max_cycles, num_epoch=1, log_interval=5)

        if i % 10 ==0:

            avg_acc = multi_domain_avg_acc(student,test_on=test_domains)
            
            wandb.log({"avg_acc":avg_acc, 'cycle': i})

            print('Saving model at: ', save_path)
            torch.save(student[0].state_dict(), osp.join(save_path, f"target_F_{source}_{args.arch}.pt"))
            torch.save(student[1].state_dict(), osp.join(save_path, f"target_B_{source}_{args.arch}.pt"))
            torch.save(student[2].state_dict(), osp.join(save_path, f"target_C_{source}_{args.arch}.pt"))