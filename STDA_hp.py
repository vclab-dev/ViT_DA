import argparse
import os, sys
import pandas as pd
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from loss import KnowledgeDistillationLoss, SoftCrossEntropyLoss
from timm.data.auto_augment import rand_augment_transform # timm for randaugment
from rlcc import rlcc
np.set_printoptions(threshold=sys.maxsize)

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

#tfm = rand_augment_transform(config_str='rand-m9-mstd0.5')
def strong_augment(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        rand_augment_transform(config_str='rand-m9-mstd0.5',hparams={'translate_const': 117}),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])


def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()
    txt_eval_dn = open(args.txt_eval_dn).readlines()

    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                         drop_last=True)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 3, shuffle=False, num_workers=args.worker,
                                      drop_last=False)
    dsets["strong_aug"] = ImageList_idx(txt_test, transform=strong_augment())
    dset_loaders["strong_aug"] = DataLoader(dsets["strong_aug"], batch_size=train_bs * 3, shuffle=False, num_workers=args.worker,
                                       drop_last=False)
    if args.dset =='domain_net':
        dsets["eval_dn"] = ImageList_idx(txt_eval_dn, transform=image_train())
        dset_loaders["eval_dn"] = DataLoader(dsets["eval_dn"], batch_size=train_bs, shuffle=False, num_workers=args.worker,
                                            drop_last=False)
    else:
        dset_loaders["eval_dn"] = dset_loaders["test"]

    return dset_loaders,dsets

def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
    #wandb.log({"Test accuracy": accuracy*100, "Test Mean Entropy": mean_ent})

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy * 100, mean_ent

def get_pseudo_gt(data_batch, netB, netF,netC):
    netB.eval()
    netF.eval()
    features_test = netB(netF(data_batch))
    outputs_test = netC(features_test)
    netB.train()
    netF.train()
    return outputs_test


def get_strong_aug(dataset, idx):
    aug_img = torch.cat([dataset[i][0].unsqueeze(dim=0) for i in idx],dim=0)
    return aug_img

def train_target(args):
    dset_loaders,dsets = data_load(args)
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net, se=args.se, nl=args.nl).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()
    elif args.net == 'vit':
        netF = network.ViT().cuda()
    elif args.net == 'deit_s':
        netF = torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224', pretrained=True).cuda()
        netF.in_features = 1000
    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    #print(netF.state_dict().keys())
    

    modelpath = args.output_dir_src + '/source_F.pt'
    netF.load_state_dict(torch.load(modelpath))
    
    modelpath = args.output_dir_src + '/source_B.pt'
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(modelpath))
    # netC.eval()
    print('Model Loaded')
    # for k, v in netC.named_parameters():
    # 	v.requires_grad = False

    param_group = []
    param_group_cls = []
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False
    for k, v in netC.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)
    # optimizer2 = optim.SGD(param_group_cls)
    # optimizer2 = op_copy(optimizer2)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0

    print('Training Started')
    max_acc = 0
    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_test.next()
            #inputs_test_stg = strong_aug_list[tar_idx]
            inputs_test_stg = get_strong_aug(dsets["strong_aug"], tar_idx)

        if inputs_test.size(0) == 1:  #Why this?
            continue
        if (iter_num % interval_iter == 0 and args.cls_par > 0):
            netF.eval()
            netB.eval()
            netC.eval()
            ################################ Done ###################################
            print('Starting to find Pseudo Labels! May take a while :)')
            mem_label, soft_output, dd, mean_all_output, actual_label = obtain_label(dset_loaders['test'], netF, netB, netC, args) # test loader same as targe but has 3*batch_size compared to target and train
            if iter_num == 0:
                prev_mem_label = mem_label
                if args.soft_pl:
                    mem_label = dd
            else:
                dict_pl = {'Actual Label':actual_label, 'Prev Pseudo Label': prev_mem_label, 'Curr Pseudo Labels': mem_label}
                mem_label = rlcc(prev_mem_label, mem_label, dd, args.class_num)
                if not args.soft_pl:
                    # print('not soft')
                    mem_label = mem_label.argmax(axis=1).astype(int)
                    refined_label = mem_label
                else:	
                    refined_label = mem_label.argmax(axis=1)
                dict_pl.update({'Refined Pseudo Labels':refined_label})
                prev_mem_label = refined_label
                # print(mem_label.dtype)
                if iter_num % (interval_iter*10) == 0:
                    print('Write to CSV')
                    df = pd.DataFrame(dict_pl)
                    df.to_csv('rlcc_cmp.csv'+str(args.t), mode = 'a')
            print('Completed finding Pseudo Labels\n')
            mem_label = torch.from_numpy(mem_label).cuda()
            dd = torch.from_numpy(dd).cuda()
            mean_all_output = torch.from_numpy(mean_all_output).cuda()

            netF.train()
            netB.train()
            netC.train()

        inputs_test_wk = inputs_test.cuda()
        inputs_test_stg = inputs_test_stg.cuda()
        inputs_test = torch.cat([inputs_test_wk,inputs_test_stg],dim=0)

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        # features_test = netB(netF(inputs_test))
        # outputs_test = netC(features_test)
        # outputs_test = get_pseudo_gt(inputs_test, netB, netF,netC)
        # gt_orig = nn.Softmax(dim=1)(outputs_test)
        # ratio = mean_all_output/torch.mean(gt_orig,dim=0)
        # gt_w = nn.Softmax(dim=1)(gt_orig*ratio)

        features = netB(netF(inputs_test))
        outputs = netC(features)


        if args.cls_par > 0:
            with torch.no_grad():
                pred = mem_label[tar_idx]
                #pred_soft = dd[tar_idx] # vikash
            
            # print(pred.dtype, outputs.dtype)
            sig = 1/(1+np.exp(-(iter_num/max_iter))) 
            if args.soft_pl:
                print("Soft Cross Entropy")
                classifier_loss = SoftCrossEntropyLoss(outputs[0:args.batch_size], pred)
                # classifier_loss2 = nn.CrossEntropyLoss()(outputs[0:args.batch_size], torch.argmax(pred, dim=1))
                # classifier_loss = sig*classifier_loss2 + (1-sig)*classifier_loss1
            else:
                print("Hard Cross Entropy")
                classifier_loss = nn.CrossEntropyLoss()(outputs[0:args.batch_size], pred)

            classifier_loss *= args.cls_par
            # m = 0.9*np.sin(np.minimum(np.pi/2,np.pi*iter_num/max_iter))
            # classifier_loss *=m
            # if args.kd:
            #     kd_loss = KnowledgeDistillationLoss()(outputs_test, outputs_teacher/0.1) # vikash
            #     classifier_loss += kd_loss
            # if iter_num < interval_iter and args.dset == "visda-2017":
            #     classifier_loss *= 0
        else:
            classifier_loss = torch.tensor(0.0).cuda()
        
        wandb.log({'classifier loss':classifier_loss})
        #with torch.no_grad():
            #gt_w = distributed_sinkhorn(outputs_test).detach()
            #gt_s = distributed_sinkhorn(outputs_stg).detach()
            #print(gt_w,gt_s)
        # consistancy_loss = 0.5*(torch.mean(loss.soft_CE(nn.Softmax(dim=1)(outputs_stg),gt_w)) + torch.mean(loss.soft_CE(nn.Softmax(dim=1)(outputs_test),gt_s)))
        # classifier_loss += consistancy_loss
        if args.ent:
            
            softmax_out = nn.Softmax(dim=1)(outputs)		# find number of psuedo sample per class for handling class imbalance for entropy maximization
            entropy_loss = torch.mean(loss.Entropy(softmax_out))#softmax_outputs_stg = nn.Softmax(dim=1)(outputs_stg)
            #entropy_loss = torch.mean(loss.soft_CE(softmax_outputs_stg,gt_w))
            en_loss = entropy_loss.item()
            #entropy_loss = dist_loss(outputs_test, outputs_test,T=1.0)
            #wandb.log({"Entropy Loss":entropy_loss.item()})
            #entropy_loss = torch.mean(loss.Entropy(softmax_out))
            if args.gent:
                #softmax_out = nn.Softmax(dim=1)(outputs)
                msoftmax = softmax_out.mean(dim=0)
                #msoftmax_stg = softmax_outputs_stg.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                gen_loss = gentropy_loss.item()
                entropy_loss -= gentropy_loss
            #m = 0.9*np.sin(np.minimum(np.pi/2,np.pi*iter_num/max_iter))
            im_loss = entropy_loss * args.ent_par
            #print("cls loss:{} en loss:{} gen loss:{} im_loss:{}".format(classifier_loss.item(), en_loss, gen_loss, im_loss.item()))
            #im_loss = entropy_loss * m
            #wandb.log({"Im Loss":im_loss.item()})
            #wandb.log({'CE Loss': classifier_loss.item()})
            classifier_loss += im_loss
        args.consistancy = True
        if args.consistancy:
            expectation_ratio = mean_all_output/torch.mean(softmax_out[0:args.batch_size],dim=0)
            #consistency_loss = 0.5*(dist_loss(outputs[args.batch_size:],outputs[0:args.batch_size]) + dist_loss(outputs[0:args.batch_size],outputs[args.batch_size:]))
            with torch.no_grad():
                soft_label_norm = torch.norm(softmax_out[0:args.batch_size]*expectation_ratio,dim=1,keepdim=True)
                soft_label = (softmax_out[0:args.batch_size]*expectation_ratio)/soft_label_norm
                #print(soft_label.shape)
            consistency_loss = args.const_par*torch.mean(loss.soft_CE(softmax_out[args.batch_size:],soft_label))
            #print("=====================::",consistency_loss)
            cs_loss = consistency_loss.item()
            #print("cls loss:{} en loss:{} gen loss:{} im_loss:{} consistancy_loss:{}".format(classifier_loss.item(), en_loss, gen_loss, im_loss.item(),cs_loss))
            classifier_loss += consistency_loss
        wandb.log({"Total loss":classifier_loss.item(), "en loss":en_loss, "gen loss":gen_loss, "im_loss":im_loss})
            

        #classifier_loss = L2(outputs_stg,outputs_test)
        optimizer.zero_grad()
        classifier_loss.backward()
        print(f'Task: {args.name}, Iter:{iter_num}/{max_iter} \t train_classifier_loss {classifier_loss.item()}')
        optimizer.step()
        # EMA update for the teacher
        # with torch.no_grad():
        #     m = 0.9*np.sin(np.minimum(np.pi/2,np.pi*iter_num/max_iter))  # momentum parameter # try with 0.1(vikash)
        #     for param_q, param_k in zip(netF.parameters(), netF_t.parameters()):
        #         param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
        #     for param_q, param_k in zip(netB.parameters(), netB_t.parameters()):
        #         param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            acc_eval_dn, _ = cal_acc(dset_loaders["eval_dn"], netF, netB, netC, False)
            if acc_eval_dn > max_acc:
                max_acc=acc_eval_dn
            wandb.log({"STDA_Test_Accuracy":acc_eval_dn})

            log_str = '\nTask: {}, Iter:{}/{}; Final Eval test = {:.2f}%'.format(args.name, iter_num, max_iter, acc_eval_dn)
            # acc_t_te, _ = cal_acc(dset_loaders['test'], netF_t.eval(), netB_t.eval(), netC, False)
            # log_str = 'Task: {}, Iter:{}/{}; Accuracy using teacher = {:.2f}%'.format(args.name, iter_num, max_iter, acc_t_te)            
            torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
            torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_" + args.savename + ".pt"))
            torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))
            print('model saved')

            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')

            if args.earlystop:
                print('Stopping Early!')
                return netF, netB, netC

            netF.train()
            netB.train()
            netC.train()

    if args.issave:
        torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
        torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_" + args.savename + ".pt"))
        torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))
    print('Maximum Accuracy: ', max_acc)
    return netF, netB, netC

def dist_loss(input, target, T=0.1):
    soft = nn.Softmax(dim=1)

    prob_t = soft(target/T)
    log_prob_s = nn.LogSoftmax(dim=1)(input)
    dist_loss = -(prob_t*log_prob_s).sum(dim=1).mean()
    return dist_loss
def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    print(s)
    return s


def obtain_label(loader, netF, netB, netC, args):
    start_test = True
    # Accumulate feat, logint and gt labels
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    ##################### Done ##################################
    all_output = nn.Softmax(dim=1)(all_output)

    mean_all_output = torch.mean(all_output,dim=0).numpy()
    # print(all_output.shape)
    # ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    # unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0]) # find accuracy on test sampels
    # find centroid per class
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()######### Not Clear (looks like feature normalization though)#######
    ### all_fea: extractor feature [bs,N]
    # print(all_fea.shape)
    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)                  # Number of classes
    aff = all_output.float().cpu().numpy()
    ### aff: softmax output [bs,c]

    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None]) # got the initial normalized centroid (k*(d+1))
    cls_count = np.eye(K)[predict].sum(axis=0) # total number of prediction per class
    labelset = np.where(cls_count >= args.threshold) ### index of classes for which same sampeled have been detected # returns tuple
    labelset = labelset[0] # index of classes for which samples per class greater than threshold
    
    #dd = cdist(all_fea, initc[labelset], args.distance) # N*K
    #print(all_fea.shape, initc[labelset].shape)
    dd = all_fea@initc[labelset].T
    dd = np.exp(dd)
    pred_label = dd.argmax(axis=1) # predicted class based on the minimum distance
    pred_label = labelset[pred_label] # this will be the actual class
    # acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    # log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
    # args.out_file.write(log_str + '\n')
    # args.out_file.flush()
    # print(log_str+'\n')

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        #dd = cdist(all_fea, initc[labelset], args.distance)
        dd = all_fea@initc[labelset].T
        dd = np.exp(dd)
        pred_label = dd.argmax(axis=1)
        pred_label = labelset[pred_label]

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    wandb.log({"Pseudo_Label_Accuracy":acc*100})
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str + '\n')
    #exit(0)
    dd = F.softmax(torch.from_numpy(dd), dim=1)
    return pred_label, all_output.cpu().numpy(), dd.numpy().astype('float32') ,mean_all_output, all_label.cpu().numpy().astype(np.uint8)

def distributed_sinkhorn(out,eps=0.1, niters=3,world_size=1):
    Q = torch.exp(out / eps).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] * world_size # number of samples to assign
    K = Q.shape[0] # how many prototypes
    # make the matrix sums to 1
    # Q = torch.log(Q)
    sum_Q = torch.sum(Q)
    # #dist.all_reduce(sum_Q)
    Q /= sum_Q
    #print(Q)
    for it in range(niters):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        #dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    #print(Q)
    #exit(0)
    return Q.t()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rand-Augment')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=3, help="max iterations")
    parser.add_argument('--interval', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home',
                        choices=['visda-2017', 'office', 'office-home', 'office-caltech', 'pacs', 'domain_net'])
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--net', type=str, default='vit', help="alexnet, vgg16, resnet50, res101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")

    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--kd', type=bool, default=False)
    parser.add_argument('--se', type=bool, default=False)
    parser.add_argument('--nl', type=bool, default=False)

    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--cls_par', type=float, default=0.2)
    parser.add_argument('--const_par', type=float, default=0.2)
    parser.add_argument('--ent_par', type=float, default=1.3)

    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--output', type=str, default='san', help='Save ur weights here')
    parser.add_argument('--output_src', type=str, default='san', help='Load SRC training wt path')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--issave', type=bool, default=True)
    parser.add_argument('--wandb', type=int, default=0)
    parser.add_argument('--earlystop', type=int, default=0)
    parser.add_argument('--soft_pl', action='store_true')
    parser.add_argument('--suffix', type=str, default='')
    
    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'visda-2017':
        names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 7
    if args.dset == 'pacs':
        names = ['art_painting', 'cartoon', 'photo', 'sketch']
        args.class_num = 7
    if args.dset =='domain_net':
        names = ['clipart', 'infograph', 'painting', 'quickdraw','sketch', 'real']
        args.class_num = 345

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    for i in range(len(names)):

        if i == args.s:
            continue
        args.t = i

        folder = './data/'
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '.txt'
        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '.txt'
        if args.dset =='domain_net':
            args.txt_eval_dn = folder + args.dset + '/' + names[args.t] + '_test.txt'
        else:
            args.txt_eval_dn = args.t_dset_path

        mode = 'online' if args.wandb else 'disabled'
        
        import wandb
        if args.dset == 'office-home':
            wandb.init(project='STDA_Office-home', entity='vclab', name=f'{names[args.s]} to {names[args.t]} '+args.suffix, reinit=True,mode=mode)
        if args.dset == 'office':
            wandb.init(project='STDA_Office31', entity='vclab', name=f'{names[args.s]} to {names[args.t]} '+args.suffix, reinit=True,mode=mode)

        args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.s][0].upper())
        args.output_dir = osp.join(args.output, 'STDA', args.dset, names[args.s][0].upper() + names[args.t][0].upper())
        args.name = names[args.s][0].upper() + names[args.t][0].upper()

        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        args.savename = 'par_' + str(args.cls_par)
        
        args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
        args.out_file.write(print_args(args) + '\n')
        args.out_file.flush()
        train_target(args)