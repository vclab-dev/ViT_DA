import numpy as np
import sys
from collections import Counter
np.set_printoptions(threshold=sys.maxsize)

def rlcc(prev_pseudo_labels, pseudo_labels, soft_output, class_num,  alpha=0.9):

    consensus=np.zeros((class_num, class_num))
    for i in set(list(prev_pseudo_labels)):
        index_i = np.where(prev_pseudo_labels == i)
        for j in set(list(pseudo_labels)):
            index_j = np.where(pseudo_labels == j)
            intersect = np.intersect1d(index_i, index_j)
            union = np.union1d([i], pseudo_labels)
            consensus[i][j] = len(intersect)/len(union)
            
    sum = consensus.sum(axis=1) + 1e-8
    for i in range(class_num):
        consensus[i][:] = consensus[i][:]/(sum[i])
    print('consensus: ', consensus.shape)
    # print(consensus)
    prev_pseudo_labels = np.expand_dims(prev_pseudo_labels, axis=1)
    # print('prev pl',prev_pseudo_labels[:10])
    pseudo_labels = np.expand_dims(pseudo_labels, axis=1)
    # print('curr pl', pseudo_labels[:10])

    prop_prev_pl = np.matmul(soft_output, consensus)
    print(prop_prev_pl.shape)
    # print('propogated',prop_prev_pl[:10][:])
    
    refined = np.add(alpha*pseudo_labels, (1-alpha)*prop_prev_pl)
    # print('refined',refined[:10][:])
    return refined