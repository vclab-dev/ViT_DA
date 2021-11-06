import numpy as np
from collections import Counter

def rlcc(prev_pseudo_labels, pseudo_labels):
    m_t_prev = len(set(prev_pseudo_labels))
    m_t_curr = len(set(pseudo_labels))
    prev_pseudo_labels = np.array(prev_pseudo_labels)
    pseudo_labels = np.array(pseudo_labels)
    consensus=[]
    for i in set(prev_pseudo_labels):
        # print("-----i=", i)
        index_i = np.where(prev_pseudo_labels == i)
        consensus_j=[]
        for j in set(pseudo_labels):
            # print("-------j=", j)
            index_j = np.where(pseudo_labels == j)
            intersect = np.intersect1d(index_i, index_j)
            # print("intersect:", intersect)
            union = np.union1d([i], pseudo_labels)
            # print("union:", union)
            consensus_j.append(len(intersect)/len(union))
        consensus.append(consensus_j)            
    consensus = np.array(consensus)
    sum = consensus.sum(axis=1)
    for i in range(m_t_prev):
        consensus[i][:] = consensus[i][:]/sum[i]
    print(prev_pseudo_labels.shape)

    prop_pl = np.matmul(consensus.T, prev_pseudo_labels)
    print(prop_pl)
# def intersection(list1, list2):
#     return list(set(list1) & set(list2))

list1 = [1,1,2,3,4,1,2,1]
list2 = [1,2,1,3,4,1,2,5]
list_inter = rlcc(list1, list2)
# print(list_inter)