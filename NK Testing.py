#Zach Wu 2016
#Note: NK Testing does not account for stop codons or deletrions

import numpy as np
import sys
import random
import NK_Landscape
from scipy.stats import poisson

sample_parent = [random.randint(0,19) for i in range(10)]
num_AA = 20

def single_mutant_walk(parent, n):
    '''sequence should be in [19,3,0,1] format
        n is length of walk
    '''
    current = parent
    seq_list = [current]

    for i in range(n):
        #print(current)
        mut_index = random.randint(0,len(parent)-1)
        curr_AA = current[mut_index]
        poss_AA = set([j for j in range(num_AA)])
        poss_AA.remove(curr_AA)
        mut_AA = list(poss_AA)[random.randint(0,len(poss_AA)-1)]
        current[mut_index] = mut_AA
        seq_list.append(current)
    print(seq_list[-1])
    if n == 1: #if n == 1, probably using this for epm.
        return current
    else:
        return seq_list

def epm(parent, library_size, rate = 1):
    '''parent sequence should be in [19,3,0,1] format
        rate is the average number of mutations
        I'm just going to assume a poisson distribution for now
        check http://www.ncbi.nlm.nih.gov/pubmed/15939434

        -Fox assumes gamma distribution
    '''
    dist_countL, num_mutL = EPM_Poisson_countd(rate, library_size)

    print(str(dist_countL) + ' ' + str(num_mutL))

    seq_list = [] #does not include parent
    parent_temp = parent

    for i,num_mut in enumerate(num_mutL):
        for j in range(dist_countL[i]):
            seq_list.append(mutate_sequence(parent_temp, num_mut))
    return seq_list

def mutate_sequence(parent2, num_mutations):
#     mut_seq = parent
#     mut_index = random.randint(0,len(parent)-1)
#     curr_AA = mut_seq[mut_index]
#     poss_AA = set([j for j in range(20)])

#     list_mut = []
    mut_seq = parent2
    for i in range(num_mutations):
        mut_seq = single_mutant_walk(mut_seq,1)
    print(mut_seq)

    return mut_seq

def EPM_Poisson_countd(mu, library_size):
    #returns the Poisson mutation rate distribution for a given library size

    probs_list = []
    mut_list = []
    alpha = 1-1/(library_size*10)
    a,b = poisson.interval(alpha, mu, loc=0)
    a = int(a)
    b = int(b)
    for k in range(a,b+1):
        k_count = int(round(poisson.pmf(k,mu)*library_size,0))
        if k_count != 0:
            probs_list.append(k_count)
            mut_list.append(k)
    dif = sum(probs_list) - library_size
#     print(dif)
    mutation_list = [i for i in range(a,b+1)]
    index = mutation_list.index(mu)
    probs_list[index] -= dif

#     print(probs_list)
#     print(sum(probs_list))
    return probs_list, mut_list

def NK_Featurize(seqList):
    featurized_seq = []
    for i, seq in enumerate(seqList):
        temp_seq = [0] * len(seq) * num_AA
        for j, AA in enumerate(seq):
            temp_seq[j*20 + AA] = 1
        featurized_seq.append(temp_seq)
    return featurized_seq
###############################################################
###############################################################
###############################################################


import time
n = 10                #sequence length
library_size = 3*n

beg = time.clock()
test = NK_Landscape.NKLandscape(n,1, savespace = False, epi_dist = 'gamma')
interactions_list = test.nk_interactions()
epistatic_list = test.nk_epistatic()
end = time.clock()
print(-beg+end)

parent = [np.random.randint(num_AA) for i in range(n)]
print('Parent:' + str(parent))
#library1 = epm(parent, library_size, rate=3)
library2 = single_mutant_walk(parent,library_size)
print('end')


#library1_features = NK_Featurize(library1)
library2_features = NK_Featurize(library2)

library1_fitnessL = []
library2_fitnessL = []


# print(interactions_list)
# for i in range(len(library1_features)):
#     t1 = test.get_Energy(library1[i])
print(library2)