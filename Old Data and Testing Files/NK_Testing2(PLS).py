#Zach Wu 2016
#Note: NK Testing does not account for stop codons or deletions

#Add stipulation for unique mutations?
#(ie: if one site is mutated already, do not mutate again)

import numpy as np
import sys
import random
import NK_Landscape
from scipy.stats import poisson
from scipy.stats import pearsonr

import warnings
warnings.simplefilter('once')

sample_parent = [random.randint(0,19) for i in range(10)]
num_AA = 20

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

def mutate(p2, n):
    '''mutates parent n times
    '''
    current_seq2 = p2

    for i in range(n):
        #print(current)
        mut_index = random.randint(0,len(p2)-1)
        curr_AA = current_seq2[mut_index]
        poss_AA = set([j for j in range(num_AA)])
        poss_AA.remove(curr_AA)
        mut_AA = list(poss_AA)[random.randint(0,len(poss_AA)-1)]
        current_seq2[mut_index] = mut_AA

    return current_seq2

def epm_library(parent, library_size, rate = 1):
    '''parent sequence should be in [19,3,0,1] format
        rate is the average number of mutations
        I'm just going to assume a poisson distribution for now
        check http://www.ncbi.nlm.nih.gov/pubmed/15939434

        -Fox assumes gamma distribution
    '''
    dist_countL, num_mutL = EPM_Poisson_countd(rate, library_size)

    seq_list = []
    seq_list.append(parent[:])
    for i, num_mut in enumerate(num_mutL):
        for j in range(dist_countL[i]):
            seq_list.append(mutate(parent, num_mut)[:])

    return seq_list

def single_mutant_library(parent, n):
    ''' sequence should be in [19,3,0,1] format
        n is length of walk
    '''
    current_seq = list(parent)[:]
    seq_list = []
    seq_list.append(parent[:])
    for i in range(n):
        temp = current_seq[:]
        temp = mutate(current_seq, 1)

        current_seq = temp[:]
        seq_list.append(temp)
    return seq_list

###############################################################
###############################################################
###############################################################

import time

#run models
import PartialLeastSquares as PLS


K_list = [0,1,2,3]
n = 40                    #Sample length = 40
library_size = 3*n        #Library size = 3*n


#whooo
f = open('NK_Model_Testing(PLS).txt', 'w')
f2 = open('NK_Model_Correlations(PLS).txt','w')

predict_master_list = []
true_master_list = []

#Specify parent sequence and library
parent = [np.random.randint(num_AA) for i in range(n)]
epm_lib = single_mutant_library(parent, library_size)    #Currently only looking at single mutant walk
                                                         #repeat for random mutants?

#Featurize
lib_features = NK_Featurize(epm_lib)

#Run Models
beg = time.clock()
for K in K_list:
    print('K = ' + str(K))

    f.write('\n##########\nNew Fitness Landscape \nK = ' + str(K) + '\n##########\n')
    f2.write('K Value = ' + str(K) + '\n------\n')
    #Make NK_Landscape
    landscape = NK_Landscape.NKLandscape(n,K, savespace = False, epi_dist = 'gamma')
    interactions_list = landscape.nk_interactions()
    epistatic_list = landscape.nk_epistatic()

    #Determine Fitnesses
    lib_fitnessL = []
    for i in range(len(lib_features)):
        fitness = landscape.get_Energy(epm_lib[i])
        lib_fitnessL.append(fitness)

    predict_list_by_clf = []
    true_list_by_clf = []


    f.write('\nNew CLF:\n' + 'PLS' + str(j) + '\n---------\n')

    #Fit to model and save fitnesses:
    for j in range(len(lib_features)):
        X_test, Y_test = [lib_features[j]], [lib_fitnessL[j]]
        X_train = lib_features[0:j] + lib_features[j+1:]
        Y_train = lib_fitnessL[0:j] + lib_fitnessL[j+1:]

        np.savetxt('X_temp.csv', X_train, delimiter = ',')
        np.savetxt('Y_temp.csv', Y_train, delimiter = ',')

        try:
            pls = PLS.PartialLeastSquares(XMatrix_file = 'X_temp.csv', YMatrix_file = 'Y_temp.csv', epsilon = 0.001)
            pls.get_XMatrix_from_csv()
            pls.get_YMatrix_from_csv()
            B = pls.PLS()

            Y_predicted = B * X_test

            print(type(Y_predicted))
            predict_list.append(Y_predicted[0])
            true_list.append(Y_test[0])

            f.write(str(Y_predicted[0]) + ',' + str(Y_test[0]) + '\n')


    #Determine Pearson's R
    r = np.corrcoef(predict_list, true_list)
    write_me = 'Rsquared : ' + str(r[0,1]**2) + '\n'
    print(write_me)
    f.write(write_me)
    f2.write(str(r[0,1]**2) + ',')
    predict_list_by_clf.append(predict_list)
    true_list_by_clf.append(true_list)

f.close()
f2.close()

print('end')
print('runtime = ' + str(time.clock() - beg))
