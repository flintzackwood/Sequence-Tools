#Zach Wu 2016
#Note: NK Testing does not account for stop codons or deletions

#Adding new models from scikit learn

#Add stipulation for unique mutations?
#(ie: if one site is mutated already, do not mutate again)

#When COPYING lists, must use [:], else making EQUALITY statement

import numpy as np
import sys
import random
import NK_Landscape
from scipy.stats impocdrt poisson
from scipy.stats import pearsonr
import warnings
import time
from tqdm import tqdm
import csv
import pandas as pd
import seaborn
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
#warnings.simplefilter('once')

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
            seq_list.append(mutate(parent[:], num_mut)[:])

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
beg = time.clock()

K_list = [0,1,2,3]
rate_list = [1,2,3,4,5]
n = 400                    #Sample length = 40
library_size = 3*n        #Library size = 3*n

#whooo
f = open('NK_Model_Check.txt', 'w')

#Construct full library

a = [i for i in range(20)]
parent = [random.randint(0,19) for i in range(n)]
epm_lib = epm_library(parent, library_size, rate = 3)

title_list = []
fitness_list = []

for K in K_list:
    print('\n-------\nK = ' + str(K) + '\n-------\n')
    landscape = NK_Landscape.NKLandscape(n, K, savespace = False, epi_dist = 'gamma', epi_type = 'add')
    interactions_list = landscape.nk_interactions()
    epistatic_list = landscape.nk_epistatic()

    for rate in rate_list:
        epm_lib = epm_library(parent, library_size, rate = 3)
        #Return normalized fitnesses
        lib_fitness = [landscape.get_Energy(i) for i in epm_lib]
        p_fitness = lib_fitness[0]
        lib_fitness = lib_fitness/p_fitness

        df = pd.DataFrame({'Sequence' : epm_lib, 'Fitness' : lib_fitness})
        df = df.sort_values(by = 'Fitness', ascending = False)

        #Output Fitness Values to CSV File
        f = 'n(' + str(n) + ')k(' + str(K) + ')mut_rate(' + str(rate) + ').csv'
        df.to_csv(f)
        df.index = range(1,len(df) + 1)

        #Create NK Model graph
        plt.figure(figsize=(12,9))

        ax = plt.subplot(111)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        plt.xlim(0,library_size)

        plt.title('ROF Curve: N=' + str(n) +', K=' + str(K) + ', Mut-Rate=' + str(rate))
        plt.plot(range(1,len(df) + 1), df.Fitness)
        plt.savefig(f[:-3] + 'png', bbox_inches = 'tight')

        title = 'N=' + str(n) +', K=' + str(K) + ', Mut-Rate=' + str(rate)

        title_list.append(title)
        lib_fitness = sorted(lib_fitness, reverse = True)
        fitness_list.append(lib_fitness)
fitness_array = np.asarray(fitness_list).T
df2 = pd.DataFrame(fitness_array, columns = title_list)


plt.figure()
df2.plot(subplots=True, layout=(len(K_list), len(rate_list)), figsize=(20,15), sharex=True)
plt.savefig('eureka.png', bbox_inches = 'tight')
        # fig, axes = plt.subplots(nrows = len(K_list), ncols = len(rate_list))

# for i, K in K_list:
#     for j, mut_rate in rate_list:
#         df[title_list[i*len(rate_list) + j]].plt(ax=axes[

#     with open (file_name, 'wb') as f:
#         writer = csv.writer(f)
#         writer.writerows(lib_fitness)

# f.close()
# f2.close()
print('end')
print('runtime = ' + str(time.clock() - beg))
