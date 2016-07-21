#Zach Wu 2016

#To-Do
''' EPM update (Drummond)

'''

#Notes:
'''     - until featurized by NK_Featurize, all sequences are of format [19,3,0,1]
'''

import numpy as np
import sys
import random
import NK_Landscape
from scipy.stats import poisson
from scipy.stats import pearsonr
import warnings
import time
from tqdm import tqdm
import math
import pandas as pd
import pickle
import datetime

import warnings
warnings.filterwarnings("ignore")

################################################################
#     Relevant Parameters                                      #
################################################################

num_AA = 20

n_list = [2,3,4]             #sequence length  NOTE: n = 5 does not run on Macbook Air (too much data)
K_list = [0,1,2,3]
library_sizes = [100,200,300,400,500,600,1000,10000,100000]
num_landscapes_per = 10
n_min = n_list[0]
percent = 1
repeat_num = 20
library_style = 'EPM_Poisson'     #can be 'random','EPM_Drummond','EPM_Poisson','single_mutant_library'


################################################################
#     Helper Functions (move?)                                 #
################################################################
''' These helper functions should probably be moved to NK_Landscape, eventually
'''
def EPM_Poisson_countd(mu, library_size):
    '''Returns the Poisson mutation rate distribution for a given library size

    Average rate is set by mu, library size is the number of sequnces in the library
    Returns two lists, probs_list contains the number of sequences with the corresponding number of mutations in mut_list
    '''

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

    #If, due to rounding, the total library size is greater than expected
    #Subtract the difference from the mean (mu)

    dif = sum(probs_list) - library_size
    mutation_list = [i for i in range(a,b+1)]
    index = mutation_list.index(mu)
    probs_list[index] -= dif

    return probs_list, mut_list

def NK_Featurize(seqList):
    ''' Given a sequence list of format [19,0,4,1], returns the binary/ProSAR featurization
        where each amino acid at each position is its own dimension
        (For the NK model, no other featurization makes sense.)

        Returns a list of featurized_sequences
    '''
    featurized_seq = []
    for i, seq in enumerate(seqList):
        temp_seq = [0] * len(seq) * num_AA
        for j, AA in enumerate(seq):
            temp_seq[j*20 + AA] = 1
        featurized_seq.append(temp_seq)
    return featurized_seq

def epm_library_Poisson(parent, library_size, rate = 1):
    '''parent sequence should be in [19,3,0,1] format
        Currently, assumes a poisson distribution of mutation rates:
            ****will be updated to match Sun/Drummond's ePCR model******
    '''
    dist_countL, num_mutL = EPM_Poisson_countd(rate, library_size)

    seq_list = []
    seq_list.append(parent[:])
    for i, num_mut in enumerate(num_mutL):
        for j in range(dist_countL[i]):
            seq_list.append(mutate(parent[:], num_mut)[:])

    return seq_list

def rand_library(parent, library_size):
    '''Given a parent sequence of format [19,3,0,1] format, returns a completely random sequence list
        - includes parent
    '''
    seq_list = []
    seq_list.append(parent)

    for i in range(library_size-1):
        rand_seq = [random.randint(0,num_AA-1) for j in i]
        seq_list.append(rand_seq)

    return seq_list

def single_mutant_library(parent, library_size):
    ''' Returns a random walk of single mutants, including parent
    '''
    current_seq = list(parent)[:]
    seq_list = []
    seq_list.append(parent[:])
    for i in range(library_size):
        temp = current_seq[:]
        temp = mutate(current_seq, 1)

        current_seq = temp[:]
        seq_list.append(temp)
    return seq_list

def full_library(n):
    '''returns the full library for a protein sequence of length n
    '''
    lib_size = num_AA ** n
    base_seq = [0 for i in range(n)]

    library = []

    for i in range(lib_size):
        seq = base_seq[:]
        id = i
        for j in reversed(range(n)):
            seq[j] = math.floor(id / (num_AA**(j)))
            id = id%(num_AA**(j))
        library.append(seq)

    return library

def top_partition_library(landscape, full_library, percent = 10):
    ''' returns the top percent of a landscape as a panda dataframe
        note that the top percent refers to both top percent by count(fraction) and by value(from fitness)
    '''
    energies = [landscape.get_Energy(i) for i in total_space]

    seq_nrg_df = pd.DataFrame({'Seq' : total_space, 'Energy' : energies})
    seq_nrg = seq_nrg_df.sort_values(by = 'Energy')

    top_count = math.floor(percent/100*len(total_space))
    top_value_cutoff = seq_nrg_df['Energy'].max()*(1 - percent/100)

    tbc = seq_nrg.nlargest(top_count, 'Energy')             #returns top section by count (fraction)
    tbv = seq_nrg[seq_nrg.Energy > top_value_cutoff]          #returns top section by value

    return tbc, tbv

def write_to_all_files(file_list, write_me):
    ''' Writes string write_me to all files in file_list
    '''
    for file in file_list:
        file.write(write_me)

library_sizes = [100,200,300,400,500,600,1000,10000,100000]

mut_rate = 2

for library_size in library_sizes:
    a,b = EPM_Poisson_countd(mut_rate, library_size)
    print('Library Size \n')
    print(a)
    print(b)
