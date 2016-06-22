###############
#Use version 2#
###############


#Zach Wu 2016
#Note: NK Testing does not account for stop codons or deletions

#Add stipulation for unique mutations?
#(ie: if one site is mutated already, do not mutate again)

import numpy as np
import sys
import random
import NK_Landscape
from scipy.stats import poisson

sample_parent = [random.randint(0,19) for i in range(10)]
num_AA = 20





import time
n = 10                #sequence length
library_size = 3*n

beg = time.clock()
test = NK_Landscape.NKLandscape(n,1, savespace = False, epi_dist = 'gamma')
interactions_list = test.nk_interactions()
epistatic_list = test.nk_epistatic()
end = time.clock()
print(-beg+end)

parent1 = [np.random.randint(num_AA) for i in range(n)]
print('Parent:' + str(parent1))
#library1 = epm(parent, library_size, rate=3)


current_seq = parent1
seq_list = []
print(seq_list)
for i in range(n):
    print('Current Seq:' + str(current_seq))
    print('Parent  Seq:' + str(parent1))

    temp = [current_seq
    #temp = mutate(current_seq, 1)
    for j in range(1):
        mut_index = random.randint(0,len(parent1)-1)
        curr_AA = current_seq[mut_index]
        poss_AA = set([j for j in range(num_AA)])
        poss_AA.remove(curr_AA)
        mut_AA = list(poss_AA)[random.randint(0,len(poss_AA)-1)]
        temp[mut_index] = mut_AA

    print('Current Seq:' + str(temp))
    print('Parent  Seq:' + str(parent1))
    print('--')
    current_seq = temp
    seq_list.append(temp)
        #print('Seq_List: ' + str(seq_list))
    print('--')
library2 = seq_list

print(seq_list)
