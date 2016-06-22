

import numpy as np
import SequenceTools2
import numpy as np
from Bio.Seq import Seq
from Bio.Alphabet import IUPAC
import glob
import math
import pandas as pd

complete_AA_list = ['R','H','K','D','E','S','T','N','Q','C','G','P','A','I', 'L','M','F',\
                    'W','Y','V','*']

comp_AA_dic = dict((j,i) for i,j in enumerate(complete_AA_list))
comp_AA_dic_rev = dict((i,j) for i,j in enumerate(complete_AA_list))

file_name = 'Ost_TEM_CSV.csv'

ost_pd = pd.read_csv(file_name)
ost_pd_red = ost_pd.dropna(how='any')

WT_Seq = []

for i in range(int(len(ost_pd)/20)):
    WT_Seq.append(ost_pd.at[20*i,'WT AA'])

WT_String = ''.join(WT_Seq)
print(WT_Seq)
print(WT_String)

sequences = []
fitnesses = []
fitness_errors = []


for i in range(len(ost_pd_red)-5400):                     #######
    temp_Seq = WT_Seq
    mut_Pos = ost_pd_red.iat[i,0]
    print(mut_Pos)
    temp_Seq[mut_Pos-1] = ost_pd_red.iat[i,3]

    sequences.append(''.join(temp_Seq))
    fitnesses.append(ost_pd_red.iat[i,4])
    fitness_errors.append(ost_pd_red.iat[i,5])

print(sequences)
print(fitnesses)
print(fitness_errors)

mutated_indices = [int(i) for i in range(len(WT_Seq))]

#sequences_numbers = SequenceTools2.mutList_NDT(sequences, mutated_indices)
featureL = SequenceTools2.AAFeaturize(sequences, mutated_indices)
print(featureL[0])
print(mutated_indices)
