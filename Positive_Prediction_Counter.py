
import numpy as np
import matplotlib as plt


data = []


with open('positive_predictions.csv') as f:
    data = f.readlines()

data = np.array([[int(i.strip('\n')) for i in line.split(',')] for line in data])



complete_AA_list = ['R','H','K','D','E','S','T','N','Q','C','G','P','A','I', 'L','M','F',\
                    'W','Y','V','*']

comp_AA_dic = dict((j,i) for i,j in enumerate(complete_AA_list))
comp_AA_dic_rev = dict((i,j) for i,j in enumerate(complete_AA_list))

totals = np.zeros([4,21])
for i in range(4):

    for j in range(21):

        for dat in data:
            if dat[i] == j:
                totals[i,j] += 1

print('Total points = ' + str(len(data)))

np.savetxt("positive_counts.csv", totals.T, delimiter=",")