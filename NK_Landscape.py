#To-Do:
    #Check energy function (referencing correct indices)
    #Add savespace? (incorrect assumption, but... saves space)

#Zach Wu 2016
#Note: NK Landscape does not account for stop codons or deletions

import numpy as np
import sys
import random
from scipy.stats import poisson

class NKLandscape():
    def __init__(self, n, k, bidirect = False, savespace = False, num_AA = 20, epi_dist = 'even', epi_type = 'add'):
        '''
        '''

        self.N = n                              #number of sites
        self.K = k
        self.bidirect = bidirect
        self.save_space = savespace
        self.epi_dist = epi_dist
        self.epi_type = epi_type

        self.interactions = self.nk_interactions()
        self.epi = self.nk_epistatic()


    def nk_interactions(self):

        interaction_list = np.zeros((self.N,self.K)) - 1
        if self.bidirect == False:
            for i in range(self.N):
                possible_interactions = set([j for j in range(self.N)])
                possible_interactions.remove(i)

                for j in range(self.K):
                    rand_index = random.randint(0,len(possible_interactions)-1)
                    rand_site = list(possible_interactions)[rand_index]
                    possible_interactions.remove(rand_site)
                    interaction_list[i,j] = int(rand_site)

        else:
            interaction_list = []
            print('Has yet to be written, and may not need to be')
            '''Basically, make all interactions, order, choose highest 3
            '''
        return interaction_list

    def nk_epistatic(self):
        if self.save_space == False:
            if self.K > 3:
                epi = []
                print('Please select a smaller k value')
            else:
                if self.epi_dist == 'even':
                    epi = np.random.randn(self.N, 20**(self.K+1))
                elif self.epi_dist == 'gamma':
                    epi = np.random.gamma(0.35,0.35,(self.N, 20**(self.K+1)))
        else:
            if self.epi_dist == 'even':
                epi = np.random.randn(self.N, 20*(self.K+1))
            elif self.epi_dist == 'gamma':
                epi = np.random.gamma(0.35,0.35,(self.N, 20*(self.K+1)))

        return epi

    def get_Energy(self, sequence):
        '''sequence should be in [19,3,0,1] format
            *** Divide by length(sequence)?
                - equivalent to centering self.epi around 0
        '''
        if len(sequence) != self.N:
            print('Sequence length is incorrect.')

        else:
            if self.save_space == False:

                if self.epi_type == 'add':
                    energy = 0
                    for pos, aa in enumerate(sequence):
                        index = aa
                        for i, interaction in enumerate(self.interactions[pos]):
                            index += 20**(i+1)*(sequence[int(interaction)])
                        energy += self.epi[pos][index]
    #                     print(str(pos) + ',' + str(index))
    #                 print(energy)
                    return energy

                elif self.epi_type == 'multiply':
                    total_energy = 0
                    for pos, aa in enumerate(sequence):
                        index = aa
                        energy = 0
                        for i, interaction, in enumerate(self.interactions[pos]):
                            if i == 0:
                                energy = self.epi([pos][aa])
                            index += 20**(i+1)*(sequence[int(interaction)])
                        energy *= self.epi[pos][index]
                        total_energy += energy
                    return total_energy

            elif self.save_space == True:

                if self.epi_type == 'add':
                    energy = 0
                    for pos, aa in enumerate(sequence):
                        interaction_list = self.interactions[pos]
                        energy = self.epi(aa)
                        for ind,inter in enumerate(interaction_list):
                            energy += self.epi[20*ind + inter]
                    return energy

                elif self.epi_type == 'multiply':
                    total_energy = 0
                    for pos, aa in enumerate(sequence):
                        interaction_list = self.interactions[pos]
                        energy = self.epi(aa)
                        for ind,inter in enumerate(interaction_list):
                            energy *= self.epi[20*ind + inter]
                        total_energy += energy
                    return total_energy

