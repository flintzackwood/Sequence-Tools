#Zach Wu 2016
#Note: NK Testing does not account for stop codons or deletions

#Adding new models from scikit learn

#Add stipulation for unique mutations?
#(ie: if one site is mutated already, do not mutate again)

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

def rand_library(parent, library_size):
    '''parent is included
    '''
    seq_list = []
    seq_list.append(parent)

    for i in range(library_size-1):
        rand_seq = [random.randint(0,num_AA-1) for j in i]
        seq_list.append(rand_seq)

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


###############################################################
###############################################################
###############################################################

#Create Libraries and store energies.

import time
n_list = [2,3,4,5]             #sequence length  NOTE: n = 5 takes 90 seconds to make, took out 4,5 for testing
K_list = [0,1,2,3]
library_sizes = [100*(i+1) for i in range(10)]
num_landscapes_per = 10
n_min = n_list[0]
percent = 10
repeat_num = 20

f = open('NK_Model_Testing_Nsmall_SingleMutLib10Percent_f.txt', 'w')
f2 = open('NK_Model_LOOCorrelations_Nsmall_SingleMutLib10Percent_f2.txt','w')
f3 = open('NK_Model_TBCCorrelations_Nsmall_SingleMutLib10Percent_f3.txt','w')
f4 = open('NK_Model_TBVCorrelations_Nsmall_SingleMutLib10Percent_f4.txt','w')
f5 = open('NK_Model_TopFracCorrelations_Nsmall_SingleMutLib10Percent_f5.txt','w')
f6 = open('NK_Model_TopPlateCorrelations_Nsmall_SingleMutLib10Percent_f6.txt','w')

f4_black_list = []

landscape_data_list = [[[] for k in range(len(K_list))] for n in range(len(n_list))]
print('Making Landscapes')
time_start = time.clock()
for n in n_list:
    print('n = ' + str(n))
    for K in K_list:
        if K < n:
            print('K = ' + str(K))
            for j in tqdm(range(num_landscapes_per)):
                landscape = NK_Landscape.NKLandscape(n,K, savespace = False, epi_dist = 'gamma')
                total_space = full_library(n)
                top_by_count, top_by_value = top_partition_library(landscape, total_space ,percent = percent)
                if len(top_by_value) < 5:
                            top_by_value =  top_by_count.head(5)
                            f4.write('n = ' + str(n) + ' // K = ' + str(K) + ' // ' + ' // index = ' + str(j) + ' did not have enough in tbv.\n')
                            f4_black_list.append([n,K,j])

                landscape_data_list[n-n_min][K].append([landscape, total_space, top_by_count, top_by_value])
print('Landscapes Done')
print('Time : ' + str(time.clock() - time_start))


###############################################################
###############################################################
###############################################################
from sklearn.linear_model import *
from sklearn.neighbors import *
from sklearn.neural_network import *
from sklearn.svm import *
from sklearn.tree import *
from sklearn.ensemble import *
from sklearn import cross_validation
#from sklearn.cross_decomposition import PLSRegression ##New One
from sklearn.kernel_ridge import KernelRidge
###############################################################
#clf_list2 = [ARDRegression(), BayesianRidge(), ElasticNet(), LassoLarsCV(), LinearRegression(), SGDRegressor(), KNeighborsRegressor(), LinearSVR(), DecisionTreeRegressor(), AdaBoostRegressor(), RandomForestRegressor(), GradientBoostingRegressor(), BaggingRegressor(), KernelRidge(), NuSVR()]
clf_list = [LinearRegression(), KNeighborsRegressor()]
# n = 40                    #Sample length = 40
# library_size_list = [i*80 for i in range(10)]
# library_size = 3*n        #Library size = 3*n





f.write('Single Mutant Library, 200 Iterations')
f2.write('Single Mutant Library, 200 Iterations')

# f.write('N = ' + str(n))
# f.write('\nLibrary Size = ' + str(library_size) + '\n--------\n')
# f2.write('N = ' + str(n))
# f2.write('\nLibrary Size = ' + str(library_size) + '\n--------\n')


predict_master_list = []
true_master_list = []


#Write CLF list to file
for i in clf_list:
    print(str(i) + '\n')
    f2.write(str(i) + ', \n')
time_start = time.clock()

#Go through different K Values
#REMAKE landscapes, error prone libraries, and corresponding fitnesses
#Train base/default models

for n in n_list:
    print('\n\n##############\n##############\n     n = ' + str(n) + '\n##############\n##############\n')

    for K in K_list:
        if K < n :
            print('\n\nK = ' + str(K))
            predict_list_by_clf = []
            true_list_by_clf = []

            new_NK_write = '\n##########\nNew Fitness Landscape \nn = ' + str(n) + ' // K = ' + str(K) + '\n##########\n'

            f.write(new_NK_write)
            f2.write(new_NK_write)
            f3.write(new_NK_write)
            f4.write(new_NK_write)
            f5.write(new_NK_write)
            f6.write(new_NK_write)
            #f2.write('\n\n------\nn Value = ' + str(n) + '// K Value = ' + str(K) + '\n------\n')

            #Fit to models and save fitnesses:
            for i, clf in enumerate(clf_list): #!!!!!
                print('Current clf:\n' + str(clf))
                #Save prediction and true lists

                write_me = '\nNew CLF: ' + str(clf) + '\n---------\n'
                f.write(write_me)
                f2.write(write_me)
                f3.write(write_me)
                f4.write(write_me)
                f5.write(write_me)
                f6.write(write_me)

                print('\nProgress Bar: ')

                for library_size in library_sizes:
                    write_me = '\nLibrary size = ' + str(library_size) + '\n\n'
                    print(write_me)
                    f.write(write_me)
                    f2.write(write_me)
                    f3.write(write_me)
                    f4.write(write_me)
                    f5.write(write_me)
                    f6.write(write_me)

                    t5_2, t3, t4, t5, t6 = [[] for x in range(5)]
                    u5_2, u3, u4, u5, u6 = [[] for x in range(5)]

                    #WRITE TO FILES

                    predict_list = []
                    true_list = []

                    for landscape_data_list_index in tqdm(range(num_landscapes_per)):
                        blacklisted = False
                        for i in f4_black_list:
                            if i == [n, K, landscape_data_list_index]:
                                blacklist = True
                        #Call NK_Landscape          -- N/K are currently hardcoded.
                        landscape, total_space, tbc, tbv = landscape_data_list[n-n_min][K][landscape_data_list_index]
                        tbc_array, tbv_array = tbc.as_matrix(columns = ['Seq'])[:,0], tbv.as_matrix(columns = ['Seq'])[:,0]
                        tbc_true_nrg, tbv_true_nrg = tbc.as_matrix(columns = ['Energy'])[:,0], tbv.as_matrix(columns = ['Energy'])[:,0]
                        interactions_list = landscape.nk_interactions()
                        epistatic_list = landscape.nk_epistatic()


                        for j in (range(repeat_num)):                   #Old tqdm spot (for progress bar)
                            #Specify parent sequence and library
                            parent = [np.random.randint(num_AA) for i in range(n)]
                            epm_lib = single_mutant_library(parent, library_size)
                            #epm_lib = epm_library(parent, library_size, rate = 3)  # Single Mutants!!
                            #Featurize
                            lib_features = NK_Featurize(epm_lib)


                            #Determine Fitnesses
                            lib_fitnessL = []
                            for i in range(len(lib_features)):
                                fitness = landscape.get_Energy(epm_lib[i])
                                lib_fitnessL.append(fitness)

                            #rand_ind = random.randint(0, len(epm_lib)-1)



                            #1. Correlations for LOO Classification
                            for z in range(library_size):
                                rand_ind = z
                                #LOO Classification
                                X_test, Y_test = [lib_features[rand_ind]], [lib_fitnessL[rand_ind]]
                                X_train = lib_features[0:rand_ind] + lib_features[rand_ind+1:]
                                Y_train = lib_fitnessL[0:rand_ind] + lib_fitnessL[rand_ind+1:]

                                try:
                                    clf.fit(X_train, Y_train)
                                    Y_predicted = clf.predict(X_test)
                                    predict_list.append(Y_predicted[0])
                                    true_list.append(Y_test[0])

                                    f.write(str(Y_predicted[0]) + ', ' + str(Y_test[0]) + '\n')

                                except np.linalg.linalg.LinAlgError as err:
                                    print(clf)
                                    print('failed')


                            try:
                                clf.fit(lib_features, lib_fitnessL)
                                Y_predict_tbc = clf.predict(NK_Featurize(tbc_array))      #Check tbc!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                                Y_predict_tbv = clf.predict(NK_Featurize(tbv_array))
                                r = np.corrcoef(Y_predict_tbc, tbc_true_nrg)
                                f3.write(str(r[0,1]**2) + '\n')
                                t3.append(r[0,1]**2)
                                r = np.corrcoef(Y_predict_tbv, tbv_true_nrg)
                                f4.write(str(r[0,1]**2) + '\n')
                                t4.append(r[0,1]**2)


                                #Make needed matrices
                                predict_nrgs = clf.predict(NK_Featurize(total_space))
                                predict_lib_df = pd.DataFrame({'Seq' : total_space, 'Energy' : predict_nrgs})

                                predict_lib = predict_lib_df.sort_values(by = 'Energy')
                                top_count = math.floor(percent/100*len(total_space))
                                top_value_cutoff = predict_lib['Energy'].max()*(1 - percent/100)

                                tbc_pred = predict_lib.nlargest(top_count, 'Energy')               #returns top section by count (fraction)
                                tbv_pred = predict_lib[predict_lib.Energy > top_value_cutoff]          #returns top section by value
                                tbc_pred_array, tbv_pred_array = tbc_pred.as_matrix(columns = ['Seq'])[:,0], tbv_pred.as_matrix(columns = ['Seq'])[:,0]
                                if len(tbv_pred_array) < 5:
                                    tbv_pred_array = tbc_pred_array[:5]
                                tbc_pred100 = tbc_pred_array[:100]

                                #Count and Record
                                tbc_count = 0
                                tbv_count = 0
                                tbc_100count = 0
                                for ind, seq in enumerate(tbc_pred_array):
                                    for tbc_real in tbc_array:
                                        if tbc_real == seq:
                                            tbc_count += 1
                                            if ind < 100:
                                                tbc_100count += 1
                                            break

                                for ind, seq in enumerate(tbv_pred_array):
                                    for tbv_real in tbv_array:
                                        if tbv_real == seq:
                                            tbv_count += 1

                                f5.write(str(tbc_count) + ',' + str(len(tbc_pred_array)) + ',' + str(len(tbc_array)) + ',' + str(tbv_count) + ',' + str(len(tbv_pred_array)) + ',' + str(len(tbv_array)) + ',\n')
                                t5.append(tbc_count / len(tbc_pred_array))
                                t5_2.append(tbv_count/len(tbv_pred_array))
                                f6.write(str(tbc_100count) + '\n')
                                t6.append(tbc_100count / 100)



                            except np.linalg.linalg.LinAlgError as err:
                                print(clf)
                                print('failed')




                        #Determine Pearson's R

                    #1. LOO Classification
                    r = np.corrcoef(predict_list, true_list)
                    write_me = 'Rsquared : ' + str(r[0,1]**2) + '\n'
                    #print(write_me)
                    f.write(write_me)
                    f2.write(str(r[0,1]**2) + '\n')
                    f3.write('\n-----\n' + str(np.average(t3)) + ',' + str(np.std(t3)) + '\n\n')
                    f4.write('\n-----\n' + str(np.average(t4)) + ',' + str(np.std(t4)) + '\n\n')
                    f5.write('\n-----\n' + str(np.average(t5)) + ',' + str(np.std(t5)) + ',' + str(np.average(t5_2)) + ',' + str(np.std(t5_2)) + '\n\n')
                    f6.write('\n-----\n' + str(np.average(t6)) + ',' + str(np.std(t6)) + '\n\n')

                    predict_list_by_clf.append(predict_list)
                    true_list_by_clf.append(true_list)


            predict_master_list.append(predict_list)
            true_master_list.append(true_list_by_clf)
f.close()
f2.close()
f3.close()
f4.close()
f5.close()
f6.close()
print('end')
print('runtime = ' + str(time.clock() - time_start))
