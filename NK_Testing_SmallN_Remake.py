#Zach Wu 2016

#To-Do
''' EPM update (Drummond)
'''

#Lines to modify for testing
'''37, 250,
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

n_list = [2,3]             #sequence length  NOTE: n = 5 does not run on Macbook Air (too much data)
K_list = [0,1,2,3]
library_sizes = [100,200]											#[100,200,300,400,500,600,1000,10000,100000]
num_landscapes_per = 1 		#10
n_min = n_list[0]
percent = 1
repeat_num = 5 				#20

#Below, still need to add to initial write_me's
library_type = 'random'     #can be 'random','EPM_Drummond','EPM_Poisson','single_mutant_library'
mut_rate = 2

all_data_pickle_filename = 'NK_Testing_SmallN_Data'  			#these filenames will be updated with _[date/time].p
landscapes_pickle_filename = 'NK_Testing_Landscapes' 			#when written

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

def epm_library_Poisson(parent, library_size, mut_rate = 1):
    '''parent sequence should be in [19,3,0,1] format
        Currently, assumes a poisson distribution of mutation rates:
            ****will be updated to match Sun/Drummond's ePCR model******
    '''
    dist_countL, num_mutL = EPM_Poisson_countd(mut_rate, library_size)

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
    seq_list.append(parent[:])

    for i in range(library_size-1):
        rand_seq = [random.randint(0,num_AA-1) for j in parent]
        seq_list.append(rand_seq[:])

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

def get_library(parent, library_size, library_style = 'random', mut_rate = 1):
    '''helper function for calling the correct library type
    '''

    'random','EPM_Drummond','EPM_Poisson','single_mutant_library'

    if library_style == 'random':
        return rand_library(parent, library_size)
    elif library_style == 'EPM_Drummond':
        return 0
    elif library_style == 'EPM_Poisson':
        return epm_library_Poisson(parent, library_size, mut_rate = mut_rate)
    elif library_style == 'single_mutant_library':
        return single_mutant_library
    else:
        print('Ya dun goofed, kid.')
        return 0

def write_to_all_files(file_list, write_me):
    ''' Writes string write_me to all files in file_list
    '''
    for file in file_list:
        file.write(write_me)

def get_time_string():
	'''returns string of current date and time in format amenable to filename
	'''
	current_time = datetime.datetime.now()
	time_string = current_time.isoformat()
	time_string = time_string.replace('-','_').replace(':','_').replace('.','_')
	time_string = time_string[:19]

	return time_string

################################################################
#     Create Files for Storing Correlations                    #
################################################################
start_time = get_time_string()

#files_list = ['NK_Model_Testing_Nsmall_SingleMutLib_f.txt', 'NK_Model_LOOCorrelations_Nsmall_SingleMutLib_f2.txt', 'NK_Model_TBCCorrelations_Nsmall_SingleMutLib_f3.txt', 'NK_Model_TBVCorrelations_Nsmall_SingleMutLib_f4.txt', 'NK_Model_TopFracCorrelations_Nsmall_SingleMutLib_f5.txt', 'NK_Model_TopPlateCorrelations_Nsmall_SingleMutLib_f6.txt']

f = open('NK_Model_Testing_Nsmall_SingleMutLib_f.txt', 'w')
f2 = open('NK_Model_LOOCorrelations_Nsmall_SingleMutLib_f2.txt','w')
f3 = open('NK_Model_TBCCorrelations_Nsmall_SingleMutLib_f3.txt','w')
f4 = open('NK_Model_TBVCorrelations_Nsmall_SingleMutLib_f4.txt','w')
f5 = open('NK_Model_TopFracCorrelations_Nsmall_SingleMutLib_f5.txt','w')
f6 = open('NK_Model_TopPlateCorrelations_Nsmall_SingleMutLib_f6.txt','w')
f_list = [f,f2,f3,f4,f5,f6]


#Write start time to files
write_me = 'Date Started: ' + start_time + '\n-----\n'
write_to_all_files(f_list, write_me)
#Write relevant parameters to file
write_me = '\nN List : ' + str(n_list) + '\nK_list : ' + str(K_list) + '\nLibrary Sizes : '+ str(library_sizes) + '\nNumber Landscapes : ' + str(num_landscapes_per) + '\nPercent : ' + str(percent) + '\nRepeat Number : ' + str(repeat_num) + '\nLibrary Style : ' + library_type + '\nMutation Rate (if applicable) :' + str(mut_rate)
write_to_all_files(f_list, write_me)

all_data_df = pd.DataFrame()
all_data_summary_df = pd.DataFrame()

################################################################
#     Import Relevant Regressors (clfs) and write to files     #
################################################################
from sklearn.linear_model import *
from sklearn.neighbors import *
from sklearn.neural_network import *
from sklearn.svm import *
from sklearn.tree import *
from sklearn.ensemble import *
from sklearn import cross_validation
from sklearn.kernel_ridge import KernelRidge

#clf_list = [ARDRegression(), BayesianRidge(), ElasticNet(), LassoLarsCV(), LinearRegression(), SGDRegressor(), KNeighborsRegressor(), LinearSVR(), DecisionTreeRegressor(), AdaBoostRegressor(), RandomForestRegressor(), GradientBoostingRegressor(), BaggingRegressor(), KernelRidge(), NuSVR()]
clf_list = [LinearRegression(), KNeighborsRegressor()]
for i in clf_list:
	write_me = ' '.join(str(i).replace('\n','').replace('\t','').split()) + ',\n'
	write_to_all_files(f_list, write_me)

################################################################
#     Creating Landscapes and Storing energies                 #
################################################################
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
                    f4.write('n = ' + str(n) + ' // K = ' + str(K) + ' // ' + ' // index = ' + str(j) + ' did not have enough in tbv. Choosing Top 5\n')

                landscape_data_list[n-n_min][K].append([landscape, total_space, top_by_count, top_by_value])

file_name = landscapes_pickle_filename + '_' + start_time + '.p'
pickle.dump(landscape_data_list, open(file_name, 'wb'))

print('Landscapes Done')
print('Time : ' + str(time.clock() - time_start))

################################################################
#     Testing Models for Various Landscapes                    #
################################################################
time_start = time.clock()

#Iterating over N values
for n in n_list:
	print('\n\n##############\n##############\n     n = ' + str(n) + '\n##############\n##############\n')

	#Iterating over K values
	for K in K_list:
		if K < n:
			print('\n\nK = ' + str(K))

			#Update Files with New N,K information
			write_me = '\n##########\nNew Fitness Landscape \nn = ' + str(n) + ' // K = ' + str(K) + '\n##########\n'
			write_to_all_files(f_list, write_me)

			#Fit to models and save predictions
			#iterating over clfs
			for clf in clf_list:
				clf_str = ' '.join(str(clf).replace('\n','').replace('\t','').split())
				print('Current clf:\n' + clf_str)

				#Save prediction and true lists
				write_me = '\nNew CLF: ' + clf_str + '\n---------\n'
				write_to_all_files(f_list, write_me)

				#Iterating over library sizes
				for library_size in library_sizes:
					write_me = '\nLibrary size = ' + str(library_size) + '\n\n'
					print(write_me)
					write_to_all_files(f_list, write_me)

					#Iterating over all landscapes made
					for landscape_data_list_index in tqdm(range(num_landscapes_per)):
					#Recall Landscape
						landscape, total_space, tbc, tbv = landscape_data_list[n-n_min][K][landscape_data_list_index]
						tbc_array, tbv_array = tbc.as_matrix(columns = ['Seq'])[:,0], tbv.as_matrix(columns = ['Seq'])[:,0]
						tbc_true_nrg, tbv_true_nrg = tbc.as_matrix(columns = ['Energy'])[:,0], tbv.as_matrix(columns = ['Energy'])[:,0]
						interactions_list = landscape.nk_interactions()
						epistatic_list = landscape.nk_epistatic()

						#Reinstatiate empty lists for storing pearson r lists
						LOO_r_list_f2 = []                                         #f2 storage update
						tbc_r_list_f3 = []
						tbv_r_list_f4 = []
						tbc_count_list_f5_1 = []
						tbv_count_list_f5_2 = []
						tbc100_count_list_f6 = []

						#Prerequisite number of repeats
						for j in range(repeat_num):
							#Generate New Parent and appropriate library, featurize
							parent = [np.random.randint(num_AA) for i in range(n)]
							temp_lib = get_library(parent, library_size, library_style = library_type, mut_rate = mut_rate)
							#Return the featurized library as a list
							lib_features = NK_Featurize(temp_lib)


							#Determine Fitnesses of each sequence
							lib_fitness_list = []
							for i in range(len(lib_features)):
								fitness = landscape.get_Energy(temp_lib[i])
								lib_fitness_list.append(fitness)

							predict_list = []
							true_list = []

							###########################
							#LOO Regression (f and f2)#
							###########################
							#f and f2
							for z in range(library_size):
								#Split Files
								index = z
								X_test, Y_test = [lib_features[index]], [lib_fitness_list[index]]
								X_train = lib_features[0:index] + lib_features[index+1:]
								Y_train = lib_fitness_list[0:index] + lib_fitness_list[index+1:]

								try:
									#Train Models
									clf.fit(X_train, Y_train)
									Y_predicted = clf.predict(X_test)
									predict_list.append(Y_predicted[0])
									true_list.append(Y_test[0])

									f.write(str(Y_predicted[0]) + ', ' + str(Y_test[0]) + '\n')

								except:
									print('Unexpected error (1): ', sys.exc_info()[0])
									raise

							LOO_r = np.corrcoef(predict_list, true_list)[0,1]
							LOO_r_list_f2.append(LOO_r)                              #store overall r for LOO regression
							f.write('Pearson r : ' + str(LOO_r) + '\n')       #write r to f file
							f2.write(str(LOO_r) + '\n')                              #write r to f2

							########################
							#Top Plate Work (f3-f6)#
							########################
							try:
								clf.fit(lib_features, lib_fitness_list)

								#########
								##f3/f4##
								#########
								#Determine Predictions
								Y_predict_tbc = clf.predict(NK_Featurize(tbc_array)) #tbc_array determined in landscape making
								Y_predict_tbv = clf.predict(NK_Featurize(tbv_array))
								#Calculate Correlations
								tbc_r = np.corrcoef(Y_predict_tbc, tbc_true_nrg)[0,1]
								tbv_r = np.corrcoef(Y_predict_tbv, tbv_true_nrg)[0,1]
								#Write to Files
								f3.write(str(tbc_r) + '\n')
								f4.write(str(tbv_r) + '\n')
								#Save to Lists
								tbc_r_list_f3.append(tbc_r)
								tbv_r_list_f4.append(tbv_r)

								#########
								##f5/f6##
								#########
								#Find predicted top matrices
								predict_nrgs = clf.predict(NK_Featurize(total_space))
								predict_lib_df = pd.DataFrame({'Seq' : total_space, 'Energy' : predict_nrgs})
								#Sort
								predict_lib = predict_lib_df.sort_values(by = 'Energy')
								top_count = math.floor(percent/100 * len(total_space))
								top_value_cutoff = predict_lib['Energy'].max() * (1 - percent / 100)

								tbc_pred = predict_lib.nlargest(top_count, 'Energy')
								tbv_pred = predict_lib[predict_lib.Energy > top_value_cutoff]
								tbc_pred_array, tbv_pred_array = tbc_pred.as_matrix(columns = ['Seq'])[:,0], tbv_pred.as_matrix(columns = ['Seq'])[:,0]

								#Take top 5 if there are not enough sequences, just like earlier in landscape making
								if len(tbv_pred_array) < 5:
									tbv_pred_array = tbc_pred_array[:5]

								#Count and record number of correctly found sequences
								tbc_count = 0
								tbv_count = 0
								tbc_100count = 0

								#Iterate through predictions for top-by-count
								for ind, seq in enumerate(tbc_pred_array):
									for tbc_real in tbc_array:
										if tbc_real == seq:
											tbc_count += 1
											if ind < 100:
												tbc_100count += 1

								#Iterate through predictions for top-by-value
								for ind, seq in enumerate(tbv_pred_array):
									for tbv_real in tbv_array:
										if tbv_real == seq:
											tbv_count += 1

								#Write to files
								tbc_size, tbv_size = len(tbc_pred_array), len(tbv_pred_array)
								f5.write(str(tbc_count) + ',' + str(tbc_size) + ',' + str(tbv_count) + ',' + str(tbv_size) + ',\n')
								f6.write(str(tbc_100count) + '\n')

								#Save to lists
								tbc_count_list_f5_1.append([tbc_count, tbc_size])
								tbv_count_list_f5_2.append([tbv_count, tbv_size])

								tbc100_count_list_f6.append(tbc_100count)
							except:
								print('Unexpected error (2): ', sys.exc_info()[0])
								raise


					all_data_df = all_data_df.append({ 'N' : n, 'K' : K, 'CLF' : clf_str, 'Library_Size' : library_size, '5Landscape_Index' : landscape_data_list_index, 'LOO_R' : LOO_r_list_f2, 'TBC_R' : tbc_r_list_f3, 'TBV_R' : tbv_r_list_f4, 'TBC_Count' : tbc_count_list_f5_1, 'TBV_Count' : tbv_count_list_f5_2, 'Top_100_Count' : tbc100_count_list_f6}, ignore_index = True)

					#Determine values for summary:
					LOO_r_avg, LOO_r_std = np.average(LOO_r_list_f2), np.std(LOO_r_list_f2)
					TBC_r_avg, TBC_r_std = np.average(tbc_r_list_f3), np.std(tbc_r_list_f3)
					TBV_r_avg, TBV_r_std = np.average(tbv_r_list_f4), np.std(tbv_r_list_f4)

					TBC_counts = np.asarray(tbc_count_list_f5_1)[:,0]
					TBC_count_avg, TBC_count_std = np.average(TBC_counts), np.std(TBC_counts)
					TBC_count_total = np.average(np.asarray(tbc_count_list_f5_1)[:,1])
					TBV_counts =  np.asarray(tbv_count_list_f5_2)[:,0]
					TBV_count_avg, TBV_count_std = np.average(TBV_counts), np.std(TBV_counts)
					TBV_count_total = np.average(np.asarray(tbv_count_list_f5_2)[:,1])

					top_100_count_avg, top_100_count_std = np.average(tbc100_count_list_f6), np.std(tbc100_count_list_f6)

					#Add to summary dataframe
					all_data_summary_df = all_data_summary_df.append({ '1:N' : n, '2:K' : K, '3:CLF' : clf_str, '4:Library_Size' : library_size, '5:Landscape_Index' : landscape_data_list_index, 'LOO_r_avg' : LOO_r_avg, 'LOO_r_std' : LOO_r_std, 'TBC_r_avg' : TBC_r_avg, 'TBC_r_std' : TBC_r_std, 'TBV_r_avg' : TBV_r_avg, 'TBV_r_std' : TBV_r_std, 'TBC_count_avg':TBC_count_avg, 'TBC_count_std' : TBC_count_std, 'TBC_count_total': TBC_count_total, 'TBV_count_avg': TBV_count_avg, 'TBV_count_std': TBV_count_std, 'TBV_count_total': TBV_count_total, 'top_100_count_avg': top_100_count_avg, 'top_100_count_std': top_100_count_std}, ignore_index = True)


################################################################
#     Closing Shop                                             #
################################################################

#Pickle Final Dataframe containing all information
file_name = all_data_pickle_filename + '_' + start_time + '.p'
pickle.dump(all_data_df, open(file_name, 'wb'))
#Also output summary dataframe as excel sheet
file_name = all_data_pickle_filename + '_' + start_time + '_summary.xlsx'
all_data_summary_df.to_excel(file_name)
file_name = all_data_pickle_filename + '_' + start_time + '_summary.p'
pickle.dump(all_data_summary_df, open(file_name, 'wb'))

#Write End Time to every file
end_time = get_time_string()
write_me = '\n--------\nDate Ended: ' + end_time + '\n'
print(write_me)
write_to_all_files(f_list, write_me)
#Close all files
for file in f_list:
    file.close()
print('Done')
