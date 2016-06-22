complete_AA_list = ['R','H','K','D','E','S','T','N','Q','C','G','P','A','I', 'L','M','F',\
                    'W','Y','V','*']

comp_AA_dic = dict((j,i) for i,j in enumerate(complete_AA_list))
comp_AA_dic_rev = dict((i,j) for i,j in enumerate(complete_AA_list))


print('start')
import SequenceTools2
import numpy as np
from Bio.Seq import Seq
from Bio.Alphabet import IUPAC
import glob
import math

ref = 'CTGGCGAAACAGAAGGGTTGCATGGCTTGCCACGACCTGAAAGCCAAGAAAGTTGGCCCTGCATACGCAGATGTAGCCAAAAAATACGCAGGCCGCAAAGACGCAGTAGATTACCTGGCGGGAAAAATCAAAAAGGGCGGCTCTGGTGTTTGGGGTAGTGTGCCAATGCCACCGCAAAATGTCACCGATGCCGAAGCGAAACAACTGGCACAATGGATTTTATCAATCAAActcgagcaccaccatcaccaccactga'

#4x_NDT2 Plate:
#Note: this section is much longer because it combines sequencing data from both sequencing runs
#Note: if we ever need to combine 3 sequencing runs for the same plate, I will be mildly upset.

#Finding Mutated Indices
file_dir1 = '/Users/ZacharyWu1/Documents/SequenceTools/data/Hth_NDT2_First/*.seq'
#print(list_seq_files)
bpseqL,bpseqStrL,readLengthL,AASeqL,mutIndL, alteredBPs, fileL = SequenceTools2.seqList_from_file(file_dir1, ref = ref)
file_dir2 = '/Users/ZacharyWu1/Documents/SequenceTools/data/Hth_NDT2_Reseq/*.seq'
bpseqL2,bpseqStrL2,readLengthL2,AASeqL2,mutIndL2, alteredBPs2, fileL2 = SequenceTools2.seqList_from_file(file_dir2, ref = ref)

#Using Mutated Indices
mut_indices2 = [41,42,55,58]
bpseqL,bpseqStrL,readLengthL,AASeqL,mutIndL, alteredBPs, fileL = SequenceTools2.seqList_from_file(file_dir1, ref = ref, cutoff = 58, mut_Ind = mut_indices2)
bpseqL2,bpseqStrL2,readLengthL2,AASeqL2,mutIndL2, alteredBPs2, fileL2 = SequenceTools2.seqList_from_file(file_dir2, ref = ref, cutoff = 58, mut_Ind = mut_indices2)

#Combine Two Sequencing Runs
combined_wellID, combined_AASeqL = SequenceTools2.combineSeqData(fileL,AASeqL,fileL2,AASeqL2)
#Generate Mutation List
mutList = SequenceTools2.mutList_NDT(combined_AASeqL, mut_indices2)
#AAFeaturize
sample_features_P2 = SequenceTools2.AAFeaturize(mutList, mut_indices2)
#Read in HPLC Data from Plate 2
fname = 'data/2nd Plate Data (for Import) Averaged.csv'
regr_dat,class_dat = SequenceTools2.getScreeningData(fname)


#Match sequences to HPLC data by well
sample_seqm_P2, regression_dm_P2, class_dm_P2, well_id_list_P2 = SequenceTools2.matchData2WellID(combined_wellID, combined_AASeqL,regr_dat,class_dat)

mutList_P2 = SequenceTools2.mutList_NDT(sample_seqm_P2, mut_indices2)

print('end1')


# for i in range(len(sample_seqm_P2)):
#     for j in range(4):
#         tRange = range(j*21,(j+1)*21+1)
#         section = sample_featuresP2[i][j*21:(j+1)*21+1]
#         print(comp_AA_dic_rev[section.index(1)])
#     print(str(well_id_list_P2[i]) + ' ' + str(sample_seqm_P2[i]) + ' \t' + str(regression_dm_P2[i]) + ',' + str(class_dm_P2[i]) + '\n --')


# test = [''.join(str(i)) for i in well_id_list_P2]

# test1 = []
# for i in test:
#     print(i)
#     if i in test1:
#         print('doubled')
#     test1.append(i)
# print(len(set(test)))

#4x_NDT1 Plate:

#Get sequences from files
file_dir = '/Users/ZacharyWu1/Documents/SequenceTools/data/Hth_NDT1_Edited/*.seq'
bpseqL,bpseqStrL,readLengthL,AASeqL,mutIndL, alteredBPs, fileL = SequenceTools2.seqList_from_file(file_dir, ref = ref)
# for i in AASeqL:
#     print(str(i))

#Mutated residues are: 43,44,57,60
mut_indices1 = [43,44,57,60]
#Rerun with cutoff
bpseqL,bpseqStrL,readLengthL,AASeqL_P1,mutIndL, alteredBPs, fileL_P1 = SequenceTools2.seqList_from_file(file_dir, ref = ref, cutoff = 60, mut_Ind = mut_indices1)
#Generate Mutation List
mutList_P1 = SequenceTools2.mutList_NDT(AASeqL_P1,mut_indices1)
#Feature Representation
sample_features_P1 = SequenceTools2.AAFeaturize(mutList_P1,mut_indices1) #Manual connect?
#Screening Data
fname = 'data/1st Plate Data (for Import) Averaged.csv'
regr_dat, class_dat = SequenceTools2.getScreeningData(fname)


#Match sequence to screening data
sample_seqm_P1, regression_dm_P1, class_dm_P1, well_id_list_P1 = SequenceTools2.matchData2SeqFile(file_dir, [-10,-7], AASeqL_P1, regr_dat, class_dat)
print('end2')

#Now Machine Learning can actually happen.

#Combine 1st and 2nd plates

sample_seqm_features1 = SequenceTools2.AAFeaturize(mutList_P1, mut_indices1)
sample_seqm_features2 = SequenceTools2.AAFeaturize(mutList_P2, mut_indices2)
sample_seqm = sample_seqm_features1 + sample_seqm_features2

mutList = mutList_P1 + mutList_P2

regression_dm = regression_dm_P1 + regression_dm_P2
class_dm = class_dm_P1 + class_dm_P2

temp_wellIDs1 = [i + [1] for i in well_id_list_P1]
temp_wellIDs2 = [i + [2] for i in well_id_list_P2]
well_id_list = temp_wellIDs1 + temp_wellIDs2


#Import sklearn packages
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn import cross_validation
from sklearn.svm import LinearSVC
from sklearn.ensemble import *
from sklearn.linear_model import *
from sklearn.neighbors import *
from sklearn.svm import *
from sklearn.tree import *
from sklearn.naive_bayes import GaussianNB


#Make a list of models: reorganize accordingly
#hi = OneVsRestClassifier(LinearSVC(random_state = 0)).fit(X_train, Y_train).predict(X_test)

#for model in model list
incorrect = 0
count1 = 0

##################################
#Forming Model List for GridSearch
clf_list = [AdaBoostClassifier(), ExtraTreesClassifier(), GradientBoostingClassifier(), RandomForestClassifier(), LogisticRegression(), SGDClassifier(loss = 'log'), SGDClassifier(loss = 'modified_huber'), KNeighborsClassifier(), SVC(), LinearSVC(), DecisionTreeClassifier(), GaussianNB()]
base_clf_list = [RandomForestClassifier(), KNeighborsClassifier(), SVC()]
honed_clf_list = [RandomForestClassifier(criterion = 'gini', max_depth = 15,n_estimators = 50), SVC(C=2, degree = 2, kernel = 'poly'), KNeighborsClassifier(n_neighbors = 7, leaf_size = 45, algorithm = 'auto')]

##################################


f = open('post_gridsearch_predictions.txt','w')

import time
b1 = time.clock()
trained_clf_list = []
for clf in honed_clf_list:
    f.write(str(clf) + '\n')
    trained_clf_list.append(OneVsRestClassifier(clf).fit(sample_seqm, class_dm))

b2 = time.clock()
print('Training time : ' + str(math.floor((b2-b1)/60)) + ' minutes.')

sequence_space = []
for a in range(21):
    for b in range(21):
        for c in range(21):
            for d in range(21):
                sequence_space.append([a,b,c,d])

#sequence_space = sequence_space[:10] ##!!!

def quick_featurize(seq):
    temp_seq = [0 for i in range(len(seq)*21)]
    for i, aa in enumerate(seq):
        temp_seq[21*i + aa] = 1
    return temp_seq

for seq in sequence_space:
    #Need to featurize
    seq_feature = [quick_featurize(seq)]
    clf_predictions = []
    for clf in honed_clf_list:
        pred = OneVsRestClassifier(clf).fit(sample_seqm, class_dm).predict(seq_feature)
        clf_predictions.append(pred[0])

    for aa in seq:
        f.write(str(comp_AA_dic_rev[aa]) + ',\t' + str(aa) + ',\t')

    for prediction in clf_predictions:
        f.write(str(prediction) + ',\t')
    f.write(str(sum(clf_predictions)) + '\n')

f.close()
b3 = time.clock()
print('Prediction Time : ' + str(math.floor(b3-b2)/60) + ' minutes.')

print('End3')
