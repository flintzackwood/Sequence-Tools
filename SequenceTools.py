#Zachary Wu 2016

import numpy as np
from Bio.Seq import Seq
from Bio.Alphabet import IUPAC
import glob
import math


#removed selenocysteine (U), added *
complete_AA_list = ['R','H','K','D','E','S','T','N','Q','C','G','P','A','I', 'L','M','F',\
                    'W','Y','V','*']

comp_AA_dic = dict((j,i) for i,j in enumerate(complete_AA_list))
comp_AA_dic_rev = dict((i,j) for i,j in enumerate(complete_AA_list))


def seqList_from_file(file_dir, ref = '', cutoff = 0, mut_Ind = []):
    '''Given a directory in the form 'data/First Round/seq-edited/*.seq'
        returns bp_seq_list, bp_seq_string_list,read_length_list,AA_seq_list, mutated_indices
    '''
    list_seq_files = glob.glob(file_dir)

    bp_seq_list = []
    bp_seq_string_list = []
    read_length_list = []
    AA_seq_list = []

    #Store 3 Lists: list of base pairs as strings, Seqs, and read lengths
    for file_name in list_seq_files:
        f = open(file_name, 'r')
        line = next(f)
        while(line.strip() != '^^'):
            line = next(f)
        bp_seq = ''
        for line in enumerate(f):
            bp_seq += line[1].strip('\n')
        bp_seq_string_list.append(bp_seq)
        #bp_seq_list.append(Seq(bp_seq,IUPAC.unambiguous_dna))
        read_length_list.append(len(bp_seq))


    #Determine sequence length, cut to length%3 = 0 individually
    ref_Codon_Num = math.floor(len(ref)/3)
    for i,bp_seq in enumerate(bp_seq_string_list):
        codonNum = math.floor(len(bp_seq)/3)
        if codonNum > ref_Codon_Num and ref != '':
            bp_seq_string_list[i] = bp_seq[:(ref_Codon_Num*3)]
        else:
            bp_seq_string_list[i] = bp_seq[:(codonNum*3)]

    #Remove NDT's:
    #store altered sequences in altered_BPs
    #!!Currently 'removing' by switching all Ns with a's -- or match ref
    #Storing which mutations are changed, will use later in 'keeping' seq files
    altered_BPs = []
    poss_BP = ['A','T','C','G']
    for i, seq in enumerate(bp_seq_string_list):
        seqS = list(seq)
        altered_indices = []
        altered_aa = []
        flipped = False
        for j, aa in enumerate(seqS):
            if aa not in poss_BP:
                altered_indices.append(math.floor(j/3))
                #print(str(i) + ',' + str(j) + ',' + str(aa))
                if ref == '':
                    seqS[j] = 'A'
                    altered_aa.append('A')
                else:
                    seqS[j] = ref[j]
                    altered_aa.append(ref[j])
                flipped = True
        if flipped:
            altered_BPs.append([i,altered_indices,altered_aa])
        bp_seq_string_list[i] = ''.join(seqS)
    #altered_BPs = np.asarray(altered_BPs)

    #Create Seq files, matching AA sequence
    fileL = []
    mutS = set(mut_Ind)


    try:
        #Check if seq files have any altered base pairs
        alt_BP_Seqs = [int(i[0]) for i in altered_BPs]
    except IndexError:
        #If they don't, add sequences that are greater than cutoff
        for i,bp_seq in enumerate(bp_seq_string_list):
            if len(bp_seq)/3 > cutoff:
                temp_Seq = Seq(bp_seq,IUPAC.unambiguous_dna)
                bp_seq_list.append(temp_Seq)
                AA_seq_list.append(temp_Seq.translate())
                fileL.append(list_seq_files[i])

    else:
        #If they do, add sequences that are greater than cutoff AND
        #also do not have unreadable sequences in relevant positions
        for i,bp_seq in enumerate(bp_seq_string_list):
            #print(str(len(bp_seq)/3) + ' > ' + str(cutoff))
            #print(list_seq_files[i] + str(len(bp_seq)/3))
            if len(bp_seq)/3 > cutoff: #Check if longer than cutoff
                if i in alt_BP_Seqs:
                    j = alt_BP_Seqs.index(i) #Check if bad sequence
                    badBPs = set(altered_BPs[j][1])
                    if badBPs.intersection(mutS) == set(): #check if bad in relevant indices
                    #only add sequences that have good reads in NDT regions
                        temp_Seq = Seq(bp_seq,IUPAC.unambiguous_dna)
                        bp_seq_list.append(temp_Seq)
                        AA_seq_list.append(temp_Seq.translate())
                        fileL.append(list_seq_files[i])
                else:
                    temp_Seq = Seq(bp_seq,IUPAC.unambiguous_dna)
                    bp_seq_list.append(temp_Seq)
                    AA_seq_list.append(temp_Seq.translate())
                    fileL.append(list_seq_files[i])


    #Determine mutated residues
    try:
        comparison_sequence = AA_seq_list[0]
    except IndexError:
        print('Folder is empty')
    mutated_indices = [] #Store mutated indices
    mutation_count = []  #Store mutation count for index (find out which sites are NDT)
    for j,AA_seq in enumerate(AA_seq_list[1:]):
        for i,aa in enumerate(AA_seq):
            if comparison_sequence[i] != aa:
                if i in mutated_indices:
                    mutation_count[mutated_indices.index(i)] += 1
                else:
                    mutated_indices.append(i)
                    mutation_count.append(1)
    #Convert two arrays to format [i,j]
    #Where i is the residue and j is the mutation count
    for i,j in enumerate(mutated_indices):
        mutated_indices[i] = [j, mutation_count[i]]

    #Return all the things.
    return bp_seq_list, bp_seq_string_list, read_length_list, \
            AA_seq_list, mutated_indices, altered_BPs, fileL

def mutList_NDT(AA_seq_list, mutated_indices):
    '''returns amino acids present at each mutates residue in the form [5,2,19,0], etc.
      '''
    mutations_list = []
    for i,AA_seq in enumerate(AA_seq_list):
        mutations = [None]*len(mutated_indices)
        for j, ind in enumerate(mutated_indices):
            try: #Debugging
                mutations[j] = AA_seq[ind]
            except IndexError:
                print('Check AA_seq')
        mutations_list.append(mutations)
    # print(mutations_list)

    #Convert mutations from Strings to ints
    mutList = [[comp_AA_dic[i] for i in j] for j in mutations_list]
    return mutList

def AAFeaturize(mutList, mutated_indices):
    #change to try: except TypeError:
    #Convert sequencing information from mutList to prespecified feature representation:
    #a = [0,0,0,0,0,1,0,0...] where sum(a) = len(mutList[i])

    try:
        temp_seq = [0 for i in range(len(mutated_indices)*len(complete_AA_list))]
        temp_seq[len(complete_AA_list)*0 + mutList[0][0]] = 1
    except TypeError:
        print('entered')
        mutList = mutList_NDT(mutList, mutated_indices)

    sample_sequences = []
    for i,seq_muts in enumerate(mutList):
        temp_seq = [0 for i in range(len(mutated_indices)*len(complete_AA_list))]
        for j, mut in enumerate(seq_muts):
            temp_seq[len(complete_AA_list)*j + mut] = 1
        sample_sequences.append(temp_seq)
    return sample_sequences

def getScreeningData(file_name):
    '''file_name example: 'data/1st Plate Data (for Import) Averaged.csv'
        read in csv file containing regression(activity) and classification data
    '''
    f = open(file_name)
    #Assumes plates are always in 96 well format

    regression_data = []
    class_data = []

    for i,line in enumerate(f):
        if i<8:
            regression_data.append(line.split(','))
        elif i>8 and i<17:
            class_data.append(line.split(','))

    regression_data = [[float(j) for j in i] for i in regression_data]
    class_data = [[int(j) for j in i] for i in class_data]

    return regression_data, class_data

def matchData2SeqFile(fileL, wellID_inds, sample_sequences, regression_data, class_data):
    #Match Seq Files to Well ID
    #From seqList_from_file, takes:
    #   fileL, sample_sequences
    #From getScreeningData, takes:
    #   regression_data, class_data
    #Also requires user input: wellID_inds

    alphabet = ['A','B','C','D','E','F','G','H']
    a2n = dict((j,i) for i,j in enumerate(alphabet))
    n2a = dict((i,j) for i,j in enumerate(alphabet))
    well_id_list = []
    regression_dm = []
    class_dm = []
    neg_count = 0

    list_seq_files = glob.glob(fileL)

    sample_seqm = []
    for i,j in enumerate(list_seq_files):
        temp = j[wellID_inds[0]:wellID_inds[1]]                         #temp = j[43:46]
        temp = temp.strip('_').upper()
        temp = temp.strip('-')
        first = int(a2n[temp[0]])
        second = int(temp[1:])-1
#         print(str(first) + ' ' + str(second))

        sample_seqm.append(sample_sequences[i])
        regression_dm.append(regression_data[first][second])
        class_dm.append(class_data[first][second])
        well_id_list.append([first,second])

    return sample_seqm, regression_dm, class_dm, well_id_list

def matchData2WellID(wellIDs, sample_sequences, regression_data, class_data):
    #Match Seq Files to Well ID
    #From seqList_from_file, takes:
    #   fileL, sample_sequences
    #From getScreeningData, takes:
    #   regression_data, class_data
    #Also requires user input: wellID_inds

    alphabet = ['A','B','C','D','E','F','G','H']
    a2n = dict((j,i) for i,j in enumerate(alphabet))
    n2a = dict((i,j) for i,j in enumerate(alphabet))
    well_id_list = []
    regression_dm = []
    class_dm = []
    neg_count = 0

    sample_seqm = []
    for i,well in enumerate(wellIDs): #this part is changed from matchData2SeqFile()
        first = int(a2n[well[0]])
        second = int(well[1:])-1

        sample_seqm.append(sample_sequences[i])
        regression_dm.append(regression_data[first][second])
        try:
            class_dm.append(class_data[first][second])
        except IndexError:
            print(str(first) + ' ' + str(second))
        well_id_list.append([first,second])

    return sample_seqm, regression_dm, class_dm, well_id_list

def combineSeqData(fileL,AASeqLa,fileL2,AASeqLb):
    '''combine two sequencing data files by taking the most accurate read for each
        returns combined_wellID (list of wellIDs(str)) and combined_AASeqL(list of sequences(str))
    '''
    wellID = []
    wellID2 = []
    AASeqL = AASeqLa
    AASeqL2 = AASeqLb
    count = 0


    #Convert filenames to well lists [A,0,1]
    for file in fileL:
        tempL = list(file)[-10:-7]
        tempS = ''.join(tempL)
        tempS = tempS.strip('-')
        tempL = list(tempS)
        if len(tempL) == 3:
            wellID.append(tempL)
        elif len(tempL) == 2:
            tempL.insert(1,'0')
            wellID.append(tempL)
    for file in fileL2:
        tempL = list(file)[-10:-7]
        tempS = ''.join(tempL)
        tempS = tempS.strip('-')
        tempL = list(tempS)
        if len(tempL) == 3:
            wellID2.append(tempL)
        elif len(tempL) == 2:
            tempL.insert(1,'0')
            wellID2.append(tempL)


    #Strategy:
    #1: remove corresponding sequences from both
    #2: Add longer to final list
    #3: add last ones

    combined_AASeqL = []
    combined_wellID = []
    for i, w in enumerate(wellID):
        if w in (wellID2):
            j = wellID2.index(w)

            count += 1

            if len(AASeqL[i]) > len(AASeqL2[j]):
                combined_AASeqL.append(AASeqL[i])
            else:
                combined_AASeqL.append(AASeqL2[j])

            combined_wellID.append(''.join(w))

    for i,w in enumerate(wellID):
        if ''.join(w) not in combined_wellID:
            combined_AASeqL.append(AASeqL[i])
            combined_wellID.append(''.join(w))
    for i,w in enumerate(wellID2):
        if ''.join(w) not in combined_wellID:
            combined_AASeqL.append(AASeqL2[i])
            combined_wellID.append(''.join(w))

    return combined_wellID, combined_AASeqL

def reclassify(regression_data, low_class, high_class):
    class_data = []
    for i in regression_data:
        if i < low_class:
            class_data.append(-1)
        elif i > high_class:
            class_data.append(1)
        else:
            class_data.append(0)
    return class_data
