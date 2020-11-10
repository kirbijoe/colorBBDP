### Module containing miscellaneous helper functions ###

import numpy as np
import os
import random
from matplotlib import pyplot as plt
from matplotlib import colors
from sklearn import preprocessing
from sklearn.metrics import silhouette_score
import pandas as pd
from sklearn.cluster import KMeans
import math
from scipy import stats
import pylab
import foci as f
from sklearn import manifold
from sklearn.metrics import euclidean_distances
from scipy import spatial
from colorsims_stimuli import ColorGrid


#All WCS language numbers which are used in clustering. Removes some languages which had errors.
all_langs = list(range(1,111))
all_langs.pop(92)
all_langs.pop(91)
all_langs.pop(61)
all_langs.pop(44)

#Subset of WCS languages used for testing. Subset is representative of original distribution of BCTs.
langs_subset = [17,106,38,19,27,80,25,39,9,10,52,70,83,1,47,59,89,101,64,66,74,14,18,104]


def language_dict():
    '''Returns a dictionary where the the keys are lang_nums and the values are a dictionary containing the term numbers associated with each 
    color term in lang_num.'''

    term_enum = pd.read_table(os.path.abspath("dict.txt"))
    lang_dict = dict()
    for lang_num in np.unique(term_enum['#LNUM'])[1:]:
        term_dict = dict()
        for i in term_enum.index:
            row = term_enum.loc[i]
            if row[0] == lang_num:
                term_dict[row[3]] = int(row[1])
            lang_dict[int(lang_num)] = term_dict
        return lang_dict

def convert_rowcol_to_index(chip):
    '''Converts a color chip representation from row, col to index.'''

    if chip[0] == 'A' or chip[0] == 'J' or chip[1:] =='0':
        return -1
    else:
        row_convert = {'B':0, 'C':1, 'D':2, 'E':3, 'F':4, 'G':5, 'H':6, 'I':7}
        row = row_convert[chip[0]]
        col = int(chip[1:]) - 1
        return 40*row + col

def convert_index_to_rowcol(chip_index):
    '''Converts a color chip representation from index to row, col.''' 

    row = (chip_index-1)//40
    col = (chip_index-1)%40
    row_convert = {0:'B', 1:'C', 2:'D', 3:'E', 4:'F', 5:'G', 6:'H', 7:'I'}
    return row_convert[row] + str(col+1)

def get_category_chips(data, cat_name):
    '''Returns an 8x40 matrix (representing the color grid) with 1s in the cells which correspond to the chips an WCS participant named with cat_name.
    data is a nx320 matrix where n is the number of words in the participant's vocabulary. A cell (i,j) in data is equal to 1 if the participant used
    term i to name chip j and equal to 0 otherwise.'''

    category = np.zeros(shape=[8,40])
    for i in range(data.shape[1]):
        word_used = data.argmax(axis=0)[i]
        row = (i) // 40
        col = (i) % 40
        if word_used == cat_name:
            category[row, col] = 1
        else:
            category[row, col] = 0

    return category

def compare_categories(cat1, cat2):
    '''Returns an 8x40 matrix that shows where two categories overlap.'''

    return np.multiply(cat1, cat2)

def terms_used(data):
    '''Returns a list of the terms in the language vocabulary that a participant actually used. data is a nx320 matrix where n is the number of words 
    in the participant's vocabulary. A cell (i,j) in data is equal to 1 if the participant used term i to name chip j and equal to 0 otherwise.'''

    terms_used = []
    for row in range(len(data)):
        if sum(data[row,:]) != 0:
            terms_used.append(row)
    return terms_used

def kth_largest(arr, k):
    '''Returns the kth largest element of arr.'''

    sort_arr = sorted(arr, reverse=True)
    return sort_arr[k-1]

def find_assoc_categories(part1, part2):
    '''Returns a dictionary which maps the most similar terms between participant1 and participant 2. Returned dictionary is used to compute
    Schematic Similarity. part1 and part2 are represented by their normalized naming strategies, not word maps. Normalized naming strategies are
    nx320 matrices where n is the number of words in the participant's vocabulary. A cell (i,j) in data is equal to 1 if the participant used 
    term i to name chip j and equal to 0 otherwise.'''

    terms_used1 = terms_used(part1)
    terms_used2 = terms_used(part2)

    min_terms = min(len(terms_used1), len(terms_used2))
    if len(terms_used1) == min_terms:
        ref_participant = part1     #participant with smallest vocabulary
        oth_participant = part2     #participant with larger vocabulary
        ref_terms = terms_used1
        oth_terms = terms_used2
    else:
        ref_participant = part2     #participant with smallest vocabulary
        oth_participant = part1     #participant with larger vocabulary
        ref_terms = terms_used2
        oth_terms = terms_used1

    #Map all of the terms of the reference participant to a best match from the other participant
    assoc_categories = dict()   #key: ref_participant term, value: oth_participant term

    unpair_ref_terms = list(ref_terms)
    overlap_counts = dict()    #key: oth_participant term, value: list of all overlap sizes with every term in ref_terms
    while len(unpair_ref_terms) != 0:
        term1 = unpair_ref_terms.pop(0)
        overlap_counts[term1] = []  
        for term2 in oth_terms:
            cat_chips1 = get_category_chips(ref_participant, term1)
            cat_chips2 = get_category_chips(oth_participant, term2)
            num_overlap = np.sum(compare_categories(cat_chips1, cat_chips2))
            overlap_counts[term1].append(num_overlap)

        high_overlap_oth_term = oth_terms[np.argmax(overlap_counts[term1])]

        k = 1
        indices_tried = []  #indices of oth_terms that have already been checked (only used when there are multiple terms with the same overlap size)
        
        while high_overlap_oth_term in assoc_categories.values():
            overlap_size = overlap_counts[term1][oth_terms.index(high_overlap_oth_term)]   #overlap between term1 and oth_term
            current_ref_term = [x for x in assoc_categories if assoc_categories[x] == high_overlap_oth_term][0]   #term that has already been associated with oth_term
            current_size = overlap_counts[current_ref_term][oth_terms.index(high_overlap_oth_term)]      #size of overlap of current_ref_term with oth_term
            if overlap_size > current_size:     #new term should be associated with oth_term; kick out current_ref_term
                old_value = assoc_categories.pop(current_ref_term)  #kick out old associated ref_term
                unpair_ref_terms.append(current_ref_term)   #add it back to the list of unpaired ref_terms
                indices_tried = []
                break     
            else:   #current_ref_term stays as associated term
                k += 1

                next_largest = kth_largest(overlap_counts[term1], k)   #find second largest value in overlap_counts
                if next_largest == 0:
                    high_overlap_oth_term = None    #term1 doesn't correspond with any term in oth_term
                    indices_tried = []
                    break
                
                if overlap_counts[term1].count(next_largest) > 1:   #more than one oth_term with same size overlap
                    indices_tried.append(oth_terms.index(high_overlap_oth_term))
                    all_indices = np.where(np.array(overlap_counts[term1]) == next_largest)[0]  #all indices of oth_terms that have the same overlap size with term1                 
                    rand_index = random.choice([x for x in all_indices if x not in indices_tried])   #pick index at random that hasn't been picked already
                    high_overlap_oth_term = oth_terms[rand_index]   #get the associating oth_term
                else:
                    high_overlap_oth_term = oth_terms[overlap_counts[term1].index(next_largest)]    #get the associating oth_term
        
        assoc_categories[term1] = high_overlap_oth_term         

    #Calculate "errors" of categories in other participant that don't get an associated category.
    errors = dict()

    unassoc_terms = [x for x in oth_terms if x not in assoc_categories.values()]
    for term1 in ref_terms:
        errors[term1] = np.zeros(len(oth_terms))
        for term2 in unassoc_terms:
            cat_chips1 = get_category_chips(ref_participant, term1)
            cat_chips2 = get_category_chips(oth_participant, term2)
            term_error = np.sum(compare_categories(cat_chips1, cat_chips2))
            errors[term1][oth_terms.index(term2)] = term_error

    assoc_categories["errors"] = errors

    return assoc_categories


def all_term_similarities(part1, part2):
    '''Computes the term similarity between every pair of mapped terms obtained from find_assoc_categories(). part1 and part2 are represented by 
    their normalized naming strategies, not word maps. Normalized naming strategies are nx320 matrices where n is the number of words in the 
    participant's vocabulary. A cell (i,j) in data is equal to 1 if the participant used term i to name chip j and equal to 0 otherwise. The list
    returned from this function is used to compute the Schematic Similarity between part1 and part2. '''
    
    assoc_categories = find_assoc_categories(part1, part2)

    terms_used1 = terms_used(part1)
    terms_used2 = terms_used(part2)

    min_terms = min(len(terms_used1), len(terms_used2))
    if len(terms_used1) == min_terms:
        ref_participant = part1     #participant with smallest vocabulary
        oth_participant = part2     #participant with larger vocabulary
    else:
        ref_participant = part2     #participant with smallest vocabulary
        oth_participant = part1     #participant with larger vocabulary

    all_term_similarity = []

    for term in assoc_categories:
        if term != "errors" and assoc_categories[term] != None:
            cat_chips1 = get_category_chips(ref_participant, term)
            cat_chips2 = get_category_chips(oth_participant, assoc_categories[term])
            sample_space = np.add(cat_chips1, cat_chips2)
            sample_space_size = np.count_nonzero(sample_space)
            overlap = compare_categories(cat_chips1, cat_chips2)
            overlap_size = np.count_nonzero(overlap)
            term_similarity = float(overlap_size) / (sample_space_size + np.sum(assoc_categories["errors"][term]))
            all_term_similarity.append(term_similarity)

    return all_term_similarity

def calc_schem_similarity(all_term_similarity):
    '''Computes the Schematic Similarity between two participants. Takes a list of term_similarities obtained from all_term_similarities().'''

    return float(np.sum(all_term_similarity)) / len(all_term_similarity)


#all_participant_file = open(os.path.abspath("all_participants.txt"), 'a')  #log file which writes out the Schematic Similarity values between participants

def single_lang_comp(lang_num, all_participant_file):
    '''Computes the Schematic Similarity between all pairwise combination of participants in lang_num. All values are written to all_participant_file.'''

    directory = os.path.abspath(os.path.join("WCS Participant Data", "Lang"+str(lang_num)))
    dirs = os.listdir(directory)
    num_participants = len([f for f in dirs if os.path.isfile(os.path.join(directory, f))])

    all_similarities = []

    for i in range(1, num_participants+1):
        for j in range(i, num_participants+1):
            if i != j:
                part1 = np.array(pd.read_csv(os.path.join(directory, "Lang"+str(lang_num)+"Participant"+str(i)+".csv"), header=None))
                part2 = np.array(pd.read_csv(os.path.join(directory, "Lang"+str(lang_num)+"Participant"+str(j)+".csv"), header=None))
                schem_similarity = calc_schem_similarity(all_term_similarities(part1, part2))
                all_similarities.append(schem_similarity)
                all_participant_file.write("{}_{}\t{}_{}\t{}\n".format(lang_num, i, lang_num, j, schem_similarity))

    output = "LANG {}: Avg: {}, Std dev: {}".format(lang_num, np.average(all_similarities), np.std(all_similarities))

    return output  

def two_lang_comp(lang1, lang2, all_participant_file):
    '''Computes the Schematic Similarity between all pairwise combination of participants in lang1 and lang 2. All values are written to all_participant_file.'''

    directory1 = os.path.abspath(os.path.join("WCS Participant Data", "Lang"+str(lang1)))
    dirs1 = os.listdir(directory1)
    num_participants1 = len([f for f in dirs1 if os.path.isfile(os.path.join(directory1, f))])

    directory2 = os.path.abspath(os.path.join("WCS Participant Data", "Lang"+str(lang2)))
    dirs2 = os.listdir(directory2)
    num_participants2 = len([f for f in dirs2 if os.path.isfile(os.path.join(directory2, f))])

    all_similarities = []

    for i in range(1, num_participants1+1):
        for j in range(1, num_participants2+1):
            part1 = np.array(pd.read_csv(os.path.join(directory1, "Lang"+str(lang1)+"Participant"+str(i)+".csv"), header=None))
            part2 = np.array(pd.read_csv(os.path.join(directory2, "Lang"+str(lang2)+"Participant"+str(j)+".csv"), header=None))
            schem_similarity = calc_schem_similarity(all_term_similarities(part1, part2))
            all_similarities.append(schem_similarity)
            all_participant_file.write("{}_{}\t{}_{}\t{}\n".format(lang1, i, lang2, j, schem_similarity))
    
    output_cont = "LANG {} & {}: Avg: {}, Std dev: {}".format(lang1, lang2, np.average(all_similarities), np.std(all_similarities))

    return output_cont

def num_total_participants(lang_nums):
    '''Returns the total number of participants in the list of languages lang_nums.'''

    total = 0
    for lang in lang_nums:
        directory = os.path.abspath(os.path.join("WCS Participant Data", "Lang"+str(lang)))
        dirs = os.listdir(directory)
        total += len([f for f in dirs if os.path.isfile(os.path.join(directory, f))])
    return total

def get_ss_from_file(key1, key2):
    '''Retrieves the Schematic Similarity between two participants from file. Keys take on the form 'langnum_partnum'.
    This function assumes Schematic Similarity measures are stored in separate files by language number.'''

    if key1 == key2:
        return 0

    langs = [int(key1.split("_")[0]), int(key2.split("_")[0])]
    
    if langs[0] == langs[1]:
        ss_file = os.path.join("all_participants_files", "all_participants_samelang.txt")
    else:
        ss_file = os.path.join("all_participants_files", "all_participants_" + str(min(langs)) + ".txt")

    all_participant_ss = open(ss_file, 'r')

    line = all_participant_ss.readline()
    substr = line.split("\t")
    x1 = substr[0]
    x2 = substr[1]
    x = set([x1, x2])

    while x != set([key1, key2]):
        try:
            line = all_participant_ss.readline()
            substr = line.split("\t")
            x1 = substr[0]
            x2 = substr[1]
            x = set([x1, x2])
        except:
            #raise ValueError("Participants {} and {} were not found.".format(key1, key2))
            print("Participants {} and {} were not found.".format(key1, key2))
            return 0

    schem_sim = float(substr[2].strip())

    all_participant_ss.close()

    return schem_sim 

def trans_univ_dict(group):
    '''Translates all of the participants in group into a common language. group is a list comprised of participant keys 'langnum_partnum'.'''

    all_terms_used = []
    num_terms = []
    data = []
    for key in group:
        p_data = np.array(pd.read_csv(os.path.abspath(os.path.join("WCS Participant Data", "Lang"+key.split("_")[0], "Lang"+key.split("_")[0]+"Participant"+key.split("_")[1]+".csv")), header=None))
        terms = terms_used(p_data)
        all_terms_used.extend(terms)
        num_terms.append(len(terms))
        
        for t in terms:
            cat_vec = get_category_chips(p_data, t).flatten()
            mean = np.mean(cat_vec)
            std = np.std(cat_vec)
            norm_vec = (cat_vec - mean)/std
            data.append(norm_vec)

    X = np.array(data)

    kmeans = [KMeans(n_clusters=n, random_state=0).fit(X) for n in range(2, max(num_terms)+1)]
    #costs = [k.inertia_ for k in kmeans]   #elbow method
    costs = [silhouette_score(X, k.labels_) for k in kmeans]    #silhouette average method

    optimal_kmeans = kmeans[np.argmax(costs)]
    clusters = optimal_kmeans.labels_
    synonyms = [(all_terms_used[i], clusters[i]) for i in range(len(clusters))]

    univ_dict = {}

    index = 0
    for i in range(1, len(synonyms)+1):
        if i <= num_terms[0]:
            part = group[0]
            if i == 1:
                univ_dict[part] = {synonyms[i-1][0]: synonyms[i-1][1]}
            else:
                univ_dict[part][synonyms[i-1][0]] = synonyms[i-1][1]
                if i == num_terms[0]:
                    index += 1
                
        elif i > sum(num_terms[0:index]) and i <= sum(num_terms[0:index + 1]):
            part = group[index]
            if i == sum(num_terms[0:index])+1:
                univ_dict[part] = {synonyms[i-1][0]: synonyms[i-1][1]}
            else:
                univ_dict[part][synonyms[i-1][0]] = synonyms[i-1][1]
                if i == sum(num_terms[0:index+1]):
                    index += 1

    trans_data = []     #translated data for all participants in group
    for key in group:
        p_data = np.array(pd.read_csv(os.path.abspath(os.path.join("WCS Participant Data", "Lang"+key.split("_")[0], "Lang"+key.split("_")[0]+"Participant"+key.split("_")[1]+".csv")), header=None)).argmax(axis=0)
        p_trans = []    #individual participant's translated data
        for word in p_data:
            p_trans.append(univ_dict[key][word])
        trans_data.append(p_trans)

    return np.array(trans_data)
        
def find_expert(group):
    '''Returns individual in the group who is closest to the centroid. Value returned is a tuple with first element being the key of the expert and the
    second value being their naming strategy.'''

    group_norm = trans_univ_dict(group)     #"normalized" group where all individuals are using the same words, type=np.ndarray (shape=[group_size, 320])
    
    center = find_centroid(group).flatten()
    max_same = 0
    closest = ""
    for key in group:
        same = 0
        part = group_norm[group.index(key)]
        for i in range(320):
            if part[i] == center[i]:
                same += 1
        if same > max_same:
            max_same = same
            closest = key

    return (closest, np.array(pd.read_csv(os.path.abspath(os.path.join("WCS Participant Data", "Lang"+closest.split("_")[0], "Lang"+closest.split("_")[0]+"Participant"+closest.split("_")[1]+".csv")), header=None)).argmax(axis=0))
        
def find_centroid(group):
    '''Returns the theoretical "center" of the group. Centroid naming strategy is determined using a modal map for the group.'''

    group_norm = trans_univ_dict(group)     #"normalized" group where all individuals are using the same words, type=np.ndarray (shape=[group_size, 320])

    modal_map = np.ndarray(shape=[8,40])
    vocabulary = range(max(group_norm.flatten())+1)
		
    for chip in range(320):
        word_count = {category: 0 for category in vocabulary}
                             
        for p in group_norm:
            chip_name = p[chip]
            word_count[chip_name] += 1

        most_votes = max(word_count.values())
        most_pop_word = [key for key in word_count.keys() if word_count[key] == most_votes][0]
                
        modal_map[chip//40, chip%40] = most_pop_word

    return modal_map           

def plot_group(group, univ_trans=False, filename=None):
    '''Generates a grid layout figure with the naming strategies of all participants in group.
    univ_trans: determines whether the participants are translated into the same language or they still use their original terms'''

    grid_size = math.ceil( np.sqrt(len(group)) )
    fig, axarr = plt.subplots(grid_size, grid_size, sharex='col', sharey='row')

    if univ_trans:
        trans_data = trans_univ_dict(list(group))

    i = 0
    for r in range(grid_size):
        for c in range(grid_size):
            if i < len(group):
                p = list(group)[i]
                if univ_trans:
                    p_data = np.reshape(trans_data[i], (8,40))
                else:
                    p_data = np.array(pd.read_csv(os.path.abspath(os.path.join("WCS Participant Data", "Lang"+p.split("_")[0], "Lang"+p.split("_")[0]+"Participant"+p.split("_")[1]+".csv")), header=None))
                    p_data = np.reshape(p_data.argmax(axis=0), (8,40))
                plot_data(p_data, ax=axarr[r,c], word_labels=False, stim_labels=False, yaxis_label=False, xaxis_label=False, title=True, title_name = p)
                i += 1
            else:
                axarr[r,c].axis("off")

    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename)

    plt.show()
    plt.close()


def get_avg_ss(lang_num, num2=None, end=False):
    '''Gets the average Schematic Similarity between participants in lang_num.'''

    if end:
        samelang_ss_file = os.path.abspath(os.path.join("end_all_participants_files", "all_participants_samelang.txt"))
    else:
        samelang_ss_file = os.path.abspath(os.path.join("all_participants_files", "all_participants_samelang.txt"))

    ss_file = open(samelang_ss_file, 'r')
    readline = ss_file.readline()
    ss = []

    while readline != "":
        splitline = readline.split("\t")
        if str(lang_num)+"_" in splitline[0]:
            ss.append(float(splitline[2]))
        elif str(lang_num+1)+"_" in splitline[0]:
            break
        readline = ss_file.readline()

    ss_file.close()

    return np.average(ss)

def group_ss_stats(group):
    '''Calculates the average and standard deviation of SS of an arbitrary list of participants. group should be a list containing participant keys in 
    the form 'langnum_partnum'.'''

    pairs = [(group[i], group[j]) for i in range(len(group)) for j in range(i+1, len(group))]
    all_ss = []

    for key1, key2 in pairs:
        ss = get_ss_from_file(key1, key2)
        all_ss.append(ss)
        
    return (np.average(all_ss), np.std(all_ss))    

def lang_ss_stats(lang):
    '''Calculates the average and standard deviation SS of a WCS language.'''

    directory = os.path.abspath(os.path.join("WCS Participant Data", "Lang"+str(lang)))
    dirs = os.listdir(directory)
    num_participants = len([f for f in dirs if os.path.isfile(os.path.join(directory, f))])
    participants = ["{}_{}".format(lang, i+1) for i in range(num_participants)]

    return group_ss_stats(participants)

def plot_ss_by_lang(normfit=False):
    '''Generates a plot with two histograms: (1) within WCS language SS and (2) between WCS langauge SS.
    normfit determines whether to fit a normal distribution over the histograms.'''

    samelang_ss_file = os.path.abspath(os.path.join("all_participants_files", "all_participants_samelang.txt"))
    samelang_ss_data = np.array(pd.read_table(samelang_ss_file, header=None))
    w_data = list(map(float, samelang_ss_data[:,2]))

    b_data = []
    for i in all_langs[:-1]:    #-1 index is because the between lang files only go up to 109
        ss_file = os.path.abspath(os.path.join("all_participants_files", "all_participants_"+str(i)+".txt"))
        ss_data = np.array(pd.read_table(ss_file, header=None))
        sub_data = list(map(float, ss_data[:,2]))
        b_data.extend(sub_data)

    if normfit:
        plt.xlabel("Schematic Similarity")
        plt.ylabel("Density")
        plt.title("Histogram of Schematic Similarity")
        plt.hist(w_data, bins=100, normed=True, alpha=0.7, color="#0000FF", edgecolor='black', label="Within Language")
        plt.hist(b_data, bins=100, normed=True, alpha=0.7, color="#00FF00", edgecolor='black', label="Between Language")
        plt.legend(loc="upper right")

        w_mu, w_std = stats.norm.fit(w_data)
        b_mu, b_std = stats.norm.fit(b_data)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        w_p = stats.norm.pdf(x, w_mu, w_std)
        b_p = stats.norm.pdf(x, b_mu, b_std)
        plt.plot(x, w_p, '-k', linewidth=2)
        plt.plot(x, b_p, '-k', linewidth=2)

    else:
        plt.xlabel("Schematic Similarity")
        plt.ylabel("Frequency (proportion)")
        plt.title("Histogram of Schematic Similarity")
        plt.hist(w_data, bins=100, weights=np.ones(len(w_data))/len(w_data), alpha=0.7, color="#0000FF", edgecolor='black', label="Within Language")
        plt.hist(b_data, bins=100, weights=np.ones(len(b_data))/len(b_data), alpha=0.7, color="#00FF00", edgecolor='black', label="Between Language")
        plt.legend(loc="upper right")

    plt.show()

    return ((np.average(w_data), np.std(w_data), max(w_data), min(w_data)), (np.average(b_data), np.std(b_data), max(b_data), min(b_data)))

def bct_lookup():
    '''Returns dictionary which contains the number of BCTs each WCS language has. (keys=lang, values=num_BCTs)'''

    bct_data = np.array(pd.read_csv(os.path.abspath("colorsims NumBCT_ALL.csv"), header=None))
    bct = dict()

    for i in bct_data:
        bct[int(i[0])] = int(i[1])

    return bct

def plot_ss_by_bct(normfit=False):
    '''Generates a plot with two histograms: (1) within BCT stage SS (e.g. all pairwise SS for participants with 3 BCT) and (2) between BCT stage SS.
    normfit determines whether to fit a normal distribution over the histograms.'''

    bcts = bct_lookup()
    samelang_ss_file = os.path.abspath(os.path.join("all_participants_files", "all_participants_samelang.txt"))
    samelang_ss_data = np.array(pd.read_table(samelang_ss_file), header=None)

    w_data = []
    b_data = []

    for i in samelang_ss_data:
        lang1 = int(i[0].split('_')[0])
        lang2 = int(i[1].split('_')[0])
        bct1 = bcts[lang1] #number of BCTs in first language
        bct2 = bcts[lang2] #number of BCTs in second language

        if bct1 == bct2:
            w_data.append(float(i[2]))
        else:
            b_data.append(float(i[2]))

    for i in all_langs[:-1]:    #-1 index is because the between lang files only go up to 109
        ss_file = os.path.abspath(os.path.join("all_participants_files", "all_participants_"+str(i)+".txt"))
        ss_data = np.array(pd.read_table(ss_file, header=None))
        for j in ss_data:
            lang1 = int(j[0].split('_')[0])
            lang2 = int(j[1].split('_')[0])
            bct1 = bcts[lang1] #number of BCTs in first language
            bct2 = bcts[lang2] #number of BCTs in second language

            if bct1 == bct2:
                w_data.append(float(j[2]))
            else:
                b_data.append(float(j[2]))

    if normfit:
        plt.xlabel("Schematic Similarity")
        plt.ylabel("Density")
        plt.title("Histogram of Schematic Similarity")
        plt.hist(w_data, bins=100, normed=True, alpha=0.7, color="#0000FF", edgecolor='black', label="Within BCT")
        plt.hist(b_data, bins=100, normed=True, alpha=0.7, color="#00FF00", edgecolor='black', label="Between BCT")
        plt.legend(loc="upper right")

        w_mu, w_std = stats.norm.fit(w_data)
        b_mu, b_std = stats.norm.fit(b_data)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        w_p = stats.norm.pdf(x, w_mu, w_std)
        b_p = stats.norm.pdf(x, b_mu, b_std)
        plt.plot(x, w_p, '-k', linewidth=2)
        plt.plot(x, b_p, '-k', linewidth=2)

    else:
        plt.xlabel("Schematic Similarity")
        plt.ylabel("Frequency (proportion)")
        plt.title("Histogram of Schematic Similarity")
        plt.hist(w_data, bins=100, weights=np.ones(len(w_data))/len(w_data), alpha=0.7, color="#0000FF", edgecolor='black', label="Within BCT")
        plt.hist(b_data, bins=100, weights=np.ones(len(b_data))/len(b_data), alpha=0.7, color="#00FF00", edgecolor='black', label="Between BCT")
        plt.legend(loc="upper right")

    plt.show()

    return ((np.average(w_data), np.std(w_data), max(w_data), min(w_data)), (np.average(b_data), np.std(b_data), max(b_data), min(b_data)))

def plot_ss_random(num_groups, normfit=False):
    '''Generates a plot with two histograms: (1) within group SS and (2) between group SS. Num_groups determines how many random groups the WCS
    participants are grouped into. normfit determines whether to fit a normal distribution over the histograms.'''

    #List of participant keys
    all_participants = []

    for lang_num in all_langs:
        directory = os.path.abspath(os.path.join("WCS Participant Data", "Lang"+str(lang_num)))
        dirs = os.listdir(directory)
        num_participants = len([f for f in dirs if os.path.isfile(os.path.join(directory, f))])
        all_participants.extend([str(lang_num) + "_" + str(i) for i in range(1, num_participants+1)])

    #Assign all participants to a random cluster
    cluster_assign = random.choices(range(num_groups), k=len(all_participants))

    samelang_ss_file = os.path.abspath(os.path.join("all_participants_files", "all_participants_samelang.txt"))
    samelang_ss_data = np.array(pd.read_table(samelang_ss_file, header=None))

    w_data = []
    b_data = []

    for i in samelang_ss_data:
        part1 = i[0]
        part2 = i[1]
        group1 = cluster_assign[all_participants.index(part1)] #cluster assignment of participant1
        group2 = cluster_assign[all_participants.index(part2)] #cluster assignment of participant2

        if group1 == group2:
            w_data.append(float(i[2]))
        else:
            b_data.append(float(i[2]))

    for i in all_langs[:-1]:    #-1 index is because the between lang files only go up to 109
        ss_file = os.path.abspath(os.path.join("all_participants_files", "all_participants_"+str(i)+".txt"))
        ss_data = np.array(pd.read_table(ss_file, header=None))
        for j in ss_data:
            part1 = j[0]
            part2 = j[1]
            group1 = cluster_assign[all_participants.index(part1)] #cluster assignment of participant1
            group2 = cluster_assign[all_participants.index(part2)] #cluster assignment of participant2

            if group1 == group2:
                w_data.append(float(j[2]))
            else:
                b_data.append(float(j[2]))

    if normfit:
        plt.xlabel("Schematic Similarity")
        plt.ylabel("Density")
        plt.title("Histogram of Schematic Similarity")
        plt.hist(w_data, bins=100, normed=True, alpha=0.7, color="#0000FF", edgecolor='black', label="Within Random Group")
        plt.hist(b_data, bins=100, normed=True, alpha=0.7, color="#00FF00", edgecolor='black', label="Between Random Group")
        plt.legend(loc="upper right")

        w_mu, w_std = stats.norm.fit(w_data)
        b_mu, b_std = stats.norm.fit(b_data)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        w_p = stats.norm.pdf(x, w_mu, w_std)
        b_p = stats.norm.pdf(x, b_mu, b_std)
        plt.plot(x, w_p, '-k', linewidth=2)
        plt.plot(x, b_p, '-k', linewidth=2)

    else:
        plt.xlabel("Schematic Similarity")
        plt.ylabel("Frequency (proportion)")
        plt.title("Histogram of Schematic Similarity")
        plt.hist(w_data, bins=100, weights=np.ones(len(w_data))/len(w_data), alpha=0.7, color="#0000FF", edgecolor='black', label="Within Random Group")
        plt.hist(b_data, bins=100, weights=np.ones(len(b_data))/len(b_data), alpha=0.7, color="#00FF00", edgecolor='black', label="Between Random Group")
        plt.legend(loc="upper right")

    plt.show()

    return ((np.average(w_data), np.std(w_data), max(w_data), min(w_data)), (np.average(b_data), np.std(b_data), max(b_data), min(b_data)))

def random_clusters_iter(n, num_groups):
    '''Runs n iterations of a random clustering and returns the results from those n cluster assignments.'''

    all_participants = []

    for lang_num in all_langs:
        directory = os.path.abspath(os.path.join("WCS Participant Data", "Lang"+str(lang_num)))
        dirs = os.listdir(directory)
        num_participants = len([f for f in dirs if os.path.isfile(os.path.join(directory, f))])
        all_participants.extend([str(lang_num) + "_" + str(i) for i in range(1, num_participants+1)])

    stats = []

    for x in range(n):
        #Assign all participants to a random cluster
        cluster_assign = random.choices(range(num_groups), k=len(all_participants))

        samelang_ss_file = os.path.abspath(os.path.join("all_participants_files", "all_participants_samelang.txt"))
        samelang_ss_data = np.array(pd.read_table(samelang_ss_file, header=None))

        w_data = []
        b_data = []

        for i in samelang_ss_data:
            part1 = i[0]
            part2 = i[1]
            group1 = cluster_assign[all_participants.index(part1)] #cluster assignment of participant1
            group2 = cluster_assign[all_participants.index(part2)] #cluster assignment of participant2

            if group1 == group2:
                w_data.append(float(i[2]))
            else:
                b_data.append(float(i[2]))

        for i in all_langs[:-1]:    #-1 index is because the between lang files only go up to 109
            ss_file = os.path.abspath(os.path.join("all_participants_files", "all_participants_"+str(i)+".txt"))
            ss_data = np.array(pd.read_table(ss_file, header=None))
            for j in ss_data:
                part1 = j[0]
                part2 = j[1]
                group1 = cluster_assign[all_participants.index(part1)] #cluster assignment of participant1
                group2 = cluster_assign[all_participants.index(part2)] #cluster assignment of participant2

                if group1 == group2:
                    w_data.append(float(j[2]))
                else:
                    b_data.append(float(j[2]))

        stats.append([np.average(w_data), np.std(w_data), max(w_data), min(w_data), np.average(b_data), np.std(b_data), max(b_data), min(b_data)])
        print("Finished {} out of {}".format(x, n))

    stats = np.array(stats)
    return ((np.average(stats[:,0]), np.average(stats[:,1]),np.average(stats[:,2]), np.average(stats[:,3])), (np.average(stats[:,4]), np.average(stats[:,5]), np.average(stats[:,6]), np.average(stats[:,7])))  #((w_average, w_std, w_max, w_min), (b_average, b_std, b_max, b_min))

def get_neighbor_chips(chip_index, num_neigh=8):
    '''Returns the set of 4 vertically and horizonally adjacent neighbors of a chip where the neighbors are given as (row, col) pairs. chip_index 
    is a number between 0 and 319 which represents the index of a color chip in the flattened Munsell color grid. '''

    r = chip_index // 40  #row index
    c = chip_index % 40   #col index

    if r == 0:
        if num_neigh == 8:
            return [(r, (c-1)%40), (r, (c+1)%40), (r+1, (c-1)%40), (r+1, c), (r+1, (c+1)%40)]
        else: #num_neigh == 4
            return [(r, (c-1)%40), (r, (c+1)%40), (r+1, c)]
    elif r == 7:
        if num_neigh == 8:
            return [(r-1, (c-1)%40), (r-1, c), (r-1, (c+1)%40), (r, (c-1)%40), (r, (c+1)%40)]
        else: #num_neigh == 4
            return [(r-1, c), (r, (c-1)%40), (r, (c+1)%40)] 
    else:
        if num_neigh == 8:
            return [(r-1, (c-1)%40), (r-1, c), (r-1, (c+1)%40), (r, (c-1)%40), (r, (c+1)%40), (r+1, (c-1)%40), (r+1, c), (r+1, (c+1)%40)]
        else: #num_neigh == 4
            return [(r-1, c), (r, (c-1)%40), (r, (c+1)%40), (r+1, c)] 

def transform_data(lang_num, part_num, end=False, foci=False, foci_thresh=0.04, num_neigh=8):
    '''end: determines whether to cluster based on WCS data or simulated data from Gooyabadi et al. (2019)
    foci: if True, transforms data according to a subsample determined by the proportion of participants which labeled a certain color as focal'''
    
    if end:
        part_data = np.array(pd.read_csv(os.path.abspath(os.path.join("WCS End Participant Data", "Lang"+str(lang_num), "Lang"+str(lang_num)+"Participant"+str(part_num)+".csv")), header=None))
    else:
        part_data = np.array(pd.read_csv(os.path.abspath(os.path.join("WCS Participant Data", "Lang"+str(lang_num), "Lang"+str(lang_num)+"Participant"+str(part_num)+".csv")), header=None))
    word_map = part_data.argmax(axis=0) #This is a 1x320 array of most probable word for each stimulus.

    transformed = []

    indices_visited = []

    if foci:
        indices = [x for x in range(len(word_map)) if f.props[x] >= foci_thresh]
    else:
        indices = range(len(word_map))

    for i in indices:
        for r,c in get_neighbor_chips(i, num_neigh=num_neigh):
            j = r*40 + c
            if set((i,j)) not in indices_visited:
                if word_map[i] == word_map[j]:
                    transformed.append(1)
                else:
                    transformed.append(0)
                indices_visited.append(set((i,j)))

    return transformed

def normalize_data(data):
    '''Normalizes a vector to have a mean of 0 and a standard deviation of 1.'''

    mean = np.average(data)
    sd = np.std(data)

    return [(x - mean)/sd for x in data]    

def transform_data_all(lang_list, end=False, norm=False, foci=False, foci_thresh=0.04, num_neigh=8):
    '''Transforms the data of all participants from the languages in lang_list into binary features vectors.
    norm: normalizes the features vectors to have a mean of 0 and a standard deviation of 1
    foci: determines whether to consider a subset of color chips based on their location relative to its category
    foci_thresh: threshold which determines the subset of chips; includes all chips with value greater than foci_thresh
    num_neigh: number of neighbors to consider (either 4 or 8)'''

    part_key = [] #list of the particiapant keys in the order they appear along the rows of the data matrix
    data = []
    
    for lang in lang_list:
        if end:
            directory = os.path.abspath(os.path.join("WCS End Participant Data", "Lang"+str(lang)))
        else:
            directory = os.path.abspath(os.path.join("WCS Participant Data", "Lang"+str(lang)))
        dirs = os.listdir(directory)
        num_participants = len([f for f in dirs if os.path.isfile(os.path.join(directory, f))])

        for p in range(1, num_participants+1):
            p_data = transform_data(lang, p, end, foci, foci_thresh, num_neigh)
            if norm:
                p_data = normalize_data(p_data)
            data.append(p_data)
            part_key.append(str(lang)+"_"+str(p))

        print("Finished Lang " + str(lang))

    return (part_key, np.array(data))

def mds(dist_matrix, plots='mds'):
    '''Generates an MDS plot based on dist_matrix.'''

    mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-12, dissimilarity="precomputed", n_jobs=1)
    pos = mds.fit(dist_matrix).embedding_

    nmds = manifold.MDS(n_components=2, metric=False, max_iter=3000, eps=1e-12, dissimilarity="precomputed", n_jobs=1, n_init=1)
    npos = nmds.fit_transform(dist_matrix, init=pos)

    fig = plt.figure(1)
    ax = plt.axes([0., 0., 1., 1.])

    s = 100
    if plots == 'mds' or plots == 'both':
        plt.scatter(pos[:, 0], pos[:, 1], color='turquoise', s=s, lw=0, label='MDS')
        for i in range(np.shape(dist_matrix)[0]):
            plt.annotate(i, (pos[:, 0][i], pos[:, 1][i]))
    if plots == 'nmds' or plots == 'both':
        plt.scatter(npos[:, 0], npos[:, 1], color='darkorange', s=s, lw=0, label='NMDS')
        for i in range(np.shape(dist_matrix)[0]):
            plt.annotate(i, (npos[:, 0][i], npos[:, 1][i]))

    plt.legend(scatterpoints=1, loc='best', shadow=False)

    plt.show()

    return (pos, npos)

def mds_rsquared(orig_dist, mds_results):
    '''Returns the r-squared of an resulting MDS plot.'''

    mds_dist = distance.cdist(mds_results, mds_results, "euclidean")
    r = stats.pearsonr(orig_dist.flatten(), mds_dist.flatten())
    return r[0]**2

def get_numBCT(lang_num):
    '''Returns the number of BCTs identified for lang_num.'''

    numbct_file = open(os.path.abspath("colorsims NumBCT_ALL.csv"), 'r')
    line = numbct_file.readline()
    while line.split(",")[0] != lang_num:
        line = numbct_file.readline()

    num_BCT = int(line.split(",")[1])
    return num_BCT

def boundary_values(folder, group, grp_id, num_ksims=1):
    '''Produces a file with the boundary value for every chip in the WCS grid (chromatic part) for every participant in group
    folder = name of the folder to store the boundary values files (should match the name of the cluster_results file, but without the .txt extension)
    group = list of participant keys in the form [lang_num]_[part_num]
    grp_id = number assigned to group by model
    num_ksims = size of the search radius'''

    if not os.path.exists(os.path.abspath(os.path.join("Boundary Analysis", folder))):
        os.mkdir(os.path.abspath(os.path.join("Boundary Analysis", folder))) 
    
    bnd_val_file = open(os.path.abspath(os.path.join("Boundary Analysis", folder, "Group"+str(grp_id)+"_bound_vals_"+str(num_ksims)+"ksims.txt")), 'w')
    first_line = "Participant Key"
    for i in range(320):
        first_line += "\t"+str(i)
    first_line += "\n"
    bnd_val_file.write(first_line)

    num_agents = len(group)
    G = ColorGrid()

    conv_file = open(os.path.abspath("MunsellCIE.txt"), "r")
    LAB_table = G.LAB_lookup(conv_file)
    LUV_table = G.LUV_lookup(LAB_table)
    row_convert = {0:'B', 1:'C', 2:'D', 3:'E', 4:'F', 5:'G', 6:'H', 7:'I'}

    for key in group:
        lang_num = key.split("_")[0]
        part_num = key.split("_")[1]
        num_BCT = get_numBCT(lang_num)
        k_sim = find_ksim(num_BCT)
        p_data = np.array(pd.read_csv(os.path.abspath(os.path.join("WCS Participant Data", "Lang"+lang_num, "Lang"+lang_num+"Participant"+part_num+".csv")), header=None))
        word_map = p_data.argmax(axis=0)
    
        file_line = str(i+1)
        for chip in range(320):
            stim_within_ksim = []
            for other_chip in range(320):
                chip_row = chip // 40
                chip_col = chip % 40 

                other_chip_row = other_chip // 40
                other_chip_col = other_chip % 40

                Mun_stim1 = (row_convert[chip_row], str(chip_col+1))	#stimulus form: (row, col) 
                Mun_stim2 = (row_convert[other_chip_row], str(other_chip_col+1))

                LAB_stim1 = LAB_table[Mun_stim1]	#stimulus form: (L,a,b)
                LAB_stim2 = LAB_table[Mun_stim2]
		
                LUV_stim1 = LUV_table[LAB_stim1]	#stimulus form: (L,u,v)
                LUV_stim2 = LUV_table[LAB_stim2]

                if G.Euclid_distance(LUV_stim1, LUV_stim2) <= num_ksims*k_sim and chip != other_chip:
                    stim_within_ksim.append(other_chip)

            same_name = 0
            for chip_ksim in stim_within_ksim:
                if word_map[chip] == word_map[chip_ksim]:
                    same_name += 1
            try:
                prop_same_name = float(same_name) / len(stim_within_ksim)
            except ZeroDivisionError:
                prop_same_name = 0
            file_line += "\t"+str(prop_same_name)

        file_line += "\n"
        bnd_val_file.write(file_line)

    bnd_val_file.close()


def gen_prob_func(x_vals, y_vals):
    step_function = dict()
    for i in range(len(x_vals)):
        step_function[x_vals[i]] = y_vals[i]

    def prob_function(bound_val):
        truncate = math.floor(bound_val*10**2) / 10**2
        return step_function[truncate]

    return prob_function
    
def boundary_probs(folder, group, grp_id, num_ksims, picture=False):
    bound_data = np.array(read_table(os.path.abspath(os.path.join("Boundary Analysis", folder, "Group"+str(grp_id)+"_bound_vals_"+str(num_ksims)+"ksims.txt"))))
    bound_data = bound_data[1:,1:].astype(float)
    averages = np.average(bound_data, axis=0)   #average boundary values for each of the 320 color chips
    x_values = []
    y_values = []
    for i in range(101):    #construct x-axis ranging from 0 to 1 in incrememnts of 0.01
        x_values.append(round(1 - i*0.01, 2))
    for x in x_values:      #construct cumulative distribution of boundary values
        num_chips = (averages > x).sum()    #number of chips that have an average boundary value above x
        prob = float(num_chips)/320         #proportion of chips that have an average boundary value above x
        y_values.append(prob)

    step_function = gen_prob_func(x_values, y_values)   #function that returns the boundary probability of a chip
    all_chip_prob = []
    bound_prob_file = open(os.path.abspath(os.path.join("Boundary Analysis", folder, "Group"+str(grp_id)+"_bound_probs.txt")), 'w')
    bound_prob_file.write("Chip Num\tBoundary Value\tBoundary Probability\n")
    for i in range(len(averages)):
        chip_prob = step_function(averages[i])
        all_chip_prob.append(chip_prob)
        bound_prob_file.write("{}\t{}\t{}\n".format(i, averages[i], chip_prob)) 
        
    bound_prob_file.close()
    
    if picture:
        prob_grid_form = np.reshape(all_chip_prob, [8,40])

        image_path = os.path.abspath(os.path.join("Boundary Analysis", "Heatmaps"))

        ax = None
        fig, ax = plt.subplots()

        ax.matshow(prob_grid_form, cmap="hot_r")

        ax.set_yticks([])
        ax.set_yticklabels([])
   
        ax.set_xticks([])
        ax.set_xticklabels([])
		
        plt.savefig(os.path.join(image_path, "Group"+str(grp_id)+"_bound_probs.png"))
        plt.close()

                         

#Use this function for visualizing individual WCS naming data to assess qualitative value of grouping.
def plot_participant(lang_num, part_num, ax=None, filename=None, word_labels=False, stim_labels=False, linewidth=3, color='b', yaxis_label=False, xaxis_label=False):
    filename = os.path.abspath(os.path.join("Pictures", "Lang" + str(lang_num) + "Part" + str(part_num)))
    part_data = np.array(pd.read_csv(os.path.abspath(os.path.join("WCS Participant Data", "Lang"+str(lang_num), "Lang"+str(lang_num)+"Participant"+str(part_num)+".csv")), header=None))
    word_map = np.reshape(part_data.argmax(axis=0), (8,40)) #This is a 8x40 matrix of most probable word for each stimulus.

    if ax is None:
        fig, ax = plt.subplots()


    #List of 90 colors (list of 30 x3)
    all_colors = ['#8000ff', '#00ff00', '#0040ff', '#cc4400', '#0080ff', '#ff0040', '#ffff00', '#a67c00', '#bf9b30', '#5e3c58', '#bf00ff', '#00ffbf', '#ff0080', 
'#ffdbac', '#000000', '#ff8000', '#d4d4d4', '#555555', '#aafd96', '#00bfff', '#ff00ff', '#ff93ac', '#ffbf00', '#400000', '#204c39', '#997a8d', '#063b79', 
'#757906', '#70330b', '#00ffff', '#8000ff', '#00ff00', '#0040ff', '#cc4400', '#0080ff', '#ff0040', '#ffff00', '#a67c00', '#bf9b30', '#5e3c58', '#bf00ff', 
'#00ffbf', '#ff0080', '#ffdbac', '#000000', '#ff8000', '#d4d4d4', '#555555', '#aafd96', '#00bfff', '#ff00ff', '#ff93ac', '#ffbf00', '#400000', '#204c39', 
'#997a8d', '#063b79', '#757906', '#70330b', '#00ffff', '#8000ff', '#00ff00', '#0040ff', '#cc4400', '#0080ff', '#ff0040', '#ffff00', '#a67c00', '#bf9b30', 
'#5e3c58', '#bf00ff', '#00ffbf', '#ff0080', '#ffdbac', '#000000', '#ff8000', '#d4d4d4', '#555555', '#aafd96', '#00bfff', '#ff00ff', '#ff93ac', '#ffbf00', 
'#400000', '#204c39', '#997a8d', '#063b79', '#757906', '#70330b', '#00ffff']

    cmap = colors.ListedColormap(all_colors)
    bounds = []
    for i in range(len(all_colors)+1):
        bounds.append(i)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    ax.matshow(word_map, cmap=cmap, norm=norm)

    ax.set_yticks([])
    ax.set_yticklabels([])

    ax.set_xticks([])
    ax.set_xticklabels([])
				
    #Save to file?
    if filename is not None:
        plt.savefig(filename)


def plot_data(data, ax=None, filename=None, word_labels=False, stim_labels=False, linewidth=3, color='b', yaxis_label=False, xaxis_label=False, title=False, title_name=""):
    '''Plots a word_map straight from data. data is a word_map (8x40), where each element is the color name corresponding to that chip index.'''
    
    if ax is None:
        fig, ax = plt.subplots()

    #List of 90 colors (list of 30 x3)
    all_colors = ['#8000ff', '#00ff00', '#0040ff', '#cc4400', '#0080ff', '#ff0040', '#ffff00', '#a67c00', '#bf9b30', '#5e3c58', '#bf00ff', '#00ffbf', '#ff0080', 
'#ffdbac', '#000000', '#ff8000', '#d4d4d4', '#555555', '#aafd96', '#00bfff', '#ff00ff', '#ff93ac', '#ffbf00', '#400000', '#204c39', '#997a8d', '#063b79', 
'#757906', '#70330b', '#00ffff', '#8000ff', '#00ff00', '#0040ff', '#cc4400', '#0080ff', '#ff0040', '#ffff00', '#a67c00', '#bf9b30', '#5e3c58', '#bf00ff', 
'#00ffbf', '#ff0080', '#ffdbac', '#000000', '#ff8000', '#d4d4d4', '#555555', '#aafd96', '#00bfff', '#ff00ff', '#ff93ac', '#ffbf00', '#400000', '#204c39', 
'#997a8d', '#063b79', '#757906', '#70330b', '#00ffff', '#8000ff', '#00ff00', '#0040ff', '#cc4400', '#0080ff', '#ff0040', '#ffff00', '#a67c00', '#bf9b30', 
'#5e3c58', '#bf00ff', '#00ffbf', '#ff0080', '#ffdbac', '#000000', '#ff8000', '#d4d4d4', '#555555', '#aafd96', '#00bfff', '#ff00ff', '#ff93ac', '#ffbf00', 
'#400000', '#204c39', '#997a8d', '#063b79', '#757906', '#70330b', '#00ffff']

    cmap = colors.ListedColormap(all_colors)
    bounds = []
    for i in range(len(all_colors)+1):
        bounds.append(i)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    ax.matshow(data, cmap=cmap, norm=norm)

    #Show the word labels?
    #Here, the "word labels" corresponds to the number of rows in the color grid.
    ax.set_yticks([])
    ax.set_yticklabels([])

    ax.set_xticks([])
    ax.set_xticklabels([])

    if title:
        ax.title.set_text(title_name)

    #Save to file?
    if filename is not None:
        plt.savefig(filename)

