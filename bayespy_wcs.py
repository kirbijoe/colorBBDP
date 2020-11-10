### Module with code to execute the Beta-Bernoulli Dirichlet Process Mixture Model using the World Color Survey data. ###

import numpy as np
import utils as u
from os import path, listdir, mkdir
from bayespy.nodes import Categorical, Dirichlet
from bayespy.nodes import Beta
from bayespy.nodes import Mixture, Bernoulli
from bayespy.inference import VB
import bayespy.plot as bpplt
import math
from matplotlib import pyplot as plt
from matplotlib import colors, transforms
from scipy.spatial import distance
import random
from sklearn.cluster import KMeans
from sklearn import manifold
import pandas as pd



def map_to_naming_strategy(word_map):
    '''Converts a word_map into a naming_strategy.'''

    word_map = word_map.flatten()
    
    agent_strategy = np.zeros((int(max(word_map))+1, 320), dtype=np.int)
        
    for chip_num in range(len(word_map)):
        agent_strategy[word_map[chip_num], chip_num] = 1

    return agent_strategy

def parse_results_file(filename):
    '''Reads resulting data from a Bernoulli Dirichlet clustering.
    [0] = Latent variable Z (i.e. probability with which each observation Xi belongs to the K clusters)
    [1] = Weight matrix R (i.e. values proportional to the size of the resulting clusters)
    [2] = Participant list [0] and corresponding cluster assignments [1] (i.e. cluster assignments retrieved by taking argmax of Z)
    [n] = Further results from other methods
    
    **All data structures returned contain str type data.'''
    
    results = dict()
    file = open(path.abspath(filename), 'r')
    line = file.readline()
    r = []
    while line != "":
        if "---" in line:
            name = line[line.find("---")+len("---"):-1]
            r = np.array(r)
            if np.shape(r)[0] == 1:
                r = r.flatten()
            results[name] = r
            r = []
        else:
            r.append(line.strip().split("\t"))
            
        line = file.readline()

    return results



class BBDP:
    '''Class representing the Beta-Bernoulli Dirichlet Process Mixture Model with Variational Inference.'''

    def __init__(self, lang_list, write_to_file=True):
        self.langs = lang_list
        self.in_file = []   #keeps track of what information has already been written to the output/results file
        self.write_to_file = write_to_file  #determines whether to write results from the clustering 

    def run(self, K=25, beta=0.5, alpha=0.00001, foci_thresh=0, num_neigh=4, hinton_plot=False, end=False):
        '''Performs one run of the BBDP according to the specified parameters.'''

        print("Transforming WCS participant data into binary vectors...")
        x = u.transform_data_all(self.langs, norm=False, end=end, foci=True, foci_thresh=foci_thresh, num_neigh=num_neigh)
        print("Finished transforming participant data") 
        self.participant_list = x[0]
        
        N = len(x[0])            #number of data points (i.e. WCS participants)
        D = np.shape(x[1])[1]    #number of features
        #K = 20            #number of initial clusters
        
        R = Dirichlet(K*[alpha],
                      name='R')
        Z = Categorical(R,
                        plates=(N,1),
                        name='Z')
        
        P = Beta([beta, beta],
                 plates=(D,K),
                 name='P')
        
        X = Mixture(Z, Bernoulli, P)
        
        Q = VB(Z, R, X, P)
        P.initialize_from_random()
        X.observe(x[1])
        Q.update(repeat=1000)

        if hinton_plot:
            bpplt.hinton(Z)
            bpplt.pyplot.show()
            
            bpplt.hinton(R)
            bpplt.pyplot.show()

        #Get the weight matrix stored in Z (weights determine which cluster data point belongs to)
        z = Z._message_to_child()[0]
        z = z * np.ones(Z.plates+(1,))
        z = np.squeeze(z)
        self.z = z

        #Get the weights stored in R (proportional to the size of the clusters)
        r = np.exp(R._message_to_child()[0])
        r = r * np.ones(R.plates+(1,))
        r = np.squeeze(r)
        self.r = r

        #Get the cluster assignment of each data point
        self.c_assign = np.argmax(self.z, axis=1)

        #Write cluster results to a file
        if self.write_to_file:
            if end:
                save_path = "cluster_results_end_K={}_B={}_a={}_t={}_nn={}".format(K, beta, alpha, foci_thresh, num_neigh)
            else:
                save_path = "cluster_results_K={}_B={}_a={}_t={}_nn={}".format(K, beta, alpha, foci_thresh, num_neigh)
            while path.exists(save_path+".txt"):
                #save_path already exists
                try:
                    old_file_num = int(save_path[save_path.find('(')+1:-1])
                    new_file_num = old_file_num + 1
                    save_path = save_path[0:save_path.find('(')] + '(' + str(new_file_num) + ')'
                except ValueError:
                    save_path = save_path + " (1)"

            self.save_path = save_path       
            file = open(path.abspath(self.save_path+".txt"), 'w')
            
            #Write cluster assignment matrix Z (gives the probability that observation i belongs to cluster j)
            if 'Z' not in self.in_file:
                for i in range(len(self.z)):
                    line = "\t".join([str(x) for x in self.z[i]]) + "\n"
                    file.write(line)
                file.write('---Z\n')
                self.in_file.append('Z')

            #Write cluster weights matrix R (proportional to the size of the resulting clusters)
            if 'R' not in self.in_file:
                line = "\t".join([str(x) for x in self.r]) + "\n"
                file.write(line)
                file.write('---R\n')
                self.in_file.append('R')

            #Write deterministic cluster assignments with the corresponding participant key
            if 'C' not in self.in_file:
                line1 = "\t".join([str(x) for x in self.participant_list]) + "\n"
                line2 = "\t".join([str(x) for x in self.c_assign]) + "\n"              
                file.write(line1)
                file.write(line2)
                file.write('---C\n')
                self.in_file.append('C')
            
            file.close()

        return self.c_assign

    def _run(self, x, K=25, beta=0.5, alpha=0.00001, hinton_plot=False, end=False):
        '''Only to be used when doing parameter optimization.'''

        self.participant_list = x[0]
        
        N = len(x[0])            #number of data points (i.e. WCS participants)
        D = np.shape(x[1])[1]    #number of features
        #K = 20            #number of initial clusters
        
        R = Dirichlet(K*[alpha],
                      name='R')
        Z = Categorical(R,
                        plates=(N,1),
                        name='Z')
        
        P = Beta([beta, beta],
                 plates=(D,K),
                 name='P')
        
        X = Mixture(Z, Bernoulli, P)
        
        Q = VB(Z, R, X, P)
        P.initialize_from_random()
        X.observe(x[1])
        Q.update(repeat=1000)

        log_likelihood = Q.L[Q.iter-1]

        if hinton_plot:
            bpplt.hinton(Z)
            bpplt.pyplot.show()
            
            bpplt.hinton(R)
            bpplt.pyplot.show()

        #Get the weight matrix stored in Z (weights determine which cluster data point belongs to)
        z = Z._message_to_child()[0]
        z = z * np.ones(Z.plates+(1,))
        z = np.squeeze(z)
        self.z = z

        #Get the weights stored in R (proportional to the size of the clusters)
        r = np.exp(R._message_to_child()[0])
        r = r * np.ones(R.plates+(1,))
        r = np.squeeze(r)
        self.r = r

        #Get the cluster assignment of each data point
        self.c_assign = np.argmax(self.z, axis=1)

        return log_likelihood

    def get_groups(self):
        '''Returns a list of lists which represents the resulting clusters from the BBDP.'''

        self.groups = []
        labels = np.unique(self.c_assign)
        for label in labels:
            c_parts = np.where(self.c_assign == label, self.participant_list, None)
            c_parts = [x for x in c_parts if x != None]
            self.groups.append(c_parts)

        #Write groups to results file
        if self.write_to_file and 'G' not in self.in_file:
            file = open(path.abspath(self.save_path + ".txt"), 'a')
            for g in self.groups:
                line = "\t".join(g) + "\n"
                file.write(line)
            file.write("---G\n")
            file.close()
            self.in_file.append('G')
            
        return self.groups
    
    def plot_group_centers(self, cut_small=False, save_fig=True):
        '''Plots the centroids of the resulting clusters organized from largest to smallest cluster.
        cut_small determines whether or not to plot the small clusters (i.e. clusters with <1% of sample).'''

        grid_size = math.ceil( np.sqrt(len(self.groups)) )
        fig, axarr = plt.subplots(grid_size, grid_size, sharex='col', sharey='row')

        weights = [self.r[i] for i in np.unique(self.c_assign)]     #remove non-existent groups np.unique(self.c_assign)
        sort_weights = sorted(weights, reverse=True)
        index_list = []
        for w in sort_weights:
            indices = np.where(weights == w)[0]
            if indices[0] not in index_list:
                index_list.extend(indices)

        self.group_centers = [None]*len(self.groups)

        i = 0
        for r in range(grid_size):
            for c in range(grid_size):
                if i < len(self.groups):
                    group = self.groups[index_list[i]]
                    if len(group) == 1:
                        key = group[0]
                        p_data = np.array(pd.read_csv(path.abspath(path.join("WCS Participant Data", "Lang"+key.split("_")[0], "Lang"+key.split("_")[0]+"Participant"+key.split("_")[1]+".csv")), header=None))
                        data = np.reshape(p_data.argmax(axis=0), (8,40))
                    else:
                        data = u.find_centroid(group)
                    u.plot_data(data, ax=axarr[r,c], word_labels=False, stim_labels=False, yaxis_label=False, xaxis_label=False, title=True, title_name=str(round(sort_weights[i], 4)))
                    self.group_centers[index_list[i]] = data.astype(int)
                    print("Finished group {} of {}".format(i+1, len(self.groups)))
                    i += 1
                else:
                    axarr[r,c].axis("off")

        plt.tight_layout()

        if save_fig:
            plt.savefig(self.save_path + ".png")

        plt.show()
        plt.close()

        if cut_small:
            print("Generating figure without small groups:")
            count_small = 0
            for g in self.groups:
                if len(g) < math.floor(len(self.participant_list)*0.01):
                    count_small += 1

            num_groups_keep = len(self.groups) - count_small
            index_list_small = index_list[:num_groups_keep]
            grid_size_small = math.ceil( np.sqrt(num_groups_keep) )
            fig, axarr = plt.subplots(grid_size_small, grid_size_small, sharex='col', sharey='row')

            i = 0
            for r in range(grid_size_small):
                for c in range(grid_size_small):
                    if i < num_groups_keep:
                        group = self.groups[index_list[i]]
                        data = self.group_centers[index_list[i]].reshape((8,40))
                        u.plot_data(data, ax=axarr[r,c], word_labels=False, stim_labels=False, yaxis_label=False, xaxis_label=False, title=True, title_name=str(round(sort_weights[i], 4)))
                        print("Finished group {} of {}".format(i+1, num_groups_keep))
                        i += 1
                    else:
                        axarr[r,c].axis("off")

            plt.tight_layout()

            if save_fig:
                plt.savefig(self.save_path + "_remove_small.png")

            plt.show()
            plt.close()

        if self.write_to_file and "G_centers" not in self.in_file:
            file = open(path.abspath(self.save_path + ".txt"), 'a')
            for word_map in self.group_centers:
                line = "\t".join([str(word) for word in word_map.flatten()]) + "\n"
                file.write(line)
            file.write("---G_centers\n")
            file.close()
            self.in_file.append("G_centers")                         

    def ss_dist_matrix(self):
        '''Returns a matrix where the (i,j) entry is the distance (1-SS) between the centroids of cluster i and cluster j.'''
        
        N = len(self.groups)
        self.dist_matrix = np.zeros(shape=[N,N])
        for i in range(N):
            for j in range(i+1, N):
                centroid_i = map_to_naming_strategy(self.group_centers[i])
                centroid_j = map_to_naming_strategy(self.group_centers[j])
                self.dist_matrix[i,j] = 1 - u.calc_schem_similarity(u.all_term_similarities(centroid_i, centroid_j))
                self.dist_matrix[j,i] = self.dist_matrix[i,j]

        if self.write_to_file and "SS_distances" not in self.in_file:
            file = open(path.abspath(self.save_path + ".txt"), 'a')
            for i in range(self.dist_matrix.shape[0]):
                line = "\t".join([str(x) for x in self.dist_matrix[i]]) + "\n"
                file.write(line) 
            file.write("---SS_distances\n")
            file.close()  
            self.in_file.append("SS_distnaces")      

        return self.dist_matrix

    def mds(self, plots='mds'):
        '''Generates an MDS plot to provide a spatial representation of the resulting clusters.
        self.mds_results[0] = (x,y) coordinates results from multidimensional scaling
        self.mds_results[1] = (x,y) coordinate results from non-metric multidimensional scaling
        plots = "mds", "nmds", or "both"'''

        mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-12, dissimilarity="precomputed", n_jobs=1)
        pos = mds.fit(self.dist_matrix).embedding_

        nmds = manifold.MDS(n_components=2, metric=False, max_iter=3000, eps=1e-12, dissimilarity="precomputed", n_jobs=1, n_init=1)
        npos = nmds.fit_transform(self.dist_matrix, init=pos)

        fig = plt.figure(1)
        ax = plt.axes([0., 0., 1., 1.])

        s = 100
        if plots == 'mds' or plots == 'both':
            scatter = plt.scatter(pos[:, 0], pos[:, 1], c=[len(g) for g in self.groups], cmap='hot_r', edgecolors='black', s=[len(g) for g in self.groups], label='MDS')
            fig.colorbar(scatter)
            for i in range(np.shape(self.dist_matrix)[0]):
                plt.annotate(i, (pos[:, 0][i], pos[:, 1][i]))
        if plots == 'nmds' or plots == 'both':
            plt.scatter(npos[:, 0], npos[:, 1], color='darkorange', s=s, lw=0, label='NMDS')
            for i in range(np.shape(self.dist_matrix)[0]):
                plt.annotate(i, (npos[:, 0][i], npos[:, 1][i]))

        #plt.legend(scatterpoints=1, loc='best', shadow=False)

        plt.tight_layout()
        plt.show()
        
        self.mds_results = (pos, npos)

    def _get_group_of_participant(self, p):
        '''Returns the group a participant is in.
        p: participant id (i.e. str in the format 'langnum_partnum')'''
        
        return np.where(np.array([p in g for g in self.groups]))[0][0]

    def _cluster_distribution(self, num, comp="lang", cut_small=False, id=False):
        '''Returns the number of clusters a language was divided into.
        comp: "lang" or "bct"
        cut_small: determines whether to include small clusters (less than 25 participants (i.e. 1% of population)) in the analysis
        id: determines whether the values of the dict are a count of participants in each cluster or a list of participant ids.'''
        
        in_groups = dict()     #keeps track of how participants are in each group/cluster

        if comp == "lang":
            directory = path.abspath(path.join("WCS Participant Data", "Lang"+str(num)))
            dirs = listdir(directory)
            num_participants = len([f for f in dirs if path.isfile(path.join(directory, f))])
            participants = ["{}_{}".format(num, i+1) for i in range(num_participants)]
        else: #comp == "bct"
            num_participants = 0
            participants = []
            bct_lookup = u.bct_lookup()
            for lang_num in self.langs:
                if bct_lookup[lang_num] == num:
                    directory = path.abspath(path.join("WCS Participant Data", "Lang"+str(lang_num)))
                    dirs = listdir(directory)
                    lang_num_participants = len([f for f in dirs if path.isfile(path.join(directory, f))])
                    num_participants += lang_num_participants
                    participants.extend(["{}_{}".format(lang_num, i+1) for i in range(lang_num_participants)])

        for p in participants:
            group = self._get_group_of_participant(p)

            if (cut_small and len(self.groups[group]) >= math.floor(len(self.participant_list)*0.01)) or (not cut_small):    #25 ~= 1% of total WCS participants
                try:
                    if id:
                        in_groups[group].append(p)
                    else:
                        in_groups[group] += 1
                except KeyError:
                    if id:
                        in_groups[group] = [p]
                    else:
                        in_groups[group] = 1

        return in_groups

    def num_clusters_all_langs(self, comp="lang", cut_small=False):
        '''Returns a dictionary whose keys are WCS langs/BCT stages and whose values are the number of clusters the language/stage was split up into.'''
        
        if comp == "lang":
            self.lang_num_clusts = {i: len(self._cluster_distribution(i, comp, cut_small)) for i in self.langs}
            return self.lang_num_clusts
        else: #comp == "bct"
            self.bct_num_clusts = {i: len(self._cluster_distribution(i, comp, cut_small)) for i in range(3,13)}
            return self.bct_num_clusts

    def clust_dist_all_langs(self, comp="lang", cut_small=False):
        '''Returns a dictionary whose keys are WCS langs/BCT stages and whose values are dictionaries that show which clusters participants were grouped into.'''
        
        if comp == "lang":
            self.lang_clust_dists = {i: self._cluster_distribution(i, comp, cut_small) for i in self.langs}
            return self.lang_clust_dists  
        else: #comp == "bct" 
            self.bct_clust_dists = {i: self._cluster_distribution(i, comp, cut_small) for i in range(3,13)}
            return self.bct_clust_dists

    def _group_error(self, num, comp="lang", dist_type="ss", cut_small=False):
        '''Calculates the group error of a WCS language or BCT stage.
        num = either lang_num or num_BCTs
        comp = "lang" or "bct"'''
        
        if comp == "lang":
            directory = path.abspath(path.join("WCS Participant Data", "Lang"+str(num)))
            dirs = listdir(directory)
            num_participants = len([f for f in dirs if path.isfile(path.join(directory, f))])
            if cut_small:
                participants = ["{}_{}".format(num, i+1) for i in range(num_participants) if len(self.groups[self._get_group_of_participant("{}_{}".format(num, i+1))]) >= math.floor(len(self.participant_list)*0.01)]
            else:
                participants = ["{}_{}".format(num, i+1) for i in range(num_participants)]
        else: #comp == "bct"
            num_participants = 0
            participants = []
            bct_lookup = u.bct_lookup()
            for lang_num in self.langs:
                if bct_lookup[lang_num] == num:
                    directory = path.abspath(path.join("WCS Participant Data", "Lang"+str(lang_num)))
                    dirs = listdir(directory)
                    lang_num_participants = len([f for f in dirs if path.isfile(path.join(directory, f))])
                    num_participants += lang_num_participants
                    if cut_small:
                        participants.extend(["{}_{}".format(lang_num, i+1) for i in range(lang_num_participants) if len(self.groups[self._get_group_of_participant("{}_{}".format(lang_num, i+1))]) >= math.floor(len(self.participant_list)*0.01)])
                    else:
                        participants.extend(["{}_{}".format(lang_num, i+1) for i in range(lang_num_participants)])

        if dist_type == "mds":
            dist_matrix = distance.cdist(self.mds_results[1], self.mds_results[1], "euclidean")
        else:   #dist_type == "ss"
            dist_matrix = self.dist_matrix

        dists = []

        for i in range(len(participants)):
            for j in range(i+1, len(participants)):
                i_group = self._get_group_of_participant(participants[i])
                j_group = self._get_group_of_participant(participants[j])
                dist = dist_matrix[i_group, j_group]
                dists.append(dist)

        return np.average(dists)

    def group_error_all_langs(self, comp="lang", dist_type="ss", cut_small=False):
        '''Computes the group error of all langs.'''

        if comp == "lang":
            self.lang_group_errors = {i: self._group_error(i, comp, dist_type, cut_small) for i in self.langs}
            return self.lang_group_errors
        else: #comp == "bct"
            self.bct_group_errors = {i: self._group_error(i, comp, dist_type, cut_small) for i in range(3,13)}
            return self.bct_group_errors

    def _simpsons_index(self, num, comp="lang", cut_small=False):
        '''Calculates Simpson's Index for WCS language or BCT stage.'''
        
        in_groups = self._cluster_distribution(num, comp, cut_small)
        num_participants = sum(in_groups.values())
        try:
            simp_index = 1 - sum([n*(n-1) for n in in_groups.values()])/(num_participants*(num_participants - 1))
        except ZeroDivisionError:
            simp_index = None

        return simp_index

    def simpsons_index_all_langs(self, comp="lang", cut_small=False):
        '''Computes Simpson's Diversity Index for all langs.'''

        if comp == "lang":
            self.lang_simpsons_indices = {i: self._simpsons_index(i, comp, cut_small) for i in self.langs}
            return self.lang_simpsons_indices 
        else: #comp == "bct"
            self.bct_simpsons_indices = {i: self._simpsons_index(i, comp, cut_small) for i in range(3,13)}
            return self.bct_simpsons_indices 

    def plot_group_numbers(self, save_fig=True):
        '''Plots the group centroids of the resulting clusters using the cluster index as the header of each subfigure. This representation
        is useful when generating a key for MDS plots.'''

        grid_size = math.ceil( np.sqrt(len(self.groups)) )
        fig, axarr = plt.subplots(grid_size, grid_size, sharex='col', sharey='row')

        weights = [self.r[i] for i in np.unique(self.c_assign)]    #remove non-existent groups
        sort_weights = sorted(weights, reverse=True)
        index_list = []
        for w in sort_weights:
            indices = np.where(weights == w)[0]
            index_list.extend(indices)

        i = 0
        for r in range(grid_size):
            for c in range(grid_size):
                if i < len(self.groups):
                    group = self.groups[index_list[i]]
                    data = self.group_centers[index_list[i]].reshape((8,40))
                    u.plot_data(data, ax=axarr[r,c], word_labels=False, stim_labels=False, yaxis_label=False, xaxis_label=False, title=True, title_name=str(index_list[i]))
                    i += 1
                else:
                    axarr[r,c].axis("off")

        plt.tight_layout()

        if save_fig:
            plt.savefig(self.save_path + "_wgrpnum.png")

        plt.show()
        plt.close()

    def cluster_ss(self, plot=True):
        '''Computes SS statistics for each of the resulting clusters from the IMM.'''
        
        samelang_ss_file = path.abspath(path.join("all_participants_files", "all_participants_samelang.txt"))
        data = []
        data.append(np.array(pd.read_table(samelang_ss_file, header=None)))

        index = 0
        for i in self.langs[:-1]:   #-1 index is because the between lang files only go up to 109
            btw_ss_file = path.abspath(path.join("all_participants_files", "all_participants_"+str(i)+".txt"))
            try:
                data[index] = np.concatenate((data[index], np.array(pd.read_table(btw_ss_file, header=None))))
            except:     #Too much data to contain in one matrix (memory constraint)---have to break into multiple structures
                data.append(np.array(pd.read_table(btw_ss_file, header=None)))
                index += 1

        self.within_group_ss = [[] for i in range(len(self.groups))]
        self.btw_group_ss = []

        for data_i in data:
            for row in data_i:
                p1 = row[0]
                p2 = row[1]
                g = self._same_cluster(p1, p2)
                if g == -1:
                    self.btw_group_ss.append(float(row[2]))
                else:
                    self.within_group_ss[g].append(float(row[2]))

        self.ss = []
        for x in self.within_group_ss:
            if x == []:
                self.ss.append((1,0))
            else:
                self.ss.append((np.average(x), np.std(x)))

        if self.write_to_file and "G_ss_stats" not in self.in_file:
            matrix = np.array(self.ss).transpose()  #reorganized ss stats for groups (row_0=averages, row_1=stdevs)
            file = open(path.abspath(self.save_path + ".txt"), 'a')
            for i in range(matrix.shape[0]):
                line = "\t".join([str(x) for x in matrix[i]]) + "\n"
                file.write(line)
            file.write("---G_ss_stats\n")
            self.in_file.append("G_ss_stats")

            # matrix = np.array(self.within_group_ss)
            # for i in range(matrix.shape[0]):
            #     line = "\t".join([str(x) for x in matrix[i]]) + "\n"
            #     file.write(line)
            # file.write("---Within_group_ss\n")
            # self.in_file.append("Within_group_ss")

            # line = "\t".join([str(x) for x in self.btw_group_ss]) + "\n"
            # file.write(line)
            # file.write("---Between_group_ss\n")
            # self.in_file.append("Between_group_ss")
            file.close()

        return self.ss

    def summary_cluster_ss(self, plot=False):
        '''Computes statistics for SS for the within and between groups, where the groups are the clusters from the IMM.'''

        within_group_ss = [ss for g in self.within_group_ss for ss in g]   #flatten the list

        if plot:
            plt.xlabel("Schematic Similarity")
            plt.ylabel("Frequency (proportion)")
            plt.title("Histogram of Schematic Similarity for Resulting Clusters")
            plt.hist(within_group_ss, bins=100, weights=np.ones(len(within_group_ss))/len(within_group_ss), alpha=0.7, color="#0000FF", edgecolor='black', label="Within Cluster")
            plt.hist(self.btw_group_ss, bins=100, weights=np.ones(len(self.btw_group_ss))/len(self.btw_group_ss), alpha=0.7, color="#00FF00", edgecolor='black', label="Between Cluster")
            plt.legend(loc="upper right")
            plt.show()

        return [(np.average(within_group_ss), np.std(within_group_ss)), (np.average(self.btw_group_ss), np.std(self.btw_group_ss))]

    def _same_cluster(self, p1, p2):
        '''If p1 and p2 were assigned to the same cluster, returns the group index, else returns False.'''
        
        for g in self.groups:
            if p1 in g and p2 in g:
                return list(self.groups).index(g)

        return -1    #Didn't find a group with both p1 and p2

    def gen_boundary_files(self, num_ksims=1):
        '''Generate all of the boundary value and boundary probability files for a run of a clustering model. 
        num_ksims = size of search radius'''
        
        for i in range(len(self.groups)):
            u.boundary_values(self.save_path, self.groups[i], i, num_ksims)
            u.boundary_probs(self.save_path, self.groups[i], i, num_ksims)

    def plot_bound_probs(self, cut_small=False):
        '''Plots boundary probability heat maps for each of the resulting clusters. This function cannot be run unless the Boundary Analysis
        files have been generated first. (Files are generated by calling self.gen_boundary_files())'''
        
        grid_size = math.ceil( np.sqrt(len(self.groups)) )
        fig, axarr = plt.subplots(grid_size, grid_size, sharex='col', sharey='row')

        weights = [self.r[i] for i in np.unique(self.c_assign)]     #remove non-existent groups (unused groups have weight=0)
        sort_weights = sorted(weights, reverse=True)
        index_list = []
        for w in sort_weights:
            indices = np.where(weights == w)[0]
            index_list.extend(indices)

        i = 0
        print("Generating figure...")
        for r in range(grid_size):
            for c in range(grid_size):
                if i < len(self.groups):
                    group = self.groups[index_list[i]]
                    bound_data = np.array(pd.read_table(path.abspath(path.join("Boundary Analysis", self.save_path, "Group"+str(index_list[i])+"_bound_probs.txt"))))
                    prob_grid_form = np.reshape(bound_data[1:,2].astype(float), [8,40])

                    ax = axarr[r,c]

                    ax.matshow(prob_grid_form, cmap="hot_r")

                    ax.set_yticks([])
                    ax.set_yticklabels([])
               
                    ax.set_xticks([])
                    ax.set_xticklabels([])
                                               
                    i += 1
                else:
                    axarr[r,c].axis("off")

        plt.tight_layout()

        image_path = path.abspath(path.join("Boundary Analysis", "Heatmaps"))
        plt.savefig(path.join(image_path, self.save_path + ".png"))

        plt.show()
        plt.close()

        if cut_small:
            print("Generating figure without small groups...")
            count_small = 0
            for g in self.groups:
                if len(g) < math.floor(len(self.participant_list)*0.01):
                    count_small += 1

            num_groups_keep = len(self.groups) - count_small
            index_list_small = index_list[:num_groups_keep]
            grid_size_small = math.ceil( np.sqrt(num_groups_keep) )
            fig, axarr = plt.subplots(grid_size_small, grid_size_small, sharex='col', sharey='row')

            i = 0
            for r in range(grid_size_small):
                for c in range(grid_size_small):
                    if i < num_groups_keep:
                        group = self.groups[index_list[i]] ### THIS NEEDS ATTENTION
                        bound_data = np.array(pd.read_table(path.abspath(path.join("Boundary Analysis", self.save_path, "Group"+str(index_list[i])+"_bound_probs.txt"))))
                        prob_grid_form = np.reshape(bound_data[1:,2].astype(float), [8,40])

                        ax = axarr[r,c]

                        ax.matshow(prob_grid_form, cmap="hot_r")

                        ax.set_yticks([])
                        ax.set_yticklabels([])
                   
                        ax.set_xticks([])
                        ax.set_xticklabels([])
                                                   
                        i += 1
                    else:
                        axarr[r,c].axis("off")

            plt.tight_layout()

            image_path = path.abspath(path.join("Boundary Analysis", "Heatmaps"))
            plt.savefig(path.join(image_path, self.save_path + "_remove_small.png"))

            plt.show()
            plt.close()

    def lang_cluster_centroids(self, lang_num, cut_small=True):
        '''Plots the cluster centroids of the clusters that lang_num is split up into.'''

        clust_dist = self._cluster_distribution(num, comp, cut_small)
        clust_dist_sort = sorted(clust_dist.items(), key=lambda x: x[1], reverse=True)
        grid_size = math.ceil( np.sqrt(len(clust_dist)) )
        fig, axarr = plt.subplots(grid_size, grid_size, sharex='col', sharey='row')  
        
        i = 0
        for r in range(grid_size):
            for c in range(grid_size):
                if i < len(clust_dist_sort):
                    data = self.group_centers[clust_dist_sort[i][0]].reshape((8,40)) 
                    u.plot_data(data, ax=axarr[r,c], word_labels=False, stim_labels=False, yaxis_label=False, xaxis_label=False, title=True, title_name=str(clust_dist_sort[i][1]))
                    i += 1  
                else:
                    axarr[r,c].axis("off")
        plt.tight_layout()

        image_path = path.abspath(path.join("Boundary Analysis", "Heatmaps", self.save_path))
        if not path.exists(image_path):
            mkdir(image_path)

        if cut_small:
            image_name = comp.capitalize() + str(num) + "_centroids_remove_small.png"
        else:
            image_name = comp.capitalize() + str(num) + "_centroids.png"
        plt.savefig(path.join(image_path, image_name))
        plt.close()

    def lang_cluster_heatmaps(self, num, comp="lang", cut_small=True):
        '''Plots the cluster heatmaps of the clusters that lang_num is split up into.'''

        clust_dist = self._cluster_distribution(num, comp, cut_small)
        clust_dist_sort = sorted(clust_dist.items(), key=lambda x: x[1], reverse=True)
        grid_size = math.ceil( np.sqrt(len(clust_dist)) )
        fig, axarr = plt.subplots(grid_size, grid_size, sharex='col', sharey='row')  
        
        i = 0
        for r in range(grid_size):
            for c in range(grid_size):
                if i < len(clust_dist_sort):
                    bound_data = np.array(pd.read_table(path.abspath(path.join("Boundary Analysis", self.save_path, "Group"+str(clust_dist_sort[i][0])+"_bound_probs.txt"))))
                    prob_grid_form = np.reshape(bound_data[1:,2].astype(float), [8,40])

                    ax = axarr[r,c]

                    ax.matshow(prob_grid_form, cmap="hot_r")

                    ax.set_yticks([])
                    ax.set_yticklabels([])
               
                    ax.set_xticks([])
                    ax.set_xticklabels([])

                    i += 1  
                else:
                    axarr[r,c].axis("off")
        plt.tight_layout()

        image_path = path.abspath(path.join("Boundary Analysis", "Heatmaps", self.save_path))
        if not path.exists(image_path):
            mkdir(image_path)

        if cut_small:
            image_name = comp.capitalize() + str(num) + "_heatmaps_remove_small.png"
        else:
            image_name = comp.capitalize() + str(num) + "_heatmaps.png"
        plt.savefig(path.join(image_path, image_name))
        plt.close()

    def _ss_part_to_centroid(self, p_id, comp="wcs"):
        '''Returns the SS between a WCS participant and the centroid of a group it belongs to: its WCS language (comp='wcs') or its assigned cluster (comp='cluster').'''
        
        p_data = np.array(pd.read_csv(path.abspath(path.join("WCS Participant Data", "Lang"+p_id.split("_")[0], "Lang"+p_id.split("_")[0]+"Participant"+p_id.split("_")[1]+".csv")), header=None))
        p_data = np.reshape(p_data.argmax(axis=0), (8,40))

        if comp == "wcs":
            lang_num = int(p_id.split("_")[0])
            centroid = np.array(pd.read_csv(path.abspath(path.join("WCS_modalmaps_csv", "modal_map_"+str(lang_num)+".csv")), header=None)).astype(int)
        else: #comp == "cluster"
            group_num = self._get_group_of_participant(p_id)
            centroid = self.group_centers[group_num]

        ss = u.calc_schem_similarity(u.all_term_similarities(map_to_naming_strategy(p_data), map_to_naming_strategy(centroid)))

        return ss


    
class BBDP_from_file(BBDP):
    '''Reads in the information from a BBDP results file.'''

    def __init__(self, filename, langs = u.all_langs):
        BBDP.__init__(self, langs)
        self.save_path = filename
        self.results = parse_results_file(filename + ".txt")
        self.z = self.results['Z'].astype(float)
        self.r = self.results['R'].astype(float)
        self.participant_list = self.results['C'][0]
        self.c_assign = self.results['C'][1].astype(int)

        #Check which methods were already run
        if 'G' in self.results.keys():
            self.groups = self.results['G']
        if "G_centers" in self.results.keys():
            self.group_centers = self.results["G_centers"].astype(int)
        if "G_ss_stats" in self.results.keys():
            self.ss = self.results["G_ss_stats"].astype(float).transpose()
        if "SS_distances" in self.results.keys():
            self.dist_matrix = self.results["SS_distances"].astype(float)
        # if "Within_group_ss" in self.results.keys():
        #     self.within_group_ss = self.results["Within_group_ss"].astype(float)
        # if "Between_group_ss" in self.results.keys():
        #     self.btw_group_ss = self.results["Between_group_ss"].astype(float)

        self.in_file = list(self.results.keys())

    def get_groups(self):
        try:
            self.groups = self.results['G']
            print("NOTE: Groups have already been computed for this simulation.")
        except KeyError:
            return BBDP.get_groups(self)

    def plot_group_centers(self, cut_small=False):
        try:
            self.group_centers = self.results["G_centers"].astype(float).astype(int)
            print("NOTE: Group centers have already been computed for this simulation.")
        except KeyError:
            BBDP.plot_group_centers(self, cut_small)

    def cluster_ss(self):
        try:
            self.ss = self.results["G_ss_stats"].astype(float).transpose()
            print("NOTE: Group SS stats have already been computed for this simulation. \nDo you still want to execute cluster_ss()? (Type 'Y' for yes or 'N' for no.)")
            ans = input()
            while ans.capitalize() != "Y" and ans.capitalize() != "N":
                print("Invalid input: Please type either 'Y' or 'N'.")
                ans = input()

            if ans.capitalize() == "Y":
                BBDP.cluster_ss(self)

        except KeyError:
            return BBDP.cluster_ss(self)

    def ss_dist_matrix(self):
        try:
            self.dist_matrix = self.results["SS_distances"].astype(float)
            print("NOTE: Between-group SS distances have already been computed for this simulation.")
        except KeyError:
            return BBDP.ss_dist_matrix(self)




def param_opt(n, K=25, comp="none", plot=True, end=False):
    '''Performs a random search to determine the optimal parameters to pass to the BBDP. 
    n determines the number of iterations to test. K is the upper bound on the number of clusters.'''
    
    alpha_exps = [x for x in random.choices(range(-7, 5),k=n)]      #exponents (powers of 10) for alphas
    alpha_vals = [10**x for x in alpha_exps]                        #actual alphas
    beta_vals = [random.uniform(0.275,1) for x in range(n)]         #only pick betas between 0.275 and 1 because anything less causes the log likelihood to be nan

    B = BBDP(u.all_langs, False)
    print("Transforming WCS participant data into binary vectors...")
    data = u.transform_data_all(B.langs, norm=False, end=end, foci=True, foci_thresh=0, num_neigh=4)
    print("Finished transforming participant data")

    opt_function = lambda x,y: sum(x) + sum([1-j for j in y])    #maximize similarity within clusters and minimize similarity between clusters

    all_K = []              #list of number of inferred groups for each combination of parameters
    all_LL = []             #list of converged log likelihoods for each combination of parameters 
    opt_LL = -1000000       #highest log likelihood so far
    opt_params_LL = tuple() # combination of parameters yielding highest log likelihood so far
    all_vals = []           #list of  wellformedness values for each combination of parameters
    opt_val = 0             #highest wellformedness so far
    opt_params = tuple()    #combination of parameters yielding highest wellformedness so far
    index_delete = []       #list of indices to delete from the alpha/beta_vals arrays (combinations of parameters that didn't work)
    for i in range(n):
        # print("*** Iteration {} out of {} ***".format(i+1, n))
        alpha = alpha_vals[i]
        beta = beta_vals[i]
        print("*** Iteration {} out of {} ***: alpha={}, beta={}".format(i+1, n, alpha, beta))

        try:
            LL = B._run(data, K=K, beta=beta, alpha=alpha, hinton_plot=False, end=False)
            B.get_groups()
            inf_K = len(B.groups)
            print("Computing group schematic similarity stats...")
            B.cluster_ss()
            stats = B.summary_cluster_ss(plot=False)    #((w_avg, w_std), (b_avg, b_std))
            print("Finished computing group schematic similarity stats")
        except Exception as e:     #If the combination of parameters doesn't allow the IMM to run...
            print("Error: " + str(e))
            print("Bad combination of parameters: alpha={}, beta={}".format(alpha, beta))
            index_delete.append(i)
            continue
        except KeyboardInterrupt:
            print("Error: KeyboardInterrupt")
            print("Bad combination of parameters: alpha={}, beta={}".format(alpha, beta))
            index_delete.append(i)
            continue

        new_val = opt_function([ss for g in B.within_group_ss for ss in g], B.btw_group_ss)
        all_vals.append(new_val)
        all_LL.append(LL)
        all_K.append(inf_K)

        if new_val > opt_val:
            opt_val = new_val
            opt_params = (alpha, beta)

        if LL > opt_LL:
            opt_LL = LL
            opt_params_LL = (alpha, beta)

    index_delete.reverse()
    for j in index_delete:
        alpha_exps.pop(j)
        alpha_vals.pop(j)
        beta_vals.pop(j)

    if plot:    #plot scatter plot of opt_func values with (log) alpha, beta values on x, y axis (respectively)
        if comp == "lang":
            div_norm = colors.DivergingNorm(vcenter=u.ss_wellformed_by_lang())
            color_map = 'coolwarm'
        elif comp == "bct":
            div_norm = colors.DivergingNorm(vcenter=u.ss_wellformed_by_bct())
            color_map = 'coolwarm'
        else: #comp == "none"
            div_norm = None
            color_map = 'hot_r'

        # start with a square Figure
        fig = plt.figure(figsize=(8, 8))

        # Add a gridspec with two rows and two columns and a ratio of 2 to 7 between
        # the size of the marginal axes and the main axes in both directions.
        # Also adjust the subplot parameters for a square plot.
        gs = fig.add_gridspec(2, 2,  width_ratios=(7, 2), height_ratios=(2, 7),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.05, hspace=0.05)

        ax = fig.add_subplot(gs[1, 0])
        ax_x = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_y = fig.add_subplot(gs[1, 1], sharey=ax)

        # no labels
        ax_x.tick_params(axis="x", labelbottom=False)
        ax_y.tick_params(axis="y", labelleft=False)

        # the scatter plot:
        ax.set_xlabel("Log Alpha")
        ax.set_ylabel("Beta")
        scatter = ax.scatter(alpha_exps, beta_vals, c=all_vals, norm=div_norm, cmap=color_map, edgecolors='black')

        # now marginal plots:
        marg_alpha_exp = sorted(np.unique(alpha_exps))
        avg_all_vals_a = [np.average([all_vals[i[0]] for i in np.argwhere(np.array(alpha_exps) == a)]) for a in marg_alpha_exp]
        ax_x.plot(marg_alpha_exp, avg_all_vals_a)
        # ax_x.fill_between(marg_alpha_exp, min(avg_all_LL_a), avg_all_LL_a, alpha=0.3)

        # first of all, the base transformation of the data points is needed
        all_vals_sort = [all_vals[beta_vals.index(b)] for b in sorted(beta_vals)]
        ax_y.plot(all_vals_sort, sorted(beta_vals))
        # ax_y.fill_between(beta_vals, min(all_LL), all_LL, alpha=0.3)

        fig.suptitle("Wellformedness Values for Alpha and Beta Hyperparameters")
        fig.colorbar(scatter)
        fig.tight_layout()
        # if end:
        #     figname = "Optimal parameters_end_{}_wellformed_a={}_b={}.png".format(n, opt_params[0], opt_params[1])
        # else:
        #     figname = "Optimal parameters_{}_wellformed_a={}_b={}.png".format(n, opt_params[0], opt_params[1])
        # plt.savefig(figname)
        plt.show()

        plt.clf()
        # start with a square Figure
        fig = plt.figure(figsize=(8, 8))

        # Add a gridspec with two rows and two columns and a ratio of 2 to 7 between
        # the size of the marginal axes and the main axes in both directions.
        # Also adjust the subplot parameters for a square plot.
        gs = fig.add_gridspec(2, 2,  width_ratios=(7, 2), height_ratios=(2, 7),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.05, hspace=0.05)

        ax = fig.add_subplot(gs[1, 0])
        ax_x = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_y = fig.add_subplot(gs[1, 1], sharey=ax)

        # no labels
        ax_x.tick_params(axis="x", labelbottom=False)
        ax_y.tick_params(axis="y", labelleft=False)

        # the scatter plot:
        ax.set_xlabel("Log Alpha")
        ax.set_ylabel("Beta")
        scatter = ax.scatter(alpha_exps, beta_vals, c=all_LL, norm=div_norm, cmap=color_map, edgecolors='black')

        # now marginal plots:
        marg_alpha_exp = sorted(np.unique(alpha_exps))
        avg_all_LL_a = [np.average([all_LL[i[0]] for i in np.argwhere(np.array(alpha_exps) == a)]) for a in marg_alpha_exp]
        ax_x.plot(marg_alpha_exp, avg_all_LL_a)
        # ax_x.fill_between(marg_alpha_exp, min(avg_all_LL_a), avg_all_LL_a, alpha=0.3)

        # first of all, the base transformation of the data points is needed
        all_LL_sort = [all_LL[beta_vals.index(b)] for b in sorted(beta_vals)]
        ax_y.plot(all_LL_sort, sorted(beta_vals))
        # ax_y.fill_between(beta_vals, min(all_LL), all_LL, alpha=0.3)

        fig.suptitle("Log Likelihoods for Alpha and Beta Hyperparameters")
        fig.colorbar(scatter)
        fig.tight_layout()
        # if end:
        #     figname = "Optimal parameters_end_{}_loglikelihood_a={}_b={}.png".format(n, opt_params_LL[0], opt_params_LL[1])
        # else:
        #     figname = "Optimal parameters_{}_loglikelihood_a={}_b={}.png".format(n, opt_params_LL[0], opt_params_LL[1])
        # plt.savefig(figname)
        plt.show()

    # Write results to file
    if end:
        filename = path.abspath("Optimal parameters_end_results_"+str(n)+".txt")
    else:
        filename = path.abspath("Optimal parameters_results_"+str(n)+".txt")
    file = open(filename, 'w')
    file.write("Alpha\tBeta\tWellformedness\tELBO\tNum Clusters")
    for i in range(len(all_K)):
        line = "{}\t{}\t{}\t{}\t{}\n".format(alpha_vals[i], beta_vals[i], all_vals[i], all_LL[i], all_K[i])
        file.write(line)
    file.close()

    return (opt_params, opt_params_LL)

