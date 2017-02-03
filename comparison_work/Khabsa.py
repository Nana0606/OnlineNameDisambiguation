# implement the idea in the paper titled "online name disambiguation with contraints"
# For fair comparison, use groundtruth class label during the batch mode

# Author: Baichuan Zhang

import os;
import sys;
import random;
import numpy as np;
from scipy.spatial import distance;


def File_Reader(feature_file, label_file):

	# read the ground-truth file

	doc_set = set();
	doc_label_dict = {};

	for line in label_file:

		line = line.strip();

		if "<entity id" in line:

			# extract label information
			label = int(line[line.find('"') + 1:line.rfind('"')].strip());

		elif "<doc rank" in line:

			# extract doc information and consider rank id as doc id
			doc_id = int(line[line.find('"') + 1: line.rfind('"')].strip());
			doc_set.add(doc_id);

			if doc_id not in doc_label_dict:
				doc_label_dict[doc_id] = label;

			else:
				# if one doc belongs to multiple labels, then we consider it as ambiguous case and discard them
				del doc_label_dict[doc_id];
				doc_set.remove(doc_id);

		elif "<discarded>" in line:
			
			# for the unlabeled web link, discard them 
			break;

	doc_list = list(doc_set);

	# read feature file
	# first column is doc_id and the rest is feature values (all freq count)

	doc_featurelist_dict = {};

	for line in feature_file:

		line = line.strip();

		linetuple = line.split(",");

		doc_index = int(linetuple[0]);

		# only count for non-discard document
		if doc_index in doc_list:
			doc_featurelist_dict[doc_index] = map(int, linetuple[1:]);

	return [doc_list, doc_featurelist_dict, doc_label_dict];


def Data_Split(doc_list, doc_featurelist_dict, doc_label_dict, percent):

	# randomly split training and test dataset with their corresponding label information
	train_set_dict = {};
	test_set_list = [];
	test_label_list = [];

	random.shuffle(doc_list);
	num_doc = len(doc_list);

	train_num = int(percent * num_doc);
	doc_train = doc_list[:train_num];
	doc_test = doc_list[train_num:];

	for i in doc_train:
		
		train_label = doc_label_dict[i];
		
		if train_label not in train_set_dict:
			train_set_dict[train_label] = [doc_featurelist_dict[i]];
		
		else:
			train_set_dict[train_label].append(doc_featurelist_dict[i]);


	for j in doc_test:

		test_set_list.append(doc_featurelist_dict[j]);
		test_label_list.append(doc_label_dict[j]);


	return [train_set_dict, test_set_list, test_label_list];


# compute the mean F1 in clustering setting (chapter 17 in Zaki's data mining book)

def Compute_F1(test_label_list, predict_label_list):

	predict_label_dict = {};

	for pos in range(0,len(predict_label_list)):

		predict_label_ = int(predict_label_list[pos]);
		
		if predict_label_dict.get(predict_label_, -1) == -1:
		
			pred_paperid_list = [];
			
			# paper index starts from 1
			pred_paperid_list.append(pos+1);
			predict_label_dict[predict_label_] = pred_paperid_list;
		
		else:
		
			pred_paperid_list = predict_label_dict[predict_label_];	
			pred_paperid_list.append(pos+1);
			predict_label_dict[predict_label_] = pred_paperid_list;

	predict_num_cluster = len(predict_label_dict);
	
	true_label_dict = {};
	for inpos in range(0,len(test_label_list)):
		
		true_label = int(test_label_list[inpos]);
		
		if true_label_dict.get(true_label, -1) == -1:
		
			paperid_list = [];
			# paper index starts from 1
			paperid_list.append(inpos+1);
			true_label_dict[true_label] = paperid_list;
		
		else:
		
			paperid_list = true_label_dict[true_label];	
			paperid_list.append(inpos+1);
			true_label_dict[true_label] = paperid_list;

	true_num_cluster = len(true_label_dict);

	# let's denote C(r) as clustering result and T(k) as partition (ground-truth), and construct r*k contingency table
	r_k_table = [];
	
	for v1 in predict_label_dict.itervalues():
	
		k_list = [];
	
		for v2 in true_label_dict.itervalues():
			N_ij = int(len(set(v1).intersection(v2)));
			k_list.append(N_ij);

		r_k_table.append(k_list);

	r_k_matrix = np.array(r_k_table);
	r_num = int(r_k_matrix.shape[0]);

	# compute the F1 for each C_i (row)
	sum_f1 = 0.0;
	for row in range(0, r_num):
	
		row_sum = np.sum(r_k_matrix[row,:]);
	
		if row_sum != 0:
			max_col_index = np.argmax(r_k_matrix[row,:]);	
			row_max_value = r_k_matrix[row,max_col_index];
			prec = float(row_max_value)/row_sum;
			col_sum = np.sum(r_k_matrix[:,max_col_index]);
			rec = float(row_max_value)/col_sum;
			row_f1 = float(2*prec*rec)/(prec+rec);
			sum_f1 = sum_f1 + row_f1;
	
	aver_f1 = float(sum_f1)/r_num;

	return [aver_f1, predict_num_cluster, true_num_cluster];	


# perform the online name disambiguation based on distance based metric

def Inc_ND(train_set_dict, test_set_list, test_label_list, ratio_threshold):
	
	# initialize emerging class label

	emerge_label = 5000;
	predict_label_list = [];

	for pos in range(0,len(test_set_list)):
		
		doc_instance = test_set_list[pos];
		cluster_distlist_dict = {};
		
		dist_collection = [];

		for k1, v1 in train_set_dict.items():

			dist_list = [];
			
			for v1_ in range(0,len(v1)):
				
				# compute the euclidean distance between each test point and all documents in each class
				dist = distance.euclidean(doc_instance, v1[v1_]);

				dist_collection.append(dist);
				dist_list.append(dist);
			
			cluster_distlist_dict[k1] = dist_list;

		# as we use supervised setting in the batch mode, so no merging cluster operation during the online setting	
		# return the highest ratio value with the class_id information and the ratio value represents the density of the \epsilon-neighbor of test data point
		
		# for each test point coming from a streaming manner, we estimate a new distance threshold
		epsilon = np.mean(dist_collection);

		highest_ratio = 0.0;
		max_class_id = -1;

		for a, b in cluster_distlist_dict.items():

			# compute the number of points in dist_list which are smaller than pre-defined threshold epsilon
			ratio = float(sum(1 for i in b if i <= epsilon))/(len(b));

			if ratio > highest_ratio:

				highest_ratio = ratio;
				max_class_id = a;

		if highest_ratio < ratio_threshold:

			train_set_dict[emerge_label] = [doc_instance];
			emerge_label = emerge_label + 1;
			predict_label_list.append(emerge_label);
		
		else: 

			train_set_dict[max_class_id].append(doc_instance);
			predict_label_list.append(max_class_id);
	
	aver_f1, predict_num_cluster, true_num_cluster = Compute_F1(test_label_list, predict_label_list);	

	return [aver_f1, predict_num_cluster, true_num_cluster];	


if __name__ == '__main__':
	
	feature_file = open(sys.argv[1], 'r');
	label_file = open(sys.argv[2], 'r');
	percent = float(sys.argv[3]);
	ratio_threshold = float(sys.argv[4]);

	doc_list, doc_featurelist_dict, doc_label_dict = File_Reader(feature_file, label_file);

	f1_list = [];

	for run in range(0, 10):

		train_set_dict, test_set_list, test_label_list = Data_Split(doc_list, doc_featurelist_dict, doc_label_dict, percent);
		aver_f1, predict_num_cluster, true_num_cluster = Inc_ND(train_set_dict, test_set_list, test_label_list, ratio_threshold);
		f1_list.append(aver_f1);

		print 'number of true clusters is ' + str(true_num_cluster);
		print 'number of predict clusters is ' + str(predict_num_cluster);
		print;

	mean_f1 = np.mean(f1_list);
	std_f1 = np.std(f1_list);

	print 'Macro-F1 is ' + str(mean_f1) + ' ' + 'under train / test split ' + str(percent); 
	print 'standard deviation is ' + str(std_f1);
