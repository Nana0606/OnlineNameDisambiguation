# implement the idea in the paper titled "online name disambiguation with contraints"
# For fair comparison, use groundtruth class label during the batch mode

# Author: Baichuan Zhang

import os;
import sys;
import math;
import numpy as np;
from scipy import linalg;
from scipy import stats;
from numpy import product;
from sklearn.decomposition import NMF;
from scipy.spatial import distance;

# partition the training and test set based on temporal information of each paper

def data_processing(filename, recent_year, K):

	data_matrix = np.loadtxt(str(filename), delimiter = " ");
	dimen = data_matrix.shape;
	num_row = int(dimen[0]);
	num_col = int(dimen[1]);
	f_matrix = data_matrix[0:num_row, 0:num_col-2].tolist();
	class_label = np.array(data_matrix[0:num_row, num_col-2:num_col-1]);
	label_list = map(int, class_label.T.tolist()[0]);

	year_infor = np.array(data_matrix[0:num_row, num_col-1:num_col]);
	year_list = map(int, year_infor.T.tolist()[0]);

	# do NMF on f_matrix
	model = NMF(n_components = K, random_state = None);
	model.fit(f_matrix);
	W_matrix = model.transform(f_matrix);

	transformed_matrix = np.hstack((W_matrix, class_label)).tolist();

	# group the papers based on temporal order information
	year_datalistlist_dict = {};
        for index in range(0,len(year_list)):
        	if year_datalistlist_dict.get(int(year_list[index]), -1) == -1:
                        datalistlist = [];
                        datalistlist.append(transformed_matrix[index]);
                        year_datalistlist_dict[year_list[index]] = datalistlist;
                else:
                        datalistlist = year_datalistlist_dict[year_list[index]]
                        datalistlist.append(transformed_matrix[index]);
                        year_datalistlist_dict[year_list[index]] = datalistlist;
	
	sorted_year_list = sorted(year_datalistlist_dict.keys());

	train_set_dict = {};
	train_label_list = [];
	test_set = [];
	test_label_list = [];
	
	test_year_list = sorted_year_list[len(sorted_year_list)-recent_year:];
	# training and test partition based on the temporal information and put more recent papers into test set
	for k, v in year_datalistlist_dict.items():
		if int(k) not in test_year_list:
			# put all the corresponding paper into training set
			for train in range(0,len(v)):
				train_label = int(v[train][-1]);
				if train_set_dict.get(train_label, -1) == -1:
					train_set_list = [];
					train_set_list.append(v[train][0:K]);
					train_set_dict[train_label] = train_set_list;
				else:
					train_set_list = train_set_dict[train_label];
					train_set_list.append(v[train][0:K]);
					train_set_dict[train_label] = train_set_list;

				train_label_list.append(train_label);		
		else:
			# put all the corresponding paper into test set
			for test in range(0,len(v)):
				test_set.append(v[test][0:K]);
				test_label_list.append(int(v[test][-1]));

	return [train_set_dict, train_label_list, test_set, test_label_list];

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

def Inc_ND(train_set_dict, test_set, test_label_list, epsilon, ratio_theshold):
	
	# initialize emerging class label
	emerge_label = 5000;
	predict_label_list = [];
	for pos in range(0,len(test_set)):
		one_paper = test_set[pos];
		cluster_ratio_dict = {};
		for k1, v1 in train_set_dict.items():
			dist_list = [];
			for v1_ in range(0,len(v1)):
				# compute the euclidean distance between given test paper and every paper in each class
				dist = distance.euclidean(one_paper, v1[v1_]);
				dist_list.append(dist);
			# compute the number of points in dist_list which are larger than pre-defined threshold epsilon
			ratio = float(sum(1 for i in dist_list if i >= epsilon))/(len(dist_list));
			cluster_ratio_dict[k1] = ratio;

		# as we use supervised setting in the batch mode, so no merging cluster operation during the online setting	
		# return the highest ratio value with the class_id information and the ratio value represents the density of the \epsilon-neighbor of test data point
				
		max_class_id = max(cluster_ratio_dict, key = cluster_ratio_dict.get);
		highest_ratio = float(cluster_ratio_dict[max_class_id]);
		if highest_ratio < ratio_threshold:
			train_set_dict[emerge_label] = [one_paper];
			emerge_label = emerge_label + 1;
			predict_label_list.append(emerge_label);
		
		else: 
			temp_list = train_set_dict[max_class_id];					
			temp_list.append(one_paper);
			train_set_dict[max_class_id] = temp_list;
			predict_label_list.append(max_class_id);
	
	aver_f1, predict_num_cluster, true_num_cluster = Compute_F1(test_label_list, predict_label_list);	

	return [aver_f1, predict_num_cluster, true_num_cluster];		

			
if __name__ == '__main__':

	filename = str(sys.argv[1]);
	recent_year = int(sys.argv[2]);
	K = int(sys.argv[3]);
	train_set_dict, train_label_list, test_set, test_label_list = data_processing(filename, recent_year, K);
	
	epsilon = float(sys.argv[4]);
	ratio_threshold = float(sys.argv[5]);
	aver_f1, predict_num_cluster, true_num_cluster = Inc_ND(train_set_dict, test_set, test_label_list, epsilon, ratio_threshold);
