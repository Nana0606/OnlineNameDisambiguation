# for non-exhaustive setting, model it as anormaly detection problem
# use mean F1 to evaluate the result

# Author: Baichuan Zhang 

import os;
import sys;
import subprocess;
import numpy as np;
import random;
from sklearn.decomposition import NMF;
from sklearn.metrics import f1_score;
from operator import itemgetter;
from sklearn import svm;
from sklearn.cluster import AgglomerativeClustering;
from sklearn.naive_bayes import BernoulliNB;

# use sklearn library to perform NMF on original data matrix for each given target name
def MF(filename):

	data_matrix = np.loadtxt(str(filename), delimiter = " ");
	dimen = data_matrix.shape;
	num_row = int(dimen[0]);
	num_col = int(dimen[1]);
	
	transformed_matrix = data_matrix[0:num_row, 0:num_col-1].tolist();
	class_label = np.array(data_matrix[0:num_row, num_col-2:num_col-1]);
	label_list = map(int, class_label.T.tolist()[0]);

	year_infor = np.array(data_matrix[0:num_row, num_col-1:num_col]);
	year_list = map(int, year_infor.T.tolist()[0]);

	# group the paper based on temporal information
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
	
	dimen = num_col - 2;

	sorted_year_list = sorted(year_datalistlist_dict.keys());
	train_set_list = [];
	train_label_list = [];
	test_set_list = [];
	test_label_list = [];
	
	test_year_list = sorted_year_list[len(sorted_year_list)-4:];
	# training and test partition based on the temporal information and put more recent papers into test set
	for k, v in year_datalistlist_dict.items():
		if int(k) not in test_year_list:
			# put all the corresponding paper into training set
			for train in range(0,len(v)):
				train_set_list.append(v[train][0:dimen]);
				train_label_list.append(int(v[train][-1]));		
		else:
			# put all the corresponding paper into test set
			for test in range(0,len(v)):
				test_set_list.append(v[test][0:dimen]);
				test_label_list.append(int(v[test][-1]));

	num_cluster_test = len(list(set(test_label_list)));

	return [train_set_list, train_label_list, test_set_list, test_label_list, num_cluster_test];
		
# use sklearn for Naive classification and then report the probability output of each test instance

def Classification(train_set_list, train_label_list, test_set_list, test_label_list):

	X_train = np.array(train_set_list);
	X_test = np.array(test_set_list);
	Y_train = np.array(train_label_list);
	Y_test = np.array(test_label_list);
	
	clf = BernoulliNB();
	clf.fit(X_train, Y_train);
	prob_matrix = clf.predict_proba(X_test);	

	return [test_label_list, prob_matrix];

# compute the mean F1 in clustering setting (chapter 17 in Zaki's data mining book)
def Compute_F1(test_label_list, y_HAC_pred):

	# group the papers with same predicted label from HAC together
	predict_label_dict = {};
	for pos in range(0,len(y_HAC_pred)):
		predict_label = int(y_HAC_pred[pos]);
		if predict_label_dict.get(predict_label, -1) == -1:
			pred_paperid_list = [];
			# paper index starts from 1
			pred_paperid_list.append(pos+1);
			predict_label_dict[predict_label] = pred_paperid_list;
		else:
			pred_paperid_list = predict_label_dict[predict_label];	
			pred_paperid_list.append(pos+1);
			predict_label_dict[predict_label] = pred_paperid_list;
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

	return aver_f1;

# use the output probability from Naive Bayes classifier as meta features, and then utilize hierarchical clustering for anomaly detection

def HAC(test_label_list, prob_matrix, cluster_size):

	y_HAC_pred = AgglomerativeClustering(n_clusters=cluster_size, linkage='average', connectivity=None).fit_predict(prob_matrix);
	aver_f1 = Compute_F1(test_label_list, y_HAC_pred);
	
	return aver_f1;

if __name__ == '__main__':

	filename = str(sys.argv[1]);	
	train_set_list, train_label_list, test_set_list, test_label_list, num_cluster_test = MF(filename);
	num_true_label = num_cluster_test;

	result_file = open('result.txt', 'w');
	result_file.write('cluster_size' + '\t' + 'F1' + os.linesep);

	# tune the pre-defined cluster_size in Hierarchical Clustering and choose the one which gives best F1 score
	cluster_size_list = range(num_true_label - 10, num_true_label + 10);
	
	for i in range(0,len(cluster_size_list)):
		print str(i);
		cluster_size = int(cluster_size_list[i]);
		sum_ = 0.0;
		for run in range(0,5):
			train_set_list, train_label_list, test_set_list, test_label_list, num_cluster_test = MF(filename);
			test_label_list, prob_matrix = Classification(train_set_list, train_label_list, test_set_list, test_label_list);
			aver_f1 = HAC(test_label_list, prob_matrix, cluster_size);
			sum_ = sum_ + aver_f1;
	
		result_file.write(str(cluster_size) + '\t' + str(float(sum_)/5) + os.linesep);
