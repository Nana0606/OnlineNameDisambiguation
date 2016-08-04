# Implement one-sweep Gibbs sampler for Beyesian non-exhaustive classification
# use clustering based mean F1 to evaluate the result
# for training and test parition, using temporal information of each paper for partition. And put more recent papers in test set.

# Author: Baichuan Zhang 

import os;
import sys;
import subprocess;
import numpy as np;
import random;
from sklearn.decomposition import NMF;
from sklearn.metrics import f1_score;
from operator import itemgetter;
import math;
from scipy import linalg;
from scipy import stats;
from scipy.special import gammaln;
from sklearn.cluster import AgglomerativeClustering;
import time;

# As the NNMF is incremental version, so we do not perform z-normalization at this point
def MF(filename, K, N):

	data_matrix = np.loadtxt(str(filename), delimiter = " ");
	dimen = data_matrix.shape;
	num_row = int(dimen[0]);
	num_col = int(dimen[1]);
	data_matrix = data_matrix.tolist();
	# sort the data_matrix based on the publication year information
	data_matrix.sort(key = lambda x: x[num_col-1]);
	class_label = np.array((np.array(data_matrix)[0:num_row, num_col-2:num_col-1]));
	label_list = map(int, class_label.T.tolist()[0]);
	year_infor = np.array((np.array(data_matrix)[0:num_row, num_col-1:num_col]));
	year_list = map(int, year_infor.T.tolist()[0]);
	
	sorted_year_list = sorted(list(set(year_list)));
	test_year_list = sorted_year_list[len(sorted_year_list)-N:]

	# construct the training set based on original binary feature
	binary_train_list = [];
	binary_test_list = [];
	train_label_list = [];
	test_label_list = [];
	for pos in range(0,len(data_matrix)):
		if year_list[pos] not in test_year_list:
			binary_train_list.append(data_matrix[pos][0:num_col-2]);
			train_label_list.append(int(label_list[pos]));
		else:
			binary_test_list.append(data_matrix[pos][0:num_col-2]);
			test_label_list.append(int(label_list[pos]));

	# on binary training set, run the NNMF to generate latent matrix U and V
	binary_train_matrix = np.array(binary_train_list);
	model = NMF(n_components = K, random_state = None);
	model.fit(binary_train_matrix);
	# U_matrix is the user latent matrix
	U_matrix = model.transform(binary_train_matrix).tolist();
	# V_matrix is a set of basis vectors
	V_matrix = model.components_;
	# from U_matrix, build train_set_dict
	num_train = int(len(U_matrix));
	train_set_dict = {};
	for tr in range(0,len(U_matrix)):
		train_label = int(train_label_list[tr]);
		if train_set_dict.get(train_label, -1) == -1:
			train_set_list = [];
			train_set_list.append(U_matrix[tr]);
			train_set_dict[train_label] = train_set_list;
		else:
			train_set_list = train_set_dict[train_label];
			train_set_list.append(U_matrix[tr]);
			train_set_dict[train_label] = train_set_list;

	# from V_matrix, construct the least square on binary_test_list and learn the latent features for each test point, build test_set
	test_set = [];
	for te in range(0,len(binary_test_list)):
		# t_vector is column vector
		t_vector = np.array([binary_test_list[te]]).T;
		temp_1 = np.linalg.inv(np.dot(V_matrix, V_matrix.T));
		temp_2 = np.dot(temp_1, V_matrix);	
		w_vector = np.dot(temp_2, t_vector).T.clip(0).tolist()[0];
		test_set.append(w_vector);

	dimen = K;
	
	return [train_set_dict, train_label_list, test_set, test_label_list, dimen, num_train];

# estimate the parameter in Normal-Normal-Invert Wishart model
def parameter_estimatet(train_set_dict, dimen, num_train, m):

	# estimate the u_0 as mean of training dataset
	# use pooled covariance matrix to estimate \Sigma_0 in multivariate student-t
	data_list_list = [];
	# initialize a K*K zero matrix
	sigma_0 = np.zeros((dimen, dimen));
	for k, v in train_set_dict.items():
		D_j = np.array(v);
		sigma_0 = sigma_0 + len(v)*np.cov(D_j.T, bias = 1);
		for i in range(0,len(v)):
			data_list_list.append(v[i]);
	u_0_list = np.mean(np.array(data_list_list), axis=0).tolist();	
	sigma_0 = (float(1)/(num_train - len(train_set_dict)))*(m-dimen-1)*sigma_0;
	return [u_0_list, sigma_0];

# use empirical Bayes by sampling a large number of samples from a Chinese Restaurant Process for picking up alpha
def estimate_alpha(num_train):
	
	alpha_list = range(20, 1000, 20);	
	# vague_prob is our prior belief of encountering a new class
	vague_prob = 0.5;
	max_diff_prob = sys.maxint;
	best_alpha = -1;
	for pos in range(0,len(alpha_list)):
		es_alpha = int(alpha_list[pos]);
		CRP_ratio = float(es_alpha)/(es_alpha + num_train);	
		active_count = 0;
		for i in range(0, 100000):
			ran_p = random.uniform(0,1);
			if ran_p <= CRP_ratio:
				active_count = active_count + 1;
		CRP_prob = float(active_count)/100000;
		diff_prob = abs(CRP_prob - vague_prob);
		if diff_prob <	max_diff_prob:
			max_diff_prob = diff_prob;
			best_alpha = es_alpha;
	return best_alpha;

# Since Numpy and Scipy package don't have existing multivariate student-t distribution function, so I implement its PDF function here based on "http://en.wikipedia.org/wiki/Multivariate_t-distribution" and compute the likelihood of given testdata. Also in order to avoid the overfloating issue of Gamma function, take the Log function of its PDF

# first parameter is the location vector (d*1), the second one is the positive definite scale matrix (d*d) and the third one is the degrees of freedom for the multivariate t distribution

def Multivariate_Student_t_likelihood(stud_t_1, stud_t_2, stud_t_3, testdata, dimen):

	x_vector = np.array(testdata).reshape(dimen,1);		
	log_det = np.log(np.linalg.det(stud_t_2));
	a = np.dot((x_vector - stud_t_1).reshape(1,dimen), np.linalg.inv(stud_t_2));
	Q = np.dot(a, (x_vector - stud_t_1));
	Q_value = Q[0];
	
	log_likelihood = gammaln(float(stud_t_3+dimen)/2) - gammaln(float(stud_t_3)/2) - (float(dimen)/2)*np.log(stud_t_3) - (float(dimen)/2)*np.log(np.pi) - 0.5*log_det - (float(stud_t_3+dimen)/2)*np.log(1+float(Q_value)/stud_t_3);

	likelihood = np.exp(log_likelihood);
	return likelihood;

# compute the mean F1 in clustering setting (chapter 17 in Zaki's data mining book)
def Compute_F1(test_label_list, predict_label_list):

	# group the papers with same predicted label from HAC together
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

# implement Normal-Normal-Inverted Wishart Model for classification
# Perform one-sweep Gibbs sampler

def Gibbs(train_set_dict, train_label_list, test_set, test_label_list, dimen, kapa, sigma_0, m, u_0_list, alpha, num_train):

	# non-exhaustive learning model
	# alpha is the parameter in Dirichlet Process
	emerge_label = 5000;
	predict_label_list = [];
	n_j = -1;
	# num_data is the total number of data being processed + number of training instance
	num_data = num_train;
	
	for pos in range(0,len(test_set)):
		
		testdata = test_set[pos];
		# Given each testdata, for training data points of each class label, compute the likelihood and select label with max likelihood as the predicted label of its testdata
		predlabel_prob_dict = {};

		for k_train, v_train in train_set_dict.items():
			predict_label = int(k_train);
			data_matrix = np.array(v_train); 
			data_num = int(len(v_train));
			x_bar_list = [];
			for col_ in range(0,data_matrix.shape[1]):
				col_sum = np.sum(data_matrix[:,col_]);
				col_mean = float(col_sum)/data_num;
				x_bar_list.append(col_mean);
			# first parameter of student-t distribution based on equation B.3
			stud_t_1 = (float(1)/(data_num + kapa))*np.add(kapa*np.array(u_0_list).reshape(dimen,1), data_num*np.array(x_bar_list).reshape(dimen,1)); 										
			# compute the second parameter of student-t distribution based on equation B.3
			front_const = float(data_num + kapa + 1)/((data_num+kapa)*(data_num+m+1-dimen));
			inner_matrix = sigma_0 + data_num*np.cov(data_matrix.T, bias = 1) + (float(data_num*kapa)/(kapa+data_num))*np.outer((np.array(u_0_list)-np.array(x_bar_list)), (np.array(u_0_list)-np.array(x_bar_list)));
			stud_t_2 = front_const * inner_matrix;

			# compute the third parameter of student-t distribution based on equation B.3
			stud_t_3 = data_num + m + 1 - dimen;
			
			# compute the likelihood of multivariate student-t distribution for the given testdata and given precomputed three parameters
			likelihood_ = Multivariate_Student_t_likelihood(stud_t_1, stud_t_2, stud_t_3, testdata, dimen);
			likelihood = (float(data_num)/(alpha+num_data))*likelihood_;

			predlabel_prob_dict[predict_label] = likelihood;

		# compute the marginal probability p(x_{n+1}) = p(x_{n+1}|D_{j}) with D_{j} = \emptyset

		marginal_prob = Multivariate_Student_t_likelihood(np.array(u_0_list).reshape(dimen, 1), (float(kapa+1)/(kapa*(m+1-dimen)))*sigma_0, m+1-dimen, testdata, dimen);
		new_cluster_prob = (float(alpha)/(alpha+num_data))*marginal_prob;
		predlabel_prob_dict[emerge_label] = new_cluster_prob;

		# sample from predlabel_prob_dict and decide the class membership information of current test point
		sample_label_list = [];
		CDF_list = [];
		CDF_list.append(0.0);
		weight = 0.0;
		for sample_k, sample_v in predlabel_prob_dict.items():
			sample_label_list.append(int(sample_k));
			weight = weight + float(sample_v);
			CDF_list.append(weight);

		# generate a random number between zero and weight
		rand_number = random.uniform(0,weight);
		for cdf in range(0,len(CDF_list)):
			if rand_number >= float(CDF_list[cdf]) and rand_number < float(CDF_list[cdf+1]):
				gibbs_label = int(sample_label_list[cdf]);		

		# update train_set_dict based on the one-sweep Gibbs sampler result
		if gibbs_label != emerge_label:
			# update the train_set_dict
			inter_list = train_set_dict[gibbs_label];	
			inter_list.append(testdata);
			train_set_dict[gibbs_label] = inter_list;
			# add the final predicted result into predict_label_list
			predict_label_list.append(gibbs_label);
		else:
			# detect an emerging class
			train_set_dict[gibbs_label] = [testdata];
			emerge_label = emerge_label + 1;
			# add emerge_label result into predict_label_list
			predict_label_list.append(gibbs_label);
		
		num_data = num_data + 1;
	
	aver_f1, predict_num_cluster, true_num_cluster = Compute_F1(test_label_list, predict_label_list);
	
	return [aver_f1, predict_num_cluster, true_num_cluster];

if __name__ == '__main__':

	filename = str(sys.argv[1]);
	K = 6;
	# put last N year's papers into test set
	N = 2;
	kapa = 100.0;
	# m > K + 2
	m = K + 100;
	t1 = time.time();
	train_set_dict, train_label_list, test_set, test_label_list, dimen, num_train = MF(filename, K, N);

	u_0_list, sigma_0 = parameter_estimatet(train_set_dict, dimen, num_train, m);
	best_alpha = estimate_alpha(num_train);
	aver_f1, predict_num_cluster, true_num_cluster = Gibbs(train_set_dict, train_label_list, test_set, test_label_list, dimen, kapa, sigma_0, m, u_0_list, best_alpha, num_train);
	test_time = time.time() - t1;
	print 'number of predicted clusters are ' + str(predict_num_cluster);	
	print 'best aver_f1 is ' + str(aver_f1);
