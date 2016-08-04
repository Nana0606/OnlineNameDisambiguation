# construct binary features for co-authorship, keyword, and venue based features
# Author: Baichuan Zhang

import os;
import sys;
import math;
from collections import Counter;
from os import listdir;

def Gen_Feature(filetoread, file_name):

	# construct stop word list
	stopword_file = open('stopword.txt', 'r');
	stop_list = [];
	for line in stopword_file:
		key_ = line.strip().lower();
		stop_list.append(key_);
	stop_list = list(set(stop_list));

	tf_dict_list = [];
	docu_count = 0;
	Docu_list = [];

	author_index = 0;
	nVerts = 0;
	vertlist = {};
	paper_author_list = [];
	
	paper_venue_list = [];

	paper_label_list = [];

	paper_year_list = [];

	for line in filetoread:
		line = line.strip();

		# extract target node name
		if '<FullName>' in line:
			ego_name = line[line.find('>')+1:line.rfind('<')].strip();

		# construct keyword based feature (TF-IDF)
		elif '<title>' in line:
			line = line.strip().lower();
			docu_count = docu_count + 1;
			tf_dict = {};
			line = line[line.find('>')+1:line.rfind('<')].strip();
			linetuple = line.strip().split(' ');
			temp = [];
			for pos in range(0,len(linetuple)):
				keyword = str(linetuple[pos].translate(None, ':,.@$!?;'));
				if keyword not in stop_list and keyword.isdigit() == False:
					temp.append(keyword);
					if tf_dict.get(keyword, -1) == -1:
						tf_dict[keyword] = 1;
					else:
						tf_dict[keyword] = tf_dict[keyword] + 1;
			tf_dict_list.append(tf_dict);
			Docu_list.append(list(set(temp)));

		# construct coauthor list based feature (binary)
		elif '<authors>' in line:
			index_list = [];
			coauthor_list = [];
			coauthor_list = line[line.find('>')+1:line.rfind('<')].strip().split(',');
			coauthor_list.remove(ego_name);
			# if the paper is not single author paper
			if len(coauthor_list) != 0:
				for co in range(0,len(coauthor_list)):
					if vertlist.get(coauthor_list[co].strip(), -1) == -1:
						author_index = author_index + 1;
						vertlist[coauthor_list[co].strip()] = author_index;
						index_list.append(author_index);
						nVerts = nVerts + 1;
					else:
						index_list.append(vertlist[coauthor_list[co]]);
				paper_author_list.append(sorted(index_list));
			else:
				paper_author_list.append([]);					

		# construct paper venue based feature (binary)
		elif '<jconf>' in line:
			if '(' not in line:
				line = line[line.find('>')+1:line.rfind('<')].strip().lower();
				venue_name = str(line.translate(None, ':,.@$!?;'));
				paper_venue_list.append(venue_name);
			else:
				line = line[line.find('>')+1:line.rfind('(')].strip().lower();
				venue_name = str(line.translate(None, ':,.@$!?;'));
				paper_venue_list.append(venue_name);		

		# extract the true label of each paper
		elif '<label>' in line:
			label = int(line[line.find('>')+1:line.rfind('<')].strip());
			paper_label_list.append(label);

		elif '<year>' in line:
			temporal = int(line[line.find('>')+1:line.rfind('<')].strip());
			paper_year_list.append(temporal);

	total_author_list = range(1, nVerts+1);
		
	Docu_list = sum(Docu_list, []);
	idf_dict = Counter(Docu_list);
	corpus_list = list(set(Docu_list));

	venue_collection = list(set(paper_venue_list));
	author_name = file_name[:file_name.rfind('.')].strip();
	feature_file = open("cleaned_data/" + str(author_name) + '_' + 'matrix.txt', 'w');
	
	for row in range(0,len(tf_dict_list)):
		for i in range(0,len(corpus_list)):
			key = str(corpus_list[i]);	
			if key in tf_dict_list[row]:
				feature_file.write(str(1) + ' ');
			else:
				feature_file.write(str(0) + ' ');

		# write coauthorship based binary feature into file for each paper
		for c_ in range(0,len(total_author_list)):
			coauthor_ = total_author_list[c_];
			if coauthor_ in paper_author_list[row]:
				feature_file.write(str(1) + ' ');
			else:
				feature_file.write(str(0) + ' ');

		# write the venue based feature for each paper into file
		for v_ in range(0,len(venue_collection)):
			venue_ = str(venue_collection[v_]);
			if venue_ != 'null':
				if venue_  == paper_venue_list[row]:
					feature_file.write(str(1) + ' ');
				else:
					feature_file.write(str(0) + ' ');

		# write the true label of each paper into file
		feature_file.write(str(paper_label_list[row]) + ' ');		
		feature_file.write(str(paper_year_list[row]) + os.linesep);

if __name__ == '__main__':
	
	path = 'original_data/';
	allfiles = listdir(path);
	for pos in range(0,len(allfiles)):
		file_name = allfiles[pos];
		filetoread = open(path + allfiles[pos], 'r');
		Gen_Feature(filetoread, file_name);
