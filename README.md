# OnlineNameDisambiguation

## Publication

* [Baichuan Zhang](https://www.linkedin.com/in/baichuan-zhang-7649004a), Murat Dundar, and Mohammad Al Hasan: [Bayesian Non-Exhaustive Classification A Case Study: Online Name Disambiguation using Temporal Record Streams](http://arxiv.org/abs/1607.05746), to appear in CIKM 2016 Proceedings of the 25th ACM International Conference on Information and Knowledge Management, Indianapolis, IN. 

## Requirements

We will take Ubuntu for example.

* python 2.7
```
$ sudo apt-get install python
```
* numpy
```
$ sudo apt-get install pip
$ sudo pip install numpy
```
* scipy
```
$ sudo pip install scipy
```
* scikit-learn
```
$ sudo pip install sklearn
```
* CVXOPT
```
$ pip install cvxopt --user
```

## Running Guidance:
* Copy each name entity reference from cleaned_data/ directory into Bayesian_Nonexhaustive_Model/ directory
* Run the command: "python one_sweep_gibbs_sampler.py author_file_name"
* Output will be number of generated clusters in Bayesian non-exhaustive framework and mean-F1 result for each name entity reference
* Users can tune the hyper-parameters in the directory of "Bayesian_Nonexhaustive_Model/one_sweep_gibbs_sampler.py" based on the experimental section of the published paper
