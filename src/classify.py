"""
Module containing various functions for classification (on top of the learned embeddings)
mainly useful for providing convenience functions on common benchmark classification methods

TODO: Add more "complicated" classifiers to eke out the next classification benchmark
"""
import json

# Sklearn SVC (for "fair" comparison with existing methods)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV

from random import randint
import numpy as np
import logging

from utils import get_files, get_class_labels

logger = logging.getLogger()
logger.setLevel("INFO")

def subgraph2vec_tokenizer(s):
	"""
	Tokenize the string from subgraph2vec sentence (i.e. target context1 context2 ...) Just target is to be used and context strings are ignored
	:param s: context of graph2vec file
	:return: List of targets from graph2vec file.
	"""
	return [line.split(' ')[0] for line in s.split('\n')]

def linear_svm_classify(X_train, X_test, Y_train, Y_test):
	"""
	Classifier with graph embeddings
	:param X_train: training feature vectors
	:param X_test: testing feature vectors
	:param Y_train: training set labels
	:param Y_test: test set labels
	:return: None
	"""
	params = {'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000]}
	if len(set(Y_train)) == 2:
		classifier = GridSearchCV(LinearSVC(max_iter=100000000), params, cv=10, scoring='f1', verbose=1, n_jobs=4)
	else:
		classifier = GridSearchCV(LinearSVC(max_iter=100000000), params, cv=10, scoring='f1_weighted', verbose=1, n_jobs=4)
	classifier.fit(X_train, Y_train)
	logging.info('best classifier models hyperparameters', classifier.best_params_)

	Y_pred = classifier.predict(X_test)

	acc = accuracy_score(Y_test, Y_pred)
	logging.info('Linear SVM accuracy: {}'.format(acc))

	report = classification_report(Y_test, Y_pred)
	logging.info(report)

	precision, recall, fbeta_score, support = precision_recall_fscore_support(Y_test, Y_pred)

	return (acc, precision, recall, fbeta_score)

def rbf_svm_classify(X_train, X_test, Y_train, Y_test):
	"""
	Classifier with graph embeddings
	:param X_train: training feature vectors
	:param X_test: testing feature vectors
	:param Y_train: training set labels
	:param Y_test: test set labels
	:return: None
	"""
	params = {'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000]}
	if len(set(Y_train)) == 2:
		classifier = GridSearchCV(SVC(gamma="scale"), params, cv=10, scoring='f1', verbose=1, n_jobs=12)
	else:
		classifier = GridSearchCV(SVC(gamma="scale"), params, cv=10, scoring='f1_weighted', verbose=1, n_jobs=12)
	classifier.fit(X_train, Y_train)

	Y_pred = classifier.predict(X_test)

	acc = accuracy_score(Y_test, Y_pred)

	# report = classification_report(Y_test, Y_pred)
	# logging.info(report)

	precision, recall, fbeta_score, support = precision_recall_fscore_support(Y_test, Y_pred)

	return (acc, precision, recall, fbeta_score)

def perform_classification(corpus_dir, extension, embedding_fname, class_labels_fname):
	"""
	Perform classification from 
	:param corpus_dir: folder containing subgraph2vec sentence files
	:param extension: extension of the subgraph2vec sentence files
	:param embedding_fname: file containing subgraph vectors in word2vec format
	:param class_labels_fname: files containing labels of each graph
	:return:None
	"""

	# weisfeiler lehman kernel files
	wlk_files = get_files(corpus_dir, extension)

	Y = np.array(get_class_labels(wlk_files, class_labels_fname))
	logging.info('Y (label) matrix shape: {}'.format(Y.shape))

	seed = randint(0,1000)
	with open(embedding_fname, 'r') as fh:
		graph_embedding_dict = json.load(fh)
	X = np.array([graph_embedding_dict[fname] for fname in wlk_files])

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = seed)
	logging.info('Training and Test Matrix Shapes: {}. {}. {}. {} '.format(X_train.shape , X_test.shape, Y_train.shape, Y_test.shape))

	scores = rbf_svm_classify(X_train, X_test, Y_train, Y_test)
	return scores

def cross_val_accuracy(corpus_dir, extension, embedding_fname, class_labels_fname, cv=10, mode=None):
	"""
	Performs 10 (default) fold cross validation, returns the mean accuracy and associated 
	standard deviation

	:param corpus_dir: folder containing subgraph2vec sentence files
	:param extension: extension of the subgraph2vec sentence files
	:param embedding_fname: file containing subgraph vectors in word2vec format
	:param class_labels_fname: files containing labels of each graph
	:param cv: integer stating number of folds and therefore experiments to carry out
	"""
	# our accuracies
	acc_results = []

	# weisfeiler lehman kernel files
	wlk_files = get_files(corpus_dir, extension)

	Y = np.array(get_class_labels(wlk_files, class_labels_fname))
	logging.info('Y (label) matrix shape: {}'.format(Y.shape))

	for i in range(cv):
		seed = randint(0,1000)
		with open(embedding_fname, 'r') as fh:
			graph_embedding_dict = json.load(fh)
		X = np.array([graph_embedding_dict[fname] for fname in wlk_files])

		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = seed)

		if mode == "linear":
			scores = linear_svm_classify(X_train, X_test, Y_train, Y_test)
		else:
			scores = rbf_svm_classify(X_train, X_test, Y_train, Y_test)
		
		acc_results.append(scores[0])

	return np.mean(acc_results), np.std(acc_results)
