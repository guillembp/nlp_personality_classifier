# coding: utf-8
import numpy as np
import pandas as pd
import re
import os
import pickle
import utils
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer, VectorizerMixin
from sklearn.feature_extraction.text import strip_accents_unicode
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, average_precision_score, f1_score, recall_score, classification_report
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, cross_validate, GridSearchCV, cross_val_predict, RandomizedSearchCV
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
import matplotlib.pyplot as plt
from scipy.stats import randint as sp_randint
from scipy.stats import uniform
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix


path = "/Users/guillembp/Dropbox/Thesis/NLP/"
os.chdir(path)
os.getcwd()


'''Data loading and shaping'''

def LoadData():

	with open('/Users/guillembp/Desktop/data/jobScreening_cvpr17/train/transcription_training.pkl', 'rb') as f:
		transcription_training = pickle.load(f, encoding='latin1')
	# len(transcription_training)
	# transcription_training

	with open('/Users/guillembp/Desktop/data/jobScreening_cvpr17/train/annotation_training.pkl', 'rb') as g:
		annotation_training = pickle.load(g, encoding='iso-8859-1')
	annotation_training = annotation_training['interview']
	# len(annotation_training)
	# annotation_training

	with open('/Users/guillembp/Desktop/data/jobScreening_cvpr17/validation/transcription_validation.pkl', 'rb') as f:
	    transcription_validation = pickle.load(f, encoding='latin1')
	# len(transcription_validation)
	# transcription_validation

	with open('/Users/guillembp/Desktop/data/jobScreening_cvpr17/validation/annotation_validation.pkl', 'rb') as g:
	    annotation_validation = pickle.load(g, encoding='iso-8859-1')
	annotation_validation = annotation_validation['interview']
	# len(annotation_validation)
	# annotation_validation

	with open('/Users/guillembp/Desktop/data/jobScreening_cvpr17/test/transcription_test.pkl', 'rb') as f:
		transcription_testing = pickle.load(f, encoding='latin1')
	# len(transcription_testing)
	# transcription_testing

	with open('/Users/guillembp/Desktop/data/jobScreening_cvpr17/test/annotation_test.pkl', 'rb') as g:
		annotation_testing = pickle.load(g, encoding='iso-8859-1')
	annotation_testing = annotation_testing['interview']
	# len(annotation_testing)
	# annotation_testing

	return transcription_training, annotation_training, transcription_testing, annotation_testing, transcription_testing, annotation_testing

'''lists to feed into the model'''
def list_to_trainable(key_list,X,y):
	''' Create dataframe from lists '''
	df = pd.DataFrame(
	{'key_list': key_list,
	'X': X,
	'y': y
	})
	return df

''' Create dataframe '''
def dataPreparation(transcription, annotation):
	key_list = list()
	for k,v in sorted(transcription.items()):
		key_list.append(k)
	# key_list
	# len(key_list)

	X = list()
	for k,v in sorted(transcription.items()):
		X.append(v)
	# X
	# len(X)

	y = list()
	for k,v in sorted(annotation.items()):
		y.append(float(v))
	# y
	# len(y)

	return key_list, X, y


'''Calculate thresholds for dichotomic label'''
def ExtractParameters(y):

	mean = np.mean(y, axis=0)
	# mean

	median = np.median(y, axis=0)
	# median

	sigma = np.std(y_train)
	# sigma

	return mean, median, sigma


''' Delete rows that fall inside the threshold '''
def dichotomizer(x, col, middle):
	if x[col] < middle:
		return 0
	elif x[col] >= middle:
		return 1

def DichotomizeLabel(df_trainning,df_testing,df_validating,middle,slider,val_threshold):
	''' Discards label ambiguity and converts to binary '''
	df_training_filtered = df_trainning.drop(df_trainning[(df_trainning.y>=(middle-sigma*slider)) & (df_trainning.y<(middle+sigma*slider))].index)

	df_testing_filtered = df_testing.drop(df_testing[(df_testing.y>=(middle-sigma*slider)) & (df_testing.y<(middle+sigma*slider))].index)

	df_validating_filtered = df_validating.drop(df_validating[(df_validating.y>=(middle-sigma*val_threshold)) & (df_validating.y<(middle+sigma*val_threshold))].index)

	df_training_filtered["y_dichotomic"] = df_training_filtered.apply(lambda x: dichotomizer(x,'y',middle), axis=1)
	df_training_filtered

	df_validating_filtered["y_dichotomic"] = df_validating_filtered.apply(lambda x: dichotomizer(x,'y',middle), axis=1)
	df_validating_filtered

	df_testing_filtered["y_dichotomic"] = df_testing_filtered.apply(lambda x: dichotomizer(x,'y',middle), axis=1)
	df_testing_filtered

	# train_size_0 = len(df_training_filtered[df_training_filtered["y_dichotomic"]==0])
	# train_size_1 = len(df_training_filtered[df_training_filtered["y_dichotomic"]==1])
	#
	# train_size_0 = len(df_validating_filtered[df_validating_filtered["y_dichotomic"]==0])
	# train_size_1 = len(df_validating_filtered[df_validating_filtered["y_dichotomic"]==1])
	#
	# test_size_0 = len(df_testing_filtered[df_testing_filtered["y_dichotomic"]==0])
	# test_size_1 = len(df_testing_filtered[df_testing_filtered["y_dichotomic"]==1])

	# # Plotting the remainig documents by label value.
	# ptrain = plt.figure()
	# plt.hist(df_training_filtered.y_train, bins=30)
	# plt.xlabel('Label values (interview)')
	# plt.ylabel('Frequency')
	# plt.axvline(x=middle+sigma*slider, color='k', linestyle='--')
	# plt.axvline(x=middle-sigma*slider, color='k', linestyle='--')
	# plt.text(0,0.7,'%d Documents' % train_size_0,rotation=0)
	# plt.text(0.66,0.7,'%d Documents' % train_size_1,rotation=0)
	# plt.show()
	# ptrain.savefig('training_histogram/%f_training_histogram.pdf' % slider, format='pdf', dpi=400)
	#
	# ptest = plt.figure()
	# plt.hist(df_testing_filtered.y_test, bins=30)
	# plt.xlabel('Label values (interview)')
	# plt.ylabel('Frequency')
	# plt.axvline(x=middle+sigma*slider, color='k', linestyle='--')
	# plt.axvline(x=middle-sigma*slider, color='k', linestyle='--')
	# plt.text(0,1,'%d Documents' % test_size_0,rotation=0)
	# plt.text(0.66,1,'%d Documents' % test_size_1,rotation=0)
	# plt.show()
	# ptest.savefig('testing_histogram/%f_test_histogram.pdf' % slider, format='pdf', dpi=400)
	#

	X_train = df_training_filtered['X']
	X_test = df_testing_filtered['X']
	X_validate = df_validating_filtered['X']
	y_train = df_training_filtered['y_dichotomic']
	y_test = df_testing_filtered['y_dichotomic']
	y_validate = df_validating_filtered['y_dichotomic']
	return X_train, X_test, X_validate, y_train, y_test, y_validate


def scorer(y, p):
	d = {'precision': precision_score(y, p),'recall': recall_score(y, p),'F1-weighted':f1_score(y, p, average='weighted'),'avgPreciScore': average_precision_score(y, p),'auc':roc_auc_score(y, p)}
	return pd.DataFrame(d, index=['score'])

def get_missclassified_docs(X, y_dct, preds, y_cont):
	""" Creates a DataFrame with false negatives and false positives """
	df = pd.DataFrame({'text': X, 'prediction': preds, 'dct_label': y_dct, 'cont_label': y_cont})
	errors = df[df.dct_label != df.prediction]
	return (errors[errors.dct_label==False], errors[errors.dct_label==True])

def ngrammer(tokens, ngram_range):
	mix = VectorizerMixin()
	mix.ngram_range = ngram_range
	return mix._word_ngrams(tokens)

def tokenizer(s, ngram_range = (1,4)):
	s = re.sub(r'\d+', '', s).strip() #remove numbers
	s = s.replace('\n', ' ').replace('\r', '') #remove line breaks
	s = re.sub("\S*\d\S*", "", s).strip() # remove possible mistakenly encoded characters
	s = s.lower() # Make lowercase
	pattern = re.compile(r"(?u)\b\w\w+\b")
	unigrams = pattern.findall(s)
	unigrams = [u for u in unigrams if u not in ENGLISH_STOP_WORDS]
	tokens = ngrammer(unigrams, ngram_range)
	return tokens

def create_vectors_CV(X_train, X_test, analyzer = tokenizer):
	X = pd.concat([X_train, X_test])
	vectorizer = TfidfVectorizer(analyzer=tokenizer).fit(X)
	vector = vectorizer.transform(X)
	return vector, vectorizer

def get_sample_weights(y):
	'''Parabolic weighting'''
	y_weights = list()
	mean, median, sigma = ExtractParameters(y_train)
	min_weight = 0.3
	c = 1.0
	a = (min_weight-c)/(mean**2-mean)
	for x in y:
		y_weights.append(a*x**2-a*x+c)
	return y_weights

''' MAIN '''

slider = 0.0 #np.linspace(0,2.5,100).tolist()
val_threshold = 0.0
score = list()

transcription_training, annotation_training, transcription_testing, annotation_testing, transcription_validating, annotation_validating = LoadData()

key_list_train, X_train, y_train  = dataPreparation(transcription_training, annotation_training, )
key_list_test, X_test, y_test = dataPreparation(transcription_testing, annotation_testing)
key_list_validate, X_validate, y_validate = dataPreparation(transcription_validating, annotation_validating)

y_all_data_cont = y_train+y_test

df_trainning = list_to_trainable(key_list_train, X_train, y_train)
df_testing = list_to_trainable(key_list_test, X_test, y_test)
df_validating = list_to_trainable(key_list_validate, X_validate, y_validate)

mean, median, sigma = ExtractParameters(y_train)

X_train, X_test, X_validate, y_train_dct, y_test_dct, y_validate_dct = DichotomizeLabel(df_trainning, df_testing, df_validating, mean, slider, val_threshold)

''' Transform and tokenize training and test data for CV '''
y_all_data = pd.concat([y_train_dct,y_test_dct])
X_all_data = pd.concat([X_train, X_test])
V_all_data_and_validate, vectorizer = create_vectors_CV(X_all_data, X_validate)


''' Dimensionality reduction PCA (truncated SVD) with TFIDF matrix (Latent Semantic Analysis)  '''
lsa = TruncatedSVD(n_components=113, n_iter=50, random_state=1337) # n_components = roughly sqrt(cols) of the tfidf matrix
lsa.fit(V_all_data_and_validate) #Check for the "tranform" of the matrix

''' Split back the matrix to train+test and validate  '''
len_train = X_all_data.shape[0]
len_val = X_validate.shape[0]
V_all_data = V_all_data_and_validate[0:len_train]
V_validate = V_all_data_and_validate[len_train:]

''' Naive Bayes'''
model = MultinomialNB(fit_prior=True)
model.get_params()
param_dist = {"alpha":sp_randint(2*10**(-2), 20)}

''' Support Vector Classifier '''
# For the fitting of the classifier, set the number of iterations to 10 and the scoring to default.
# model = svm.SVC()
# model.get_params()
# param_dist = {"C": sp_randint(2*10**(-1), 2000),
#               "gamma": sp_randint(2*10**(-1), 200)}
# Gamma in the rbf Kernel is the curvature of the decision boundary, i.e. how strict is the decision boundary
# C is the penalty for misclassification. C small => High bias, low variance; C large => Low bias, high variance.

''' Ridge Regression Classifier '''
#
# model = RidgeClassifier(solver='sag')
# model.get_params()
# param_dist = {"alpha": sp_randint(2*10**(-3), 200)}

''' Crossvalidation and Randomized Parameter Search '''
stratified_folds = StratifiedKFold(n_splits=5, shuffle=True) #The folds are made by preserving the percentage of samples for each class.
# parameters = {'kernel':('linear', 'rbf'), 'C':[0.01, 10], 'gamma':[0.0001,1]}
clf = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_jobs=4, cv=stratified_folds, verbose = 3) #, scoring='f1_weighted', n_iter=10

''' Sample weights '''
y_weights = get_sample_weights(y_all_data_cont)
plt.hist(y_weights, bins=30)
clf.fit(X=V_all_data, y=y_all_data, sample_weight=y_weights)
clf.best_params_ # Parameter results from the Randomized Search

''' RESULTS. Use the fitted, crossvalidated and optimal-parameter model to predict the validate output '''
preds = clf.predict(V_validate)

# By definition a confusion matrix C is such that C_{i, j} is equal to the number of observations known to be in group i but predicted to be in group j.
pd.DataFrame(confusion_matrix(y_true=y_validate_dct, y_pred=preds))
scorer(y_validate_dct,preds)
print(classification_report(y_validate_dct, preds))


df = pd.DataFrame({'text': X_validate, 'prediction': preds, 'dct_label': y_validate_dct, 'cont_label': y_validate})
errors = df[df.dct_label != df.prediction]
false_pos, false_neg = (errors[errors.dct_label==False], errors[errors.dct_label==True])

#
# false_pos, false_neg = get_missclassified_docs(X_validate, y_validate_dct, preds, y_validate)
# false_pos.to_csv(path_or_buf=path+'/missclassified/false_pos.csv')
# false_neg.to_csv(path_or_buf=path+'/missclassified/false_neg.csv')
#

# PLOT!
# p = plt.figure()
# plt.plot(slider, score)
# plt.xlabel('Sigma')
# plt.ylabel('ROC Score')
# plt.axvline(x=0.275, color='k', linestyle='--')
# plt.text(0.3,0.62,'Optimal Sigma = 0.275',rotation=0)
# plt.show()
# p.savefig('180531_1_Sigma_fixed_test.pdf', format='pdf', dpi=400)
#
# results = pd.concat([pd.DataFrame(slider),pd.DataFrame(score)], axis = 1)
# results.head(72)
