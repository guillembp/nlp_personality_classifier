# coding: utf-8
import numpy as np
import pandas as pd
import re
import os
import pickle
import utils
from sklearn.feature_extraction.text import TfidfVectorizer, VectorizerMixin
from sklearn.feature_extraction.text import strip_accents_unicode
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, average_precision_score, f1_score, recall_score, classification_report
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, cross_validate, cross_val_predict, GridSearchCV
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import matplotlib.pyplot as plt
from scipy.stats import randint as sp_randint
from scipy.stats import uniform, skewnorm
from sklearn.decomposition import TruncatedSVD, PCA
import seaborn as sns


path = "/Users/guillembp/Dropbox/Thesis/NLP/"
os.chdir(path)
os.getcwd()

def LoadDocs():
	'''Docs loading'''
	with open('/Users/guillembp/Desktop/data/jobScreening_cvpr17/train/transcription_training.pkl', 'rb') as f:
		transcription_training = pickle.load(f, encoding='latin1')
	# len(transcription_training)
	# transcription_training
	with open('/Users/guillembp/Desktop/data/jobScreening_cvpr17/validation/transcription_validation.pkl', 'rb') as f:
	    transcription_validation = pickle.load(f, encoding='latin1')
	# len(transcription_validation)
	# transcription_validation
	with open('/Users/guillembp/Desktop/data/jobScreening_cvpr17/test/transcription_test.pkl', 'rb') as f:
		transcription_testing = pickle.load(f, encoding='latin1')
	# len(transcription_testing)
	# transcription_testing
	return transcription_training, transcription_testing, transcription_testing


def LoadLabels(trait):
	'''Traits loading and shaping'''
	with open('/Users/guillembp/Desktop/data/jobScreening_cvpr17/train/annotation_training.pkl', 'rb') as g:
		annotation_training = pickle.load(g, encoding='iso-8859-1')
	annotation_training.keys()
	annotation_training = annotation_training[trait]
	# len(annotation_training)
	# annotation_training
	with open('/Users/guillembp/Desktop/data/jobScreening_cvpr17/test/annotation_test.pkl', 'rb') as g:
		annotation_testing = pickle.load(g, encoding='iso-8859-1')
	annotation_testing = annotation_testing[trait]
	# len(annotation_testing)
	# annotation_testing
	with open('/Users/guillembp/Desktop/data/jobScreening_cvpr17/validation/annotation_validation.pkl', 'rb') as g:
	    annotation_validation = pickle.load(g, encoding='iso-8859-1')
	annotation_validation = annotation_validation[trait]
	# len(annotation_validation)
	# annotation_validation
	return annotation_training, annotation_testing, annotation_validation

def listToDataframe(key_list,X,y):
	''' Create dataframe from lists '''
	df = pd.DataFrame(
	{'key_list': key_list,
	'X': X,
	'y': y
	})
	return df

def sliceDictionary(transcription, annotation):
	''' Make lists out of the dictionaries '''

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


def extractStatParameters(y):
	'''Calculate thresholds for dichotomic label'''

	mean = np.mean(y, axis=0)
	# mean

	median = np.median(y, axis=0)
	# median

	sigma = np.std(y, axis=0)
	# sigma

	return mean, median, sigma


def dichotomize(x, col, middle):
	''' Delete rows that fall inside the threshold '''
	if x[col] < middle:
		return 0
	elif x[col] >= middle:
		return 1

def DichotomizeLabel(df_trainning,df_testing,df_validating,middle,slider,val_threshold):
	''' Discards label ambiguity and converts to binary '''
	df_training_filtered = df_trainning.drop(df_trainning[(df_trainning.y>=(middle-sigma*slider)) & (df_trainning.y<(middle+sigma*slider))].index)

	df_testing_filtered = df_testing.drop(df_testing[(df_testing.y>=(middle-sigma*slider)) & (df_testing.y<(middle+sigma*slider))].index)

	df_validating_filtered = df_validating.drop(df_validating[(df_validating.y>=(middle-sigma*val_threshold)) & (df_validating.y<(middle+sigma*val_threshold))].index)

	df_validating_filtered["y_dichotomic"] = df_validating_filtered.apply(lambda x: dichotomize(x,'y',middle), axis=1)
	# df_validating_filtered
	df_training_filtered["y_dichotomic"] = df_training_filtered.apply(lambda x:
	dichotomize(x,'y',middle), axis=1)
	# df_training_filtered
	df_testing_filtered["y_dichotomic"] = df_testing_filtered.apply(lambda x: dichotomize(x,'y',middle), axis=1)
	# df_testing_filtered

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

def getMissclassifiedDocs(X, y_dct, preds, y_cont):
	""" Creates a DataFrame with false negatives and false positives """
	df = pd.DataFrame({'text': X, 'prediction': preds, 'dct_label': y_dct, 'cont_label': y_cont})
	errors = df[df.dct_label != df.prediction]
	return (errors[errors.dct_label==False], errors[errors.dct_label==True])

def ngrammer(tokens, ngram_range):
	mix = VectorizerMixin()
	mix.ngram_range = ngram_range
	return mix._word_ngrams(tokens)

def tokenizer(s, ngram_range = (1,1)):
	s = re.sub(r'\d+', 'nnuummbbeerr', s).strip() #flag numbers
	s = re.sub('!', 'eexxccllaammaattiioonn', s).strip() #flag exclamation marks
	s = s.replace('\n', ' ').replace('\r', '') #remove line breaks
	s = re.sub("\S*\d\S*", "wweeiirrddchar", s).strip() # flag mistakenly encoded characters
	s = s.lower() # Make lowercase
	pattern = re.compile(r'(?u)\b\w\w+\b')
	unigrams = pattern.findall(s)
	# unigrams = [u for u in unigrams if u not in ENGLISH_STOP_WORDS]
	tokens = ngrammer(unigrams, ngram_range)
	return tokens

def createVectorsCV(X_train, X_test, analyzer = tokenizer):
	X = pd.concat([X_train, X_test])
	vectorizer = TfidfVectorizer(analyzer=tokenizer).fit(X)
	vector = vectorizer.transform(X)
	return vector, vectorizer

def getSampleWeights(y_input,min_w):#,mean):
	y_weights = list()

	'''Parabolic weighting'''
	c = 1.0
	a = (min_w-c)/(mean**2-mean)
	for x in y_input:
		y_weights.append(a*x**2-a*x+c)

	''' 1-Normal Weighting + class balance if not instrinsic to the model'''
	# mean = np.array(y_input).mean()
	# std = np.array(y_input).std()
	# zeroes = sum(y_input<mean)
	# oneses = sum(y_input>mean)
	# bal = oneses/(oneses+zeroes)
	# for x in y_input:
	# 	y_weights.append(((2.0*3.14159*std**2)**(-1/2))*2.7182818**(-(x-mean)**2/(2.0*std**2)))
	# 	# if x < mean:
	# 	# 	y_weights.append((bal)*((2.0*3.14159*std**2)**(-1/2))*2.7182818**(-(x-mean)**2/(2.0*std**2)))
	# 	# if x > mean:
	# 	# 	y_weights.append((1-bal)*((2.0*3.14159*std**2)**(-1/2))*2.7182818**(-(x-mean)**2/(2.0*std**2)))
	# y_weights = list(1.0-y_weights/(max(y_weights)/(1.0-min_w))) #normalize
	return np.array(y_weights)

# Draw
# g = sns.pointplot(x=y_train_test_CONT,y=y_weights_opt, label_size=20)
# sns.set_style(style="white")
# g.set(xticks=[], xlabel='label values', ylabel='weights')#, font_scale = 2)
# # plt.show()
# plt.savefig('180630_ParabolicWeighting_example.pdf', format='pdf', dpi=400)
#
# os.getcwd()
#
# len(y_train_test_CONT[80::800])

#len(y_weights)

''' MAIN '''''' MAIN '''''' MAIN '''''' MAIN '''''' MAIN '''''' MAIN '''''' MAIN '''''' MAIN '''

slider = 0.0 #np.linspace(0,2.5,100).tolist()
val_threshold = 0.0
score = list()

''' Documents '''
transcription_training, transcription_testing, transcription_validating = LoadDocs()
key_list_train = list()
X_train = list()
for k,v in sorted(transcription_training.items()):
	key_list_train.append(k)
	X_train.append(v)
key_list_test = list()
X_test = list()
for k,v in sorted(transcription_testing.items()):
	key_list_test.append(k)
	X_test.append(v)
key_list_validate = list()
X_validate =list()
for k,v in sorted(transcription_validating.items()):
	key_list_validate.append(k)
	X_validate.append(v)

''' Multiple labels '''
trait_list = list(['extraversion',
					'neuroticism',
					'agreeableness',
					'conscientiousness',
					'openness',
					'interview'])

f1_df = list()
minw_param_trait_optimal = list()
classified_docs = pd.DataFrame()
classified_docs['id'] = key_list_validate
classified_docs['X_validate'] = X_validate
for trait in trait_list:
	print('\n'+trait)
	annotation_training, annotation_testing, annotation_validation = LoadLabels(trait)

	y_train_CONT =list()
	for k,v in sorted(annotation_training.items()):
		y_train_CONT.append(v)
	y_test_CONT =list()
	for k,v in sorted(annotation_testing.items()):
		y_test_CONT.append(v)
	y_validate_CONT =list()
	for k,v in sorted(annotation_validation.items()):
		y_validate_CONT.append(v)

	y_train_test_CONT = y_train_CONT+y_test_CONT

	df_trainning = listToDataframe(key_list_train, X_train, y_train_CONT)
	df_testing = listToDataframe(key_list_test, X_test, y_test_CONT)
	df_validating = listToDataframe(key_list_validate, X_validate, y_validate_CONT)

	mean, median, sigma = extractStatParameters(y_train_CONT)

	X_train, X_test, X_validate, y_train_DCT, y_test_DCT, y_validate_DCT = DichotomizeLabel(df_trainning, df_testing, df_validating, mean, slider, val_threshold)



	''' Transform and tokenize training and test data for CV '''
	y_train_test_DCT = pd.concat([y_train_DCT,y_test_DCT])
	X_train_test = pd.concat([X_train, X_test])
	V_train_test_validate, vectorizer = createVectorsCV(X_train_test, X_validate)

	len_train = X_train.shape[0]
	len_train_test = X_train_test.shape[0]
	len_val = X_validate.shape[0]

	V_train = V_train_test_validate[0:len_train]
	V_test = V_train_test_validate[len_train:len_train_test]
	V_train_test = V_train_test_validate[0:len_train_test]


	''' Dimensionality reduction PCA (truncated SVD) with TFIDF matrix (Latent Semantic Analysis) '''
	# lsa = TruncatedSVD(n_components=114, algorithm = 'arpack', random_state=1337) # n_components = roughly sqrt(cols) of the tfidf matrix
	# # lsa = PCA(n_components=114, svd_solver = 'arpak', random_state = 1337) # Only non sparse matrix
	# V_train_test_validate_LSA = lsa.fit_transform(V_train_test_validate)
	# V_train_test_validate_LSA.shape[0]

	''' Split back the matrix to train+test and validate  '''

	# V_train_test = V_train_test_validate_LSA[0:len_train_test]
	# V_train = V_train_test_validate_LSA[0:len_train]
	# V_test = V_train_test_validate_LSA[len_train:len_train_test]
	# V_validate = V_train_test_validate_LSA[len_train_test:]

	V_train_test = V_train_test_validate[0:len_train_test]
	V_train = V_train_test_validate[0:len_train]
	V_test = V_train_test_validate[len_train:len_train_test]
	V_validate = V_train_test_validate[len_train_test:]

	V_train_test.shape[0]

	''' 1. Naive Bayes'''
	# model = MultinomialNB()#fit_prior=True)
	# # model.get_params()
	# param_dist = {"alpha":list(np.arange(0.1,1.1,0.1))}

	''' 2. Support Vector Classifier '''
	# For the fitting of the classifier, set the number of iterations to 10 and the scoring to default.
	# model = svm.SVC()
	# model.get_params()
	# param_dist = {"C": sp_randint(2*10**(-1), 2000),
	#               "gamma": sp_randint(2*10**(-1), 200)}
	# Gamma in the rbf Kernel is the curvature of the decision boundary, i.e. how strict is the decision boundary
	# C is the penalty for misclassification. C small => High bias, low variance; C large => Low bias, high variance.
	print('\n 1')

	''' 3. Ridge Regression Classifier '''
	model = RidgeClassifier(solver='sag')#, class_weight='balanced')
	# model.get_params()
	param_dist = {"alpha": list(np.arange(0.01,1.1,0.15))}

	''' 4. Random Forest Classifier '''
	# model = RandomForestClassifier(verbose = 0)
	#
	# model.get_params()
	# n_estimators = [int(x) for x in np.linspace(start = 50, stop = 170, num = 4)] # Number of trees
	# max_features = ['sqrt']#, 'auto'] # Number of features to consider at every split
	# max_depth = [int(x) for x in np.linspace(3, 36, num = 4)] 	# Maximum number of levels
	# max_depth.append(None)
	# #min_samples_split = [2, 5, 10] # Minimum number of samples required to split a node
	# #min_samples_leaf = [1, 2, 4] # Minimum number of samples required at each leaf node
	# bootstrap = [True] # Method of selecting samples for training each tree
	#
	# param_dist = {'n_estimators': n_estimators,
	# 				'max_features': max_features,
	# 				'max_depth': max_depth,
	# 				#'min_samples_split': min_samples_split,
	# 				#'min_samples_leaf': min_samples_leaf,
	# 				'bootstrap': bootstrap}

	print('\n 2')

	''' Crossvalidation '''
	stratified_folds = StratifiedKFold(n_splits=3)#, shuffle=True) #The folds are made by preserving the percentage of samples for each class.

	''' Get optimal weights based on continuous label value '''
	fs_weight =list()
	w_min = list(np.arange(0.0,1.01,0.15))
	classified_docs_temp = pd.DataFrame()

	clf = GridSearchCV(estimator=model, param_grid=param_dist, n_jobs=4, cv=stratified_folds, verbose = 0) #, scoring='f1_weighted', n_iter=10
	print('\n 3')

	for w in w_min:
		print('\n Calculating weights for:'+trait+'with weight'+str(w))
		y_weights_tune = getSampleWeights(y_train_CONT,w)
		# plt.hist(y_weights_tune, bins=30)
		# len(y_weights_tune)
		clf.fit(X=V_train, y=y_train_DCT, sample_weight=y_weights_tune)
		classified_docs_temp[trait] = clf.predict(V_test)
		fs_weight.append(f1_score(y_true=y_test_DCT, y_pred=classified_docs_temp[trait], average='binary'))

	df_weights = pd.DataFrame({'weights':w_min, 'f1_score':fs_weight})
	y_minw_list = df_weights[df_weights['f1_score']==max(df_weights['f1_score'])]['weights']
	y_minw = float(y_minw_list.iloc[0])
	y_weights_opt = getSampleWeights(y_train_test_CONT,y_minw)
	print('\n Weighting optimized at 'str(y_minw))
	minw_param_trait_optimal.append(y_minw)
	clf.fit(X=V_train_test, y=y_train_test_DCT, sample_weight=y_weights_opt)
	clf.best_params_ # Parameter results from the Randomized Search

	''' RESULTS. Use the fitted, crossvalidated and optimal-parameter model to predict the validate output '''
	print('\n Appending optimized results for'+trait)
	classified_docs[trait] = clf.predict(V_validate)
	f1_df.append(f1_score(y_true=y_validate_DCT, y_pred=classified_docs[trait], average='binary'))
	print('Confusion matrix for'+trait)
	# By definition a confusion matrix C is such that C_{i, j} is equal to the number of observations known to be in group i but predicted to be in group j.
	pd.DataFrame(confusion_matrix(y_true=y_validate_DCT, y_pred=classified_docs[trait]))

f_score_results = pd.DataFrame({'0.Trait':trait_list, '1.Results':f1_df, '2.Optimal MinWeight':minw_param_trait_optimal})
f_score_results


# scorer(y_validate_DCT,classified_docs[trait])
# print(classification_report(y_validate_DCT, classified_docs[trait]))
#
# df = pd.DataFrame({'text': X_validate, 'prediction': preds, 'dct_label': y_validate_DCT, 'cont_label': y_validate_CONT})


# errors = df[df.dct_label != df.prediction]
# false_pos, false_neg = (errors[errors.dct_label==False], errors[errors.dct_label==True])
# false_pos, false_neg = getMissclassifiedDocs(X_validate, y_validate_DCT, preds, y_validate_CONT)
# false_pos.to_csv(path_or_buf=path+'/missclassified/false_pos.csv')
# false_neg.to_csv(path_or_buf=path+'/missclassified/false_neg.csv')


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
