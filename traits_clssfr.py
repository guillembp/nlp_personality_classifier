
# coding: utf-8
import spacy
import nltk
import numpy as np
import pandas as pd
import re
import os
import pickle
import utils
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, VectorizerMixin
from sklearn.feature_extraction.text import strip_accents_unicode
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.metrics import recall_score, precision_score, f1_score, average_precision_score, roc_auc_score
from sklearn.feature_extraction.text import VectorizerMixin
from langdetect.lang_detect_exception import LangDetectException
from langdetect import detect
from sklearn import svm


folderName = os.getcwd()
folderName


def scorer(y_test, p):
	d = {'precision': precision_score(y_test, p),'recall': recall_score(y_test, p),'f1-weighted':f1_score(y_test, p, average='weighted'),'ap': average_precision_score(y_test, p),'auc':roc_auc_score(y_test, p)}
	return pd.DataFrame(d, index=['score'])

def ngrammer(tokens, ngram_range):
	mix = VectorizerMixin()
	mix.ngram_range = ngram_range
	return mix._word_ngrams(tokens)

def clean_html(s):
	""" Converts all HTML elements to Unicode and removes links"""
	try:
		s = sub(r'https?://[^\s]+', '', s)
		return BeautifulSoup(s, 'html5lib').get_text() if s else ''
	except UserWarning:
		return ''
	except Exception as e:
		print(e)
		return ''

def get_errors(X_test, y_test, preds):
	""" Creates a DataFrame with false negatives and false positives """
	df = pd.DataFrame({'text': X_test, 'prediction': preds, 'label': y_test})
	problems = df[df.label != df.prediction]
	return (problems[problems.label == False], problems[problems.label == True])


def get_top_features(v, model, accepted = True, start = 1, end = 10):
	""" Get the most probable n-grams for a naive bayes model.

	>>> V_train, V_test, vectorizer = create_vectors(X_train, X_test)
	>>> model = MultinomialNB()
	>>> model.fit(V_train, y_train)
	>>> get_top_features(vectorizer, model)
	"""
	i = 1 if accepted else 0
	probs = zip(v.get_feature_names(), model.feature_log_prob_[i])
	return sorted(probs, key = lambda x: -x[1])[start:end]

def write_submission_csv(preds, ids, file_name):
	d = {'id': ids, 'label': preds}
	pd.DataFrame(d).to_csv(file_name, index = False)


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import warnings
warnings.filterwarnings('ignore')
pd.set_option('max_colwidth', 120)


# Loading the spacy
nlp = spacy.load('en')


def stop_word_removal(li):
	return [l for l in li if l not in ENGLISH_STOP_WORDS]

def clean_twitter(s):

	# MAKE A FUNCTION THAT REMOVES NON ENGLISH TWEETS
	# MAKE A FUNCTION THAT REMOVES URLS
	s = re.sub(r"http\S+", "", s)
	s = re.sub(r"https\S+", "", s)
	s = re.sub(r"www.\S+", "", s)

	# MAKE A FUNCTION THAT REMOVES "RT"
	s = re.sub(r"RT", "", s)

	#s = sub(r'@\w+', '', s) #remove @ mentions from tweets
	return s

def preprocessor(s):
	""" For all basic string cleanup.
	Think of what you can add to this to improve things. What is
	specific to your goal, how can you transform the text. Add tokens,
	remove things, unify things.
	"""
	# s = clean_html(s)
	s = strip_accents_unicode(s.lower())
	s = clean_twitter(s)
	return s

def action_tokenizer(sent):
	""" Adds a token to the end with nsubj and root verb"""
	doc = nlp(sent)
	tokens = sorted(doc, key = lambda t: t.dep_)
	return ' '.join([t.lemma_ for t in tokens if t.dep_ in ['nsubj', 'ROOT']])

action_tokenizer('a migrant died in crossing the river')

def dep_tokenizer(sent):
	""" A simple version of tokenzing with the dependencies.

	Note: this is monolingual! Also, it still doesn't take into
	account correlations!
	"""
	doc = nlp(sent)
	tokens = [t for t in doc if not t.is_stop and t.dep_ not in ['punct', '']]
	return [':'.join([t.lemma_,t.dep_]) for t in tokens]

dep_tokenizer('a migrant died in crossing the river')

def analyzer(s, ngram_range = (1,3)):
    """ Does everything to turn raw documents into tokens.

    Note: None of the above tokenizers are implemented!
    """
    s = preprocessor(s)
    pattern = re.compile(r"(?u)\b\w\w+\b")
    unigrams = pattern.findall(s)
    unigrams = [u for u in unigrams if u not in ENGLISH_STOP_WORDS]
    tokens = ngrammer(unigrams, ngram_range)
    return tokens

def create_vectors(X_train, X_test, analyzer = analyzer):
	""" Just a small helper function that applies the SKLearn Vectorizer with our analyzer """
	idx = X_train.shape[0]
	X = pd.concat([X_train, X_test])
	vectorizer = TfidfVectorizer(analyzer=analyzer).fit(X)
	vector = vectorizer.transform(X)
	return vector[0:idx], vector[idx:], vectorizer

# ''' DATA LOADING AND RESHAPING '''

with open('/Users/guillembp/Desktop/data/jobScreening_cvpr17/train/transcription_training.pkl', 'rb') as f:
	transcription_training = pickle.load(f, encoding='latin1')
len(transcription_training)
transcription_training

with open('/Users/guillembp/Desktop/data/jobScreening_cvpr17/train/annotation_training.pkl', 'rb') as g:
	annotation_training = pickle.load(g, encoding='iso-8859-1')
annotation_training = annotation_training['interview']
len(annotation_training)
annotation_training


with open('/Users/guillembp/Desktop/data/jobScreening_cvpr17/validation/transcription_validation.pkl', 'rb') as f:
    transcription_validation = pickle.load(f, encoding='latin1')
len(transcription_validation)
transcription_validation

with open('/Users/guillembp/Desktop/data/jobScreening_cvpr17/validation/annotation_validation.pkl', 'rb') as g:
    annotation_validation = pickle.load(g, encoding='iso-8859-1')
annotation_validation = annotation_validation['interview']
len(annotation_validation)
annotation_validation


# lists to feed into the model

key_list_train = list()
for k,v in sorted(transcription_training.items()):
	key_list_train.append(k)
key_list_train
len(key_list_train)

X_train = list()
for k,v in sorted(transcription_training.items()):
	X_train.append(v)
X_train
len(X_train)

y_train = list()
for k,v in sorted(annotation_training.items()):
	y_train.append(float(v))
y_train
len(y_train)

# Create dataframe for training set
df_trainning = pd.DataFrame(
    {'key_list_train': key_list_train,
     'X_train': X_train,
     'y_train': y_train
    })
df_trainning

key_list_test = list()
for k,v in sorted(transcription_validation.items()):
	key_list_test.append(k)
key_list_test
len(key_list_test)

X_test = list()
for k,v in sorted(transcription_validation.items()):
	X_test.append(v)
X_test
len(X_test)

y_test = list()
for k,v in sorted(annotation_validation.items()):
	y_test.append(float(v))
y_test
len(y_test)

# Create dataframe for validation set
df_validating = pd.DataFrame(
    {'key_list_test': key_list_test,
     'X_test': X_test,
     'y_test': y_test
    })
df_validating

# Calculate thresholds for dichotomic label
mean = sum(y_train)/float(len(y_train))
mean

y_train_sorted = sorted(y_train)
median = (y_train_sorted[2999]+y_train_sorted[3000])/2
median

sigma = np.std(y_train)
sigma

slider = 1

# Delete rows that fall inside the threshold
df_training_filtered = df_trainning.drop(df_trainning[(df_trainning.y_train>=(median-sigma*slider)) & (df_trainning.y_train<(median+sigma*slider))].index)

len(df_training_filtered)

df_validating_filtered = df_validating.drop(df_validating[(df_validating.y_test>=(median-sigma*slider)) & (df_validating.y_test<(median+sigma*slider))].index)

len(df_validating_filtered)

def dichotomizer(x, col, middle):
	if x[col] < middle:
		return -1
	elif x[col] >= middle:
		return 1

df_training_filtered["y_train_dichotomic"] = df_training_filtered.apply(lambda x: dichotomizer(x,'y_train',median), axis=1)
df_training_filtered

df_validating_filtered["y_test_dichotomic"] = df_validating_filtered.apply(lambda x: dichotomizer(x,'y_test',median), axis=1)
df_validating_filtered

X_train = df_training_filtered['X_train']
X_test = df_validating_filtered['X_test']
y_train = df_training_filtered['y_train_dichotomic']
y_test = df_validating_filtered['y_test_dichotomic']


V_train, V_test, vectorizer = create_vectors(X_train, X_test)

V_train.shape

V_train
V_test
# # Naive Bayes
#
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import recall_score, precision_score, f1_score, average_precision_score, roc_auc_score
#
# model = MultinomialNB(class_prior=[0.5,0.5])
# model.fit(V_train, y_train)
# preds = model.predict_proba(V_test)[:,1]
# roc_auc_score(y_test, preds)
#
# # Linear Support Vector Classifier
# from sklearn.svm import LinearSVC
#
# model = LinearSVC(tol = 10e-7, max_iter = -1)
# model.fit(V_train, y_train)
# preds = model.decision_function(V_test)
# roc_auc_score(y_test, preds)

# Radial Support Vector Classifier

model = svm.SVC(kernel='rbf', tol = 10e-7, max_iter = -1)
model.fit(V_train, y_train)
preds = model.decision_function(V_test)
preds
y_test
roc_auc_score(y_test, preds)

# Look at your false predictions!
false_pos, false_neg = get_errors(X_test, y_test, preds)



# Pipeline for range plotting



# Results
#
# test_df = pd.read_csv('test.csv')
# X_sub, id_sub = test_df.tweet, test_df.id
# V_train, V_test, _ = create_vectors(X, X_sub)
# model.fit(V_train, y)
# preds = model.decision_function(V_test)
# write_submission_csv(preds, id_sub, 'submission.csv')
