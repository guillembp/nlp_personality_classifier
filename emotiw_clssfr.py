
# coding: utf-8
import spacy
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, VectorizerMixin
from utils import *
import re
import os

get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import warnings
warnings.filterwarnings('ignore')
pd.set_option('max_colwidth', 120)


# Loading the spacy
nlp = spacy.load('en')

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
def stop_word_removal(li):
    return [l for l in li if l not in ENGLISH_STOP_WORDS]

import utils
from sklearn.feature_extraction.text import strip_accents_unicode


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
    s = clean_html(s)
    s = strip_accents_unicode(s.lower())
    s = clean_twitter(s)
    return s

import spacy

def action_tokenizer(sent):
    """ Adds a token to the end with nsubj and root verb"""
    doc = nlp(sent)
    tokens = sorted(doc, key = lambda t: t.dep_)
    return ' '.join([t.lemma_ for t in tokens if t.dep_ in ['nsubj', 'ROOT']])

action_tokenizer('a migrant died in crossing the river')


from langdetect.lang_detect_exception import LangDetectException
from langdetect import detect

def dep_tokenizer(sent):
    """ A simple version of tokenzing with the dependencies.

    Note: this is monolingual! Also, it still doesn't take into
    account correlations!
    """
    doc = nlp(sent)
    tokens = [t for t in doc if not t.is_stop and t.dep_ not in ['punct', '']]
    return [':'.join([t.lemma_,t.dep_]) for t in tokens]

dep_tokenizer('a migrant died in crossing the river')

import re
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

def analyzer(s, ngram_range = (2,4)):
    """ Does everything to turn raw documents into tokens.

    Note: None of the above tokenizers are implemented!
    """
    s = preprocessor(s)
    pattern = re.compile(r"(?u)\b\w\w+\b")
    unigrams = pattern.findall(s)
    unigrams = [u for u in unigrams if u not in ENGLISH_STOP_WORDS]
    tokens = ngrammer(unigrams, ngram_range)
    return tokens


''' DATA READING AND PROCESSING '''

'''
1. Open the labels file
2. Make a list of the txt file names
3. Make a loop that runs through all files in list
4. Opens every file
5. Copies content next to dictionary
'''

import numpy as np

folderName = os.getcwd()
folderName
y = pd.read_table('/Users/guillembp/Dropbox/Thesis/NLP/text-mining-master/data/lists/EmotiW_Train_Val_labels.lst', header=None, sep=' ', dtype=np.string_)
y

#
# emotion_list ={'Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'}
# avi_path = '/Users/guillembp/Desktop/data/2017_EmotiW'
# for line in emotion_list:


X = pd.read_table('/Users/guillembp/Desktop/data/jobScreening_cvpr17/validation/transcription_validation.pkl', header='None')


import pickle

with open('/Users/guillembp/Desktop/data/jobScreening_cvpr17/train/transcription_training.pkl', 'rb') as f:
    labels = pickle.load(f, encoding='latin1')

labels


with open('/Users/guillembp/Desktop/data/jobScreening_cvpr17/train/annotation_training.pkl', 'rb') as g:
    annotation_train = pickle.load(g, encoding='iso-8859-1')
annotation_train


# with open('/Users/guillembp/Desktop/data/jobScreening_cvpr17/test/annotation_test.pkl', 'rb') as g:
#     annotation_test = pickle.load(g, encoding='iso-8859-1')
# annotation_test
#
# with open('/Users/guillembp/Desktop/data/jobScreening_cvpr17/validation/annotation_validation.pkl', 'rb') as g:
#     annotation_validation = pickle.load(g, encoding='iso-8859-1')
# annotation_validation

with open('/Users/guillembp/Desktop/data/jobScreening_cvpr17/train/transcription_training.pkl', 'rb') as g:
    transcription_train = pickle.load(g, encoding='iso-8859-1')
transcription_train


cutoff = 2050
X_train, X_test, y_train, y_test = X[0:cutoff], X[cutoff:], y[0:cutoff], y[cutoff:]
X.shape

X_test.shape, y_test.shape

def create_vectors(X_train, X_test, analyzer = analyzer):
    """ Just a small helper function that applies the SKLearn Vectorizer with our analyzer """
    idx = X_train.shape[0]
    X = pd.concat([X_train, X_test])
    vectorizer = TfidfVectorizer(analyzer=analyzer).fit(X)
    vector = vectorizer.transform(X)
    return vector[0:idx], vector[idx:], vectorizer

V_train, V_test, vectorizer = create_vectors(X_train, X_test)

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
from sklearn import svm

model = svm.SVC(kernel='rbf', tol = 10e-7, max_iter = -1)
model.fit(V_train, y_train)
preds = model.decision_function(V_test)
roc_auc_score(y_test, preds)

# PCA?

# GLM with regularization?

# Look at your false predictions!
false_pos, false_neg = get_errors(X_test, y_test, preds)

# Results

test_df = pd.read_csv('kaggle/test.csv')
X_sub, id_sub = test_df.tweet, test_df.id
V_train, V_test, _ = create_vectors(X, X_sub)
model.fit(V_train, y)
preds = model.decision_function(V_test)
write_submission_csv(preds, id_sub, 'kaggle/submission.csv')
