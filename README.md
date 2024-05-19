# Improving the performance of a personality trait classifier trained on ambiguous labels
## Abstract
In this project I use Natural Language Processing techniques and several machine learning models, to compare their performance in classifing interviewees from the EmotiW 2017 dataset into 6 personality types from interview transcript data. In the ground truth (i.e. labels), most subjects have ambiguous personality type labels because of labeler disagreement, as measured in Fleiss’s kappa coefficient. In order to increase the contrast among personality traits, several solutions are compared that *dichotmize* the labels to maximizes the posterior classification performance, first by finding ambiguity thresholds and truncating the data and later by comparing two different weighting functions. Finally, we explore NLP choices, dimensionality redction with PCA, Random Forest and Multinomial Naive Bayes machine learning models and model tuning to continue improving the classifier's performance.