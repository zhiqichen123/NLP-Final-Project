#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 13:43:45 2019

@author: xupech
"""

import sys
import argparse
import re
import os
import math
import operator
import random
import json
import nltk
import pickle
import timeit
from pickle import load
from pickle import dump
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction import DictVectorizer
from nltk.corpus import movie_reviews
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from nltk.corpus import sentiwordnet as swn
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
import pandas as pd
import numpy as np

STOPLIST = set(nltk.corpus.stopwords.words('english'))

df = pd.read_json('News_Category_Dataset_v2.json', lines = True)
# df.to_csv('News_category.csv')

"""
Read in the dataset.
"""


df.category = df.category.map(lambda x: "WORLDPOST" if x == "THE WORLDPOST" else x)
df.category = df.category.map(lambda x: "STYLE" if x == "STYLE & BEAUTY" else x)
df.category = df.category.map(lambda x: "ARTS" if x == "ARTS & CULTURE" or x == "CULTURE & ARTS" else x)
df.category = df.category.map(lambda x: "PARENTS" if x == "PARENTING" else x)
df.category = df.category.map(lambda x: "ENVIRONMENT" if x == "GREEN" else x)
df.category = df.category.map(lambda x: "EDUCATION" if x == "COLLEGE" else x)
df.category = df.category.map(lambda x: "BUSINESS" if x == "MONEY" else x)
df.category = df.category.map(lambda x: "TECH & SCIENCE" if x == "TECH" or x == 'SCIENCE' else x)


category = df.groupby('category')

def PRINT_CATEGORIES():
    print("total categories:", category.ngroups)
    print(category.size())

"""
Tweak the categories. Merge mistyped categories into one category. Result in a total number of 32 categories in the end.
"""

df['text'] = df.headline + " " + df.short_description
df['words'] = df.text.apply(lambda i: nltk.word_tokenize(str(i)))

"""
Combine headline and short description together into another dimension called text. Use text to categorize news for better 
results instead of headline or short description alone.
"""

text = np.array(df['text'])
category = np.array(df['category'])
text_train, text_test, category_train, category_test = train_test_split(text, category, test_size=0.2, random_state=32)
"""
Split the data into two parts: 80% for training, 20% for test. Random state = 32 to match number of categories.
"""

uni_cv = CountVectorizer(stop_words='english',max_features=10000)
uni_cv_train_features = uni_cv.fit_transform(text_train)
uni_cv_test_features = uni_cv.transform(text_test)
"""
Use countvectorizer to encode the text in unigrams. Drop stop_words. To avoid crazy computation later, set max features = 10000.
"""

bi_cv = CountVectorizer(stop_words='english',max_features=10000, ngram_range=(2, 2))
bi_cv_train_features = bi_cv.fit_transform(text_train)
bi_cv_test_features = bi_cv.transform(text_test)

"""
Use countvectorizer to encode the text in bigrams.
"""


uni_tv = TfidfVectorizer(stop_words='english',max_features=10000,
                     sublinear_tf=True)
uni_tv_train_features = uni_tv.fit_transform(text_train)
uni_tv_test_features = uni_tv.transform(text_test)

"""
Use tfidfvectorizer to encode the text in unigrams.
"""

bi_tv = TfidfVectorizer(stop_words='english',max_features=10000,
                     sublinear_tf=True, ngram_range=(2, 2))
bi_tv_train_features = bi_tv.fit_transform(text_train)
bi_tv_test_features = bi_tv.transform(text_test)

"""
Use tfidfvectorizer to encode the text in bigrams.
"""

def TRAIN():
    """
    Unigram countvectorizer multinomial naive bayes model.
    """ 
    start = timeit.default_timer()
    
    uni_bayes_all_wordsc = MultinomialNB()
    uni_bayes_all_wordsc.fit(uni_cv_train_features, category_train)    
    print('Creating Bayes classifier in classifiers/uni_bayes_bow.pkl')
    print('Accuracy: {:.2f}'.format(uni_bayes_all_wordsc.score(uni_cv_test_features, category_test)))
    with open('./classifiers/uni_bayes_all_wordsc.pkl', 'wb') as output:
        pickle.dump(uni_bayes_all_wordsc, output, -1)
    output.close()

    stop = timeit.default_timer()
    print('Time: ', round((stop - start),2))

    
    """
    Unigram tf-idf multinomial naive bayes model.
    """
    start = timeit.default_timer()
    
    uni_bayes_all_wordst = MultinomialNB()
    uni_bayes_all_wordst.fit(uni_tv_train_features, category_train)
    print('Creating Bayes classifier in classifiers/uni_bayes_tfidf.pkl')
    print('Accuracy: {:.2f}'.format(uni_bayes_all_wordst.score(uni_tv_test_features, category_test)))
    with open('./classifiers/uni_bayes_all_wordst.pkl', 'wb') as output:
        pickle.dump(uni_bayes_all_wordst, output, -1)
    output.close()

    stop = timeit.default_timer()
    print('Time: ', round((stop - start),2))
    
    
    """
    Bigram countvectorizer multinomial naive bayes model.
    """
    start = timeit.default_timer()
    
    bi_bayes_all_wordsc = MultinomialNB()
    bi_bayes_all_wordsc.fit(bi_cv_train_features, category_train)
    print('Creating Bayes classifier in classifiers/bi_bayes_bow.pkl')
    print('Accuracy: {:.2f}'.format(bi_bayes_all_wordsc.score(bi_cv_test_features, category_test)))
    with open('./classifiers/bi_bayes_all_wordsc.pkl', 'wb') as output:
        pickle.dump(bi_bayes_all_wordsc, output, -1)
    output.close()

    stop = timeit.default_timer()
    print('Time: ', round((stop - start),2))    
    
    
    """
    Bigram tf-idf multinomial naive bayes model.
    """
    start = timeit.default_timer()
    
    bi_bayes_all_wordst = MultinomialNB()
    bi_bayes_all_wordst.fit(bi_tv_train_features, category_train)
    print('Creating Bayes classifier in classifiers/bi_bayes_tfidf.pkl')
    print('Accuracy: {:.2f}'.format(bi_bayes_all_wordst.score(bi_tv_test_features, category_test)))
    with open('./classifiers/bi_bayes_all_wordst.pkl', 'wb') as output:
        pickle.dump(bi_bayes_all_wordst, output, -1)
    output.close()

    stop = timeit.default_timer()
    print('Time: ', round((stop - start),2))      
    
       
    # =============================================================================
    # tree_all_words = DecisionTreeClassifier(criterion='entropy', random_state=32)
    # tree_all_words.fit(uni_cv_train_features, category_train)
    # tree_all_words.score(uni_cv_test_features, category_test)
    # 
    # 
    # tree_all_words = DecisionTreeClassifier(criterion='entropy', random_state=32)
    # tree_all_words.fit(uni_tv_train_features, category_train)
    # tree_all_words.score(uni_tv_test_features, category_test)
    # =============================================================================
    """
    tree takes too long time.
    """
    
    
    """
    Unigram countvectorizer logistic regression model. Since bigrams give worse results than unigrams, we skip
    bigram features in later modeling.
    """
    start = timeit.default_timer()
    
    uni_lrc = LogisticRegression(random_state=32)
    uni_lrc.fit(uni_cv_train_features, category_train)    
    print('Creating logistic regression in classifiers/uni_logistic_bow.pkl')
    print('Accuracy: {:.2f}'.format(uni_lrc.score(uni_cv_test_features, category_test)))
    with open('./classifiers/uni_lrc.pkl', 'wb') as output:
        pickle.dump(uni_lrc, output, -1)
    output.close()

    stop = timeit.default_timer()
    print('Time: ', round((stop - start),2))
    
    
    """
    Unigram tfidf logistic regression model.
    """
    start = timeit.default_timer()
    
    uni_lrt = LogisticRegression(random_state=32)
    uni_lrt.fit(uni_tv_train_features, category_train)
    print('Creating logistic regression in classifiers/uni_logistic_tfidf.pkl')
    print('Accuracy: {:.2f}'.format(uni_lrt.score(uni_tv_test_features, category_test)))
    with open('./classifiers/uni_lrt.pkl', 'wb') as output:
        pickle.dump(uni_lrt, output, -1)
    output.close()

    stop = timeit.default_timer()
    print('Time: ', round((stop - start),2))
    

    """
    Unigram countvectorizer sgd classifier model.
    """
    start = timeit.default_timer()
    
    uni_sgdc = SGDClassifier(random_state=32)
    uni_sgdc.fit(uni_cv_train_features, category_train)
    print('Creating sdg classifier in classifiers/uni_sgd_bow.pkl')
    print('Accuracy: {:.2f}'.format(uni_sgdc.score(uni_cv_test_features, category_test)))
    with open('./classifiers/uni_sgdc.pkl', 'wb') as output:
        pickle.dump(uni_sgdc, output, -1)
    output.close()

    stop = timeit.default_timer()
    print('Time: ', round((stop - start),2))
    
    
    """
    Unigram tf-idf sgd classifier model.
    """
    start = timeit.default_timer()
    
    uni_sgdt = SGDClassifier(random_state=32)
    uni_sgdt.fit(uni_tv_train_features, category_train)
    print('Creating sdg classifier in classifiers/uni_sgd_tfidf.pkl')
    print('Accuracy: {:.2f}'.format(uni_sgdt.score(uni_tv_test_features, category_test)))
    with open('./classifiers/uni_sgdt.pkl', 'wb') as output:
        pickle.dump(uni_sgdt, output, -1)
    output.close()

    stop = timeit.default_timer()
    print('Time: ', round((stop - start),2))
    
 
"""
Prediction function starts here.
"""    
    
def uni_bayes_bow(text_path):
    with open(text_path, 'r') as f:
        text = f.read()
    word = [text]

    feature_bow = uni_cv.transform(word)
    input = open('./classifiers/uni_bayes_all_wordsc.pkl', 'rb')
    model = load(input)
    input.close()
    print(model.predict(feature_bow))     


def uni_bayes_tfidf(text_path):
    with open(text_path, 'r') as f:
        text = f.read()
    word = [text]

    feature_tfidf = uni_tv.transform(word)
    input = open('./classifiers/uni_bayes_all_wordst.pkl', 'rb')
    model = load(input)
    input.close()
    print(model.predict(feature_tfidf))  
    

def uni_logistic_bow(text_path):
    with open(text_path, 'r') as f:
        text = f.read()
    word = [text]
    
    feature_bow = uni_cv.transform(word)
    input = open('./classifiers/uni_lrc.pkl', 'rb')
    model = load(input)
    input.close()
    print(model.predict(feature_bow))


def uni_logistic_tfidf(text_path):
    with open(text_path, 'r') as f:
        text = f.read()
    word = [text]
    
    feature_tfidf = uni_tv.transform(word)
    input = open('./classifiers/uni_lrt.pkl', 'rb')
    model = load(input)
    input.close()
    print(model.predict(feature_tfidf))     

    
def uni_sgd_bow(text_path):
    with open(text_path, 'r') as f:
        text = f.read()
    word = [text]
    
    feature_bow = uni_cv.transform(word)
    input = open('./classifiers/uni_sgdc.pkl', 'rb')
    model = load(input)
    input.close()
    print(model.predict(feature_bow))


def uni_sgd_tfidf(text_path):
    with open(text_path, 'r') as f:
        text = f.read()
    word = [text]
    
    feature_tfidf = uni_tv.transform(word)
    input = open('./classifiers/uni_sgdt.pkl', 'rb')
    model = load(input)
    input.close()
    print(model.predict(feature_tfidf))

"""
Here begins recommending articles based on one article.
"""
 
from sklearn.metrics import pairwise_distances

reco_cv = CountVectorizer(stop_words='english',max_features=10000)
reco_cv_features = uni_cv.fit_transform(text)

"""
Recommending articles applying bag of words method to text variable of each article.
"""
def bow_reco(news_index, num_similar):
    all_distance = pairwise_distances(reco_cv_features,reco_cv_features[news_index])
    """ all_distance is 2-d array, convert it to 1-d array. """
    new_all = all_distance.ravel()
    news_num = new_all.argsort()[0:num_similar+1]
    target = news_num[0]
    
    print('Recommendation using word counts method.')
    print('\n', 'News Just Read:')
    print(df.authors[target], '|', df.date[target], '|', df.category[target], '|', df.text[target])
    
    print('\n', 'Recommended News For You:')
    for reco in news_num[1:num_similar+1]:
        print(df.authors[reco], '|', df.date[reco], '|', df.category[reco], '|', df.text[reco])
        print('\n')

        
reco_tv = TfidfVectorizer(stop_words='english',max_features=10000,
                     sublinear_tf=True)
reco_tv_features = uni_tv.fit_transform(text)        
    
"""
Recommending articles applyinng tf-idf method to text variable of earch article.
"""
def tfidf_reco(news_index, num_similar):
    all_distance = pairwise_distances(reco_tv_features,reco_tv_features[news_index])
    """ all_distance is 2-d array, convert it to 1-d array. """
    new_all = all_distance.ravel()
    news_num = new_all.argsort()[0:num_similar+1]
    target = news_num[0]
    
    print('Recommendation using TF-IDF method.')    
    print('\n', 'News Just Read:')
    print(df.authors[target], '|', df.date[target], '|', df.category[target], '|', df.text[target])
    
    print('\n', 'Recommended News For You:')
    for reco in news_num[1:num_similar+1]:
        print(df.authors[reco], '|', df.date[reco], '|', df.category[reco], '|', df.text[reco])
        print('\n')
        
"""
Recommending articles applying bag of words method to text variable of each article, plus giving weight to
category and author.
"""
cat_ohe = OneHotEncoder()
cat_features = cat_ohe.fit_transform(category.reshape(-1, 1))

aut_ohe = OneHotEncoder()
authors = np.array(df['authors'])
aut_features = aut_ohe.fit_transform(authors.reshape(-1, 1))

def bow_reco_weight(news_index, num_similar):
    text_distance = pairwise_distances(reco_cv_features,reco_cv_features[news_index])
    text_mean = text_distance.mean()
    text_std = text_distance.std()
    new_text = (text_distance-text_mean)/text_std
    """
    Normalize distance of text so that it can be used later to calculate weight.
    """
    
    cat_distance = pairwise_distances(cat_features,cat_features[news_index])
    cat_mean = cat_distance.mean()
    cat_std = cat_distance.std()
    new_cat = (cat_distance-cat_mean)/cat_std
    
    aut_distance = pairwise_distances(aut_features,aut_features[news_index])
    aut_mean = aut_distance.mean()
    aut_std = aut_distance.std()
    new_aut = (aut_distance-aut_mean)/aut_std    
    
        
    """
    Assign weight to the normalized distance, text has weight 40%, category has weight 30% and author has 30%.
    """
    all_distance = new_text*0.4+new_cat*0.3+new_aut*0.3
    new_all = all_distance.ravel()
        
    news_num = new_all.argsort()[0:num_similar+1]
    target = news_num[0]
    
    print('Recommendation using word counts method plus giving weight to author and category.')
    print('\n', 'News Just Read:')
    print(df.authors[target], '|', df.date[target], '|', df.category[target], '|', df.text[target])
    
    print('\n', 'Recommended News For You:')
    for reco in news_num[1:num_similar+1]:
        print(df.authors[reco], '|', df.date[reco], '|', df.category[reco], '|', df.text[reco])
        print('\n')

"""
Recommending articles applying tf-idf method to text variable of each article, plus giving weight to
category and author.
"""
def tfidf_reco_weight(news_index, num_similar):
    text_distance = pairwise_distances(reco_tv_features,reco_tv_features[news_index])
    text_mean = text_distance.mean()
    text_std = text_distance.std()
    new_text = (text_distance-text_mean)/text_std
    """
    Normalize distance of text so that it can be used later to calculate weight.
    """
    
    cat_distance = pairwise_distances(cat_features,cat_features[news_index])
    cat_mean = cat_distance.mean()
    cat_std = cat_distance.std()
    new_cat = (cat_distance-cat_mean)/cat_std
    
    aut_distance = pairwise_distances(aut_features,aut_features[news_index])
    aut_mean = aut_distance.mean()
    aut_std = aut_distance.std()
    new_aut = (aut_distance-aut_mean)/aut_std    
    
        
    """
    Assign weight to the normalized distance, text has weight 40%, category has weight 30% and author has 30%.
    """
    all_distance = new_text*0.4+new_cat*0.3+new_aut*0.3
    new_all = all_distance.ravel()
        
    news_num = new_all.argsort()[0:num_similar+1]
    target = news_num[0]
    
    print('Recommendation using tfidf method plus giving weight to author and category.')
    print('\n', 'News Just Read:')
    print(df.authors[target], '|', df.date[target], '|', df.category[target], '|', df.text[target])
    
    print('\n', 'Recommended News For You:')
    for reco in news_num[1:num_similar+1]:
        print(df.authors[reco], '|', df.date[reco], '|', df.category[reco], '|', df.text[reco])
        print('\n')


def Return_Result(feature, index, text_path):

    if feature == 'bow' and index == '1':
       uni_bayes_bow(text_path)
    if feature == 'tfidf' and index == '1':
        uni_bayes_tfidf(text_path)
    if feature == 'bow' and index == '2':
        uni_logistic_bow(text_path)
    if feature == 'tfidf' and index == '2':
        uni_logistic_tfidf(text_path)
    if feature == 'bow' and index == '3':
        uni_sgd_bow(text_path)
    if feature =='tfidf' and index == '3':
        uni_sgd_tfidf(text_path)

def Recommendation_(model_indices, indices, num_recommendation):
    if model_indices == '1':
        tfidf_reco(indices, num_recommendation)
    if model_indices == '2':
        bow_reco(indices, num_recommendation)
    if model_indices == '3':
        tfidf_reco_weight(indices, num_recommendation)
    if model_indices == '4':
        bow_reco_weight(indices, num_recommendation)

parser = argparse.ArgumentParser(prog = 'News Categorizer')
parser.add_argument('--train', dest = 'train', action = 'store_true')
parser.add_argument('--run', dest = 'run', type = str, nargs = '+')
parser.add_argument('--recommend', dest='recommend', type = str, nargs = '+')

args = parser.parse_args()

if args.train:
    TRAIN()

if args.run:
    path = './' + args.run[1]
    if len(args.run) == 2:
        print("""Choose a model:
1 - unigram_bayes
2 - unigram_logistic_regression
3 - sgd classifier""")
        model_index = input('Type a number:\n')         
        Return_Result(args.run[0], model_index, path)

if args.recommend:
    index = args.recommend[0]
    num_ = args.recommend[1]
    if len(args.recommend) == 2:
        print("""Choose how you want to recommend the news, dear reader:
1 - text based only in tfidf
2 - text based only using raw count
3 - text based in tfidf plus giving weight to author and category
4 - text based using raw count plus giving weight to author and category""")
        model_index = input('Type a number:\n')
        Recommendation_(model_index, int(index), int(num_))
            


