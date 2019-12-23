#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 16:09:56 2019

@author: danielz
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
import timeit
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

df = pd.read_json('News_Category_Dataset_v2.json', lines = True)
print('completed')
df.category = df.category.map(lambda x: "WORLDPOST" if x == "THE WORLDPOST" else x)
df.category = df.category.map(lambda x: "STYLE" if x == "STYLE & BEAUTY" else x)
df.category = df.category.map(lambda x: "ARTS" if x == "ARTS & CULTURE" or x == "CULTURE & ARTS" else x)
df.category = df.category.map(lambda x: "PARENTS" if x == "PARENTING" else x)
df.category = df.category.map(lambda x: "ENVIRONMENT" if x == "GREEN" else x)
df.category = df.category.map(lambda x: "EDUCATION" if x == "COLLEGE" else x)
df.category = df.category.map(lambda x: "BUSINESS" if x == "MONEY" else x)
df.category = df.category.map(lambda x: "TECH & SCIENCE" if x == "TECH" or x == 'SCIENCE' else x)

author = df.groupby('authors', as_index = False).agg({'headline' : 'count'})
author.sort_values(by = 'headline', ascending = False, inplace = True)
authors = list(author[1:10]['authors'])
df.set_index('authors', inplace = True)
df_authors = df.loc[authors].reset_index().drop(columns = ['link', 'date'])
df_authors['text'] = df_authors.headline + ' ' + df_authors.short_description + ' ' + df_authors.category
text = list(df_authors['text'])
label = list(df_authors['authors'])


def create_feature_train_test(vectorizer):  
    features = vectorizer.fit_transform(text).toarray()
    return train_test_split(features, label, test_size=0.2, random_state=32)

def TRAIN(clf, trainfeatureset, trainlabelset, testfeatureset, testlabelset):
    if clf == 'MNB':
        clf = MultinomialNB()
    elif clf == 'LR':
        clf = LogisticRegression(C = 3, random_state=32, multi_class = 'auto')
    elif clf == 'DT':
        clf = DecisionTreeClassifier(random_state=32)
    elif clf == 'BNB':
        clf = BernoulliNB()
    start = timeit.default_timer()
    clf.fit(trainfeatureset, trainlabelset)    

    accuracy = round(clf.score(testfeatureset, testlabelset), 2)
    print(str(clf) + ' \nAccuracy: {:.2f}'.format(accuracy))

    stop = timeit.default_timer()
    elaptime = round(stop - start, 2)
    print('Time: {:.2f}s'.format(elaptime))
    return accuracy, elaptime

unicv_train, unicv_test, label_train, label_test = create_feature_train_test(CountVectorizer(stop_words = 'english'))
unicvbin_train, unicvbin_test, label_train, label_test = create_feature_train_test(CountVectorizer(stop_words = 'english', binary = True))
unitv_train, unitv_test, label_train, label_test = create_feature_train_test(TfidfVectorizer(stop_words= 'english', sublinear_tf=True))
unitvbin_train, unitvbin_test, label_train, label_test = create_feature_train_test(TfidfVectorizer(stop_words= 'english', sublinear_tf=True, binary = True))


accuracy_elaptime = pd.DataFrame({'accuracy':[0] * 12, 'elapsedtime':[0] * 12})

accuracies = []
elaptimes = []
for clf in ['LR', 'DT']:
    print('\nraw_count, CountVectorizer:')
    accuracy, elaptime = TRAIN(clf, unicv_train, label_train, unicv_test, label_test)
    accuracies.append(accuracy)
    elaptimes.append(elaptime)
    print('\nraw_count, TfidfVectorizer:')
    accuracy, elaptime = TRAIN(clf, unitv_train, label_train, unitv_test, label_test)
    accuracies.append(accuracy)
    elaptimes.append(elaptime)
    print('\nbinary, CountVectorizer:')
    accuracy, elaptime = TRAIN(clf, unicvbin_train, label_train, unicvbin_test, label_test)
    accuracies.append(accuracy)
    elaptimes.append(elaptime)
    print('\nbinary, TfidfVectorizer:')
    accuracy, elaptime = TRAIN(clf, unitvbin_train, label_train, unitvbin_test, label_test)
    accuracies.append(accuracy)
    elaptimes.append(elaptime)

print('\nraw_count, CountVectorizer:')
accuracy, elaptime = TRAIN('MNB', unicv_train, label_train, unicv_test, label_test)
accuracies.append(accuracy)
elaptimes.append(elaptime)
print('\nraw_count, TfidfVectorizer:')
accuracy, elaptime = TRAIN('MNB', unitv_train, label_train, unitv_test, label_test)
accuracies.append(accuracy)
elaptimes.append(elaptime)

print('\nbinary, CountVectorizer:')
accuracy, elaptime = TRAIN('BNB', unicvbin_train, label_train, unicvbin_test, label_test) 
accuracies.append(accuracy)
elaptimes.append(elaptime)
print('\nbianry, TfidfVectorizer:')
accuracy, elaptime = TRAIN('BNB', unitvbin_train, label_train, unitvbin_test, label_test)
accuracies.append(accuracy)
elaptimes.append(elaptime)

accuracy_elaptime['raw_count_binary'] = (['raw_count'] * 2 + ['binary'] * 2) * 3
accuracy_elaptime['vectorizer'] = ['Count', 'Tfidf'] * 6 
accuracy_elaptime['classifier'] = ['LR'] * 4 + ['DT'] * 4 + ['MNB'] * 2 + ['BNB'] * 2
accuracy_elaptime['accuracy'] = accuracies
accuracy_elaptime['elapsed_time'] = elaptimes
accuracy_elaptime = accuracy_elaptime.loc[:, ['raw_count_binary', 'vectorizer', 'classifier', 'accuracy', 'elapsed_time']].sort_values(by = 'accuracy', ascending = False)
print(accuracy_elaptime)