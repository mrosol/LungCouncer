# -*- coding: utf-8 -*-
"""
Created on Mon May 11 23:32:11 2020

@author: Maciej Rosol
"""

#%% importing libraries
import os
os.chdir(os.path.dirname(__file__))
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import KFold, cross_val_predict
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import random
import numpy as np
import seaborn as sns

#%% loadind data
data = pd.read_csv('LungCancer.txt', delimiter ='\t') 
data = data.T

data.reset_index(drop = True, inplace = True)

#%% feature selection
X = data.iloc[1:,1:] # choosing only numeric features without names of gens
y = data.iloc[1:,0] # choosing labels

best_features = SelectKBest(score_func = f_classif, k = 500)
fit_best = best_features.fit(X,y)
df_scores = pd.DataFrame(fit_best.scores_)  # scores of each features
df_features = pd.DataFrame(X.columns)   # numbers of columns
df_features_scores = pd.concat((df_scores,df_features),axis = 1)    # concatenating scores and numbers of columns
df_features_scores.columns = ['Scores','Features']
selected = df_features_scores.nlargest(500,'Scores')    # selecting 100 numbers of columns with bigest scores
X_selected = X.iloc[:,selected['Features']]     # selecting 100 best features for classification

#%% SVM classification
random.seed(1)
kf = KFold(10, shuffle = True, random_state = 1) # 10-fold cross validation

cm_SVM_test_all_features = np.zeros([5,5])
cm_SVM_train_all_features = np.zeros([5,5])
for train, test in kf.split(X,y):   # for each fold for all features
    X_train = X.iloc[train,:]
    y_train = pd.to_numeric(y.iloc[train]).round(0).astype(int)
    X_test = X.iloc[test,:]
    y_test = pd.to_numeric(y.iloc[test]).round(0).astype(int)
    mdlSVM = svm.SVC(kernel = 'poly', degree = 7, coef0 = 1, C = 1.5)
    mdlSVM.fit(X_train,y_train)
    y_hat_test = mdlSVM.predict(X_test)
    y_hat_train = mdlSVM.predict(X_train)
    cm1 = confusion_matrix(y_test, y_hat_test, labels = [1,2,3,4,5])
    cm_SVM_test_all_features += cm1
    cm1 = confusion_matrix(y_train, y_hat_train, labels = [1,2,3,4,5])
    cm_SVM_train_all_features += cm1
plt.figure()
sns.heatmap(cm_SVM_test_all_features.astype(int), cmap = plt.cm.Blues, annot=True, fmt="d")
acc_SVM_test = np.sum(np.diag(cm_SVM_test_all_features))/np.sum(cm_SVM_test_all_features)
acc_SVM_train = np.sum(np.diag(cm_SVM_train_all_features))/np.sum(cm_SVM_train_all_features)
print('SVM train accuracy all features: %0.4f' %acc_SVM_train)
print('SVM test accuracy all features: %0.4f' %acc_SVM_test)

cm_SVM_test = np.zeros([5,5])
cm_SVM_train = np.zeros([5,5])
for train, test in kf.split(X_selected,y):    # for each fold for selected features
    X_train = X.iloc[train,:]
    y_train = pd.to_numeric(y.iloc[train]).round(0).astype(int)
    X_test = X.iloc[test,:]
    y_test = pd.to_numeric(y.iloc[test]).round(0).astype(int)
    mdlSVM = svm.SVC(kernel = 'poly', degree = 7, coef0 = 1, C = 1.5)
    mdlSVM.fit(X_train,y_train)
    y_hat_test = mdlSVM.predict(X_test)
    y_hat_train = mdlSVM.predict(X_train)
    cm1 = confusion_matrix(y_test, y_hat_test, labels = [1,2,3,4,5])
    cm_SVM_test += cm1
    cm1 = confusion_matrix(y_train, y_hat_train, labels = [1,2,3,4,5])
    cm_SVM_train += cm1
plt.figure()
sns.heatmap(cm_SVM_test.astype(int), cmap = plt.cm.Blues, annot=True, fmt="d")
acc_SVM_test = np.sum(np.diag(cm_SVM_test))/np.sum(cm_SVM_test)
acc_SVM_train = np.sum(np.diag(cm_SVM_train))/np.sum(cm_SVM_train)
print('SVM train accuracy: %0.4f' %acc_SVM_train)
print('SVM test accuracy: %0.4f' %acc_SVM_test)
#%% k-NN classification
cm_KNN_test_all_features = np.zeros([5,5])
cm_KNN_train_all_features = np.zeros([5,5])
for train, test in kf.split(X,y):
    X_train = X.iloc[train,:]
    y_train = pd.to_numeric(y.iloc[train]).round(0).astype(int)
    X_test = X.iloc[test,:]
    y_test = pd.to_numeric(y.iloc[test]).round(0).astype(int)
    mdlKNN = KNeighborsClassifier(3, metric = 'minkowski', p = 2)
    mdlKNN.fit(X_train,y_train)
    y_hat_test = mdlKNN.predict(X_test)
    y_hat_train = mdlKNN.predict(X_train)
    cm1 = confusion_matrix(y_test, y_hat_test, labels = [1,2,3,4,5])
    cm_KNN_test_all_features += cm1
    cm1 = confusion_matrix(y_train, y_hat_train, labels = [1,2,3,4,5])
    cm_KNN_train_all_features += cm1
plt.figure()
sns.heatmap(cm_KNN_test_all_features.astype(int), cmap = plt.cm.Blues, annot=True, fmt="d")
acc_KNN_test_all_features = np.sum(np.diag(cm_KNN_test_all_features))/np.sum(cm_KNN_test_all_features)
acc_KNN_train_all_features = np.sum(np.diag(cm_KNN_train_all_features))/np.sum(cm_KNN_train_all_features)
print('k-NN train accuracy all features: %0.4f' %acc_KNN_train_all_features)
print('k-NN test accuracy all features: %0.4f' %acc_KNN_test_all_features)


cm_KNN_test = np.zeros([5,5])
cm_KNN_train = np.zeros([5,5])
for train, test in kf.split(X_selected,y):
    X_train = X.iloc[train,:]
    y_train = pd.to_numeric(y.iloc[train]).round(0).astype(int)
    X_test = X.iloc[test,:]
    y_test = pd.to_numeric(y.iloc[test]).round(0).astype(int)
    mdlKNN = KNeighborsClassifier(3, metric = 'minkowski', p = 2)
    mdlKNN.fit(X_train,y_train)
    y_hat_test = mdlKNN.predict(X_test)
    y_hat_train = mdlKNN.predict(X_train)
    cm1 = confusion_matrix(y_test, y_hat_test, labels = [1,2,3,4,5])
    cm_KNN_test += cm1
    cm1 = confusion_matrix(y_train, y_hat_train, labels = [1,2,3,4,5])
    cm_KNN_train += cm1
plt.figure()
sns.heatmap(cm_KNN_test.astype(int), cmap = plt.cm.Blues, annot=True, fmt="d")
acc_KNN_test = np.sum(np.diag(cm_KNN_test))/np.sum(cm_KNN_test)
acc_KNN_train = np.sum(np.diag(cm_KNN_train))/np.sum(cm_KNN_train)
print('k-NN train accuracy selected features: %0.4f' %acc_KNN_train)
print('k-NN test accuracy selected features: %0.4f' %acc_KNN_test)
#%% Naive-Bayes classification
cm_NB_train_all_features = np.zeros([5,5])
cm_NB_test_all_features = np.zeros([5,5])
for train, test in kf.split(X_selected):
    X_train = X.iloc[train,:]
    y_train = pd.to_numeric(y.iloc[train]).round(0).astype(int)
    X_test = X.iloc[test,:]
    y_test = pd.to_numeric(y.iloc[test]).round(0).astype(int)
    mdlNB = GaussianNB()
    mdlNB.fit(X_train,y_train)
    y_hat_test = mdlNB.predict(X_test)
    y_hat_train = mdlNB.predict(X_train)
    cm1 = confusion_matrix(y_test, y_hat_test, labels = [1,2,3,4,5])
    cm_NB_test_all_features += cm1
    cm1 = confusion_matrix(y_train, y_hat_train, labels = [1,2,3,4,5])
    cm_NB_train_all_features += cm1
plt.figure()
sns.heatmap(cm_NB_test_all_features.astype(int), cmap = plt.cm.Blues, annot=True, fmt="d")
acc_NB_test_all_features = np.sum(np.diag(cm_NB_test_all_features))/np.sum(cm_NB_test_all_features)
acc_NB_train_all_features = np.sum(np.diag(cm_NB_train_all_features))/np.sum(cm_NB_train_all_features)
print('Naive Bayes train accuracy all features: %0.4f' %acc_NB_train_all_features)
print('Naive Bayes test accuracy all features: %0.4f' %acc_NB_test_all_features)

cm_NB_train = np.zeros([5,5])
cm_NB_test = np.zeros([5,5])
for train, test in kf.split(X_selected):
    X_train = X.iloc[train,:]
    y_train = pd.to_numeric(y.iloc[train]).round(0).astype(int)
    X_test = X.iloc[test,:]
    y_test = pd.to_numeric(y.iloc[test]).round(0).astype(int)
    mdlNB = GaussianNB()
    mdlNB.fit(X_train,y_train)
    y_hat_test = mdlNB.predict(X_test)
    y_hat_train = mdlNB.predict(X_train)
    cm1 = confusion_matrix(y_test, y_hat_test, labels = [1,2,3,4,5])
    cm_NB_test += cm1
    cm1 = confusion_matrix(y_train, y_hat_train, labels = [1,2,3,4,5])
    cm_NB_train += cm1
plt.figure()
sns.heatmap(cm_NB_test.astype(int), cmap = plt.cm.Blues, annot=True, fmt="d")
acc_NB_test = np.sum(np.diag(cm_NB_test))/np.sum(cm_NB_test)
acc_NB_train = np.sum(np.diag(cm_NB_train))/np.sum(cm_NB_train)
print('Naive Bayes train accuracy: %0.4f' %acc_NB_train)
print('Naive Bayes test accuracy: %0.4f' %acc_NB_test)
#%% Random Forest classification
cm_RF_test_all_features = np.zeros([5,5])
cm_RF_train_all_features = np.zeros([5,5])
for train, test in kf.split(X_selected):
    X_train = X.iloc[train,:]
    y_train = pd.to_numeric(y.iloc[train]).round(0).astype(int)
    X_test = X.iloc[test,:]
    y_test = pd.to_numeric(y.iloc[test]).round(0).astype(int)
    mdlRF = RandomForestClassifier(300)
    mdlRF.fit(X_train,y_train)
    y_hat_test = mdlRF.predict(X_test)
    y_hat_train = mdlRF.predict(X_train)
    cm1 = confusion_matrix(y_test, y_hat_test, labels = [1,2,3,4,5])
    cm_RF_test_all_features += cm1
    cm1 = confusion_matrix(y_train, y_hat_train, labels = [1,2,3,4,5])
    cm_RF_train_all_features += cm1
plt.figure()
sns.heatmap(cm_RF_test_all_features.astype(int), cmap = plt.cm.Blues, annot=True, fmt="d")
acc_RF_test_all_features = np.sum(np.diag(cm_RF_test_all_features))/np.sum(cm_RF_test_all_features)
acc_RF_train_all_features = np.sum(np.diag(cm_RF_train_all_features))/np.sum(cm_RF_train_all_features)
print('Random Forest train accuracy all features: %0.4f' %acc_RF_train_all_features)
print('Random Forest test accuracy all features: %0.4f' %acc_RF_test_all_features)

cm_RF_test = np.zeros([5,5])
cm_RF_train = np.zeros([5,5])
for train, test in kf.split(X_selected):
    X_train = X.iloc[train,:]
    y_train = pd.to_numeric(y.iloc[train]).round(0).astype(int)
    X_test = X.iloc[test,:]
    y_test = pd.to_numeric(y.iloc[test]).round(0).astype(int)
    mdlRF = RandomForestClassifier(300)
    mdlRF.fit(X_train,y_train)
    y_hat_test = mdlRF.predict(X_test)
    y_hat_train = mdlRF.predict(X_train)
    cm1 = confusion_matrix(y_test, y_hat_test, labels = [1,2,3,4,5])
    cm_RF_test += cm1
    cm1 = confusion_matrix(y_train, y_hat_train, labels = [1,2,3,4,5])
    cm_RF_train += cm1
plt.figure()
sns.heatmap(cm_RF_test.astype(int), cmap = plt.cm.Blues, annot=True, fmt="d")
acc_RF_test = np.sum(np.diag(cm_RF_test))/np.sum(cm_RF_test)
acc_RF_train = np.sum(np.diag(cm_RF_train))/np.sum(cm_RF_train)
print('Random Forest train accuracy: %0.4f' %acc_RF_train)
print('Random Forest test accuracy: %0.4f' %acc_RF_test)