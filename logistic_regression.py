#digit regonition using logistic regression
#first do hyper parameter tuning to fing the best parameters
# then fit the data using those parameters
#draw roc curve
#find area of roc curve
#print confusion matrix and classification_report

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PyPDF2
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV



digits = datasets.load_digits()
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify = y, random_state = 21)
#using parameters as computed by logistic_regression_hypertuning.py
logreg = LogisticRegression(C= 0.051794746792312128, penalty = 'l1')
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)
print("the score of logistic regression is : " +str(logreg.score(X_test,y_test)))

print("##### classification report for KNN algorithm")
print("score : "+str(logistic_regression_hypertuning.score(X_test, y_test)))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


