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
from sklearn.model_selection import RandomizedSearchCV



digits = datasets.load_digits()
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify = y, random_state = 21)

#we will do hyperparameter tuning to find the best parameter using gridsearchCV
c_space = np.logspace(-5,8,15)
param_grid= {'C' : c_space, 'penalty':['l1','l2']}
logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg, param_grid, cv = 5)
logreg_cv.fit(X_train,y_train)

print("the best parameter for logistic regression using paramter hypertuning(GridSearchCV) is : " +str(logreg_cv.best_params_))
print("the best score for logistic regression using parameter hypertuning(GridSearchCV) is : "+str(logreg_cv.best_score_))


#the out og this program is
#the best parameter for logistic regression using paramter hypertuning is : {'penalty': 'l1', 'C': 0.051794746792312128}
#the best score for logistic regression using parameter hypertuning is : 0.969769291965
#we will use these parameters for our logistic regression implemented in logistic_regression.py file


#hyperparameter tuning using randomized searchCV
logreg_cv = RandomizedSearchCV(logreg,param_grid,cv = 5)

logreg_cv.fit(X_train,y_train)

print("the best parameter for logistic regression using paramter hypertuning(RandomizedSearchCV) is : " +str(logreg_cv.best_params_))
print("the best score for logistic regression using parameter hypertuning(RandomizedSearchCV) is : "+str(logreg_cv.best_score_))



