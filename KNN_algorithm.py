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


#Here i am working with the MNIST digits recognition dataset, which has 10 classes, the digits 0 through 9!
#A reduced version of the MNIST dataset is one of scikit-learn's included datasets, and that is the one i will use.
#Each sample in this scikit-learn dataset is an 8x8 image representing a handwritten digit.
#Each pixel is represented by an integer in the range 0 to 16, indicating varying levels of black.
#For MNIST dataset, scikit-learn provides an 'images' key in addition to the 'data' and 'target' keys. Because it is a 2D array of the images
#corresponding to each sample, this 'images' key is useful for visualizing the images. On the other hand, the 'data' key contains the feature array
#the images as a flattened array of 64 pixels. I have used the K Nearest Neighbors algorithm




digits = datasets.load_digits()
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify = y, random_state = 21)

def learn(k) :
    """this function is used to find the efficiency of algorithm for different values of k and just prints them"""
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train,y_train)
    print(knn.score(X_test, y_test))

def num_predict(i,k) :
    """recognises the number whose data is passed and prints it on console. It also saves the image in the same directory for crosschecking"""
    plt.imshow(digits.images[i], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.savefig('fig1.pdf')
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train,y_train)
    num = knn.predict([digits.data[i]])
    return num





learn(2) #for checking the efficiency for any values of k
ans = num_predict(1015,7) # here we are passing 2 arguments, the index of data of image that we are trying to predict and k
print("the recognised value is = "+ str(ans))




#the following code draws a line plot between efficiency and different values of k for both testing and training data

neighbors = np.arange(1,20)
train_accuracy = np.empty(len(neighbors))
test_accuracy=np.empty(len(neighbors))

for i,k in enumerate(neighbors) :
    knn = KNeighborsClassifier(k)
    knn.fit(X,y)
    train_accuracy[i] = knn.score(X_train,y_train)
    test_accuracy[i] = knn.score(X_test,y_test)
    plt.clf()


plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')

plt.show(block = True)
plt.savefig('efficiencycomparison.png')

# the image is saved as png file in the working directory