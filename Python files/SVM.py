import numpy as np
import pandas as pd
import random
import csv
from csv import reader
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
np.random.seed(0)
import pickle 
filename='/Users/naveen/Documents/Final_project/data_sets/thankprakash.csv'
dataset=pd.read_csv(filename,low_memory=False)

datasets=np.array(dataset)
#split the data in features and lebals wise
X=datasets[:np.shape(datasets)[0],:42]
Y=datasets[:np.shape(datasets)[0],42]


print("No of samples(Number of frames" ,len(datasets))
print("No of features in input: ",len(X[0]))
classes=np.unique(Y)
print("No of classes(waving , not waving and hand rising (two case) : ",classes)

# shuffle the data np.random 
X,Y=shuffle(X,Y,random_state=0)
# Split the dataset in two equal parts
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)	

print("Print training data and testing data length :" ,len(X_train),len(X_test))
clf1 = svm.SVC(kernel='linear', gamma=0.7, C=1.0)
clf1.fit(X_train, Y_train)

Y_pred = clf1.predict(X_test)

score=accuracy_score(Y_pred,Y_test)
print("Print Accuracy of predicting :",score)

print("Poses are not predicting correctly :")

for x in range(len(Y_test)):
	if Y_test[x]!=Y_pred[x]:
		print(Y_test[x],Y_pred[x])

