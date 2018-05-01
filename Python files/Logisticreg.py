import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from csv import reader
import matplotlib.pyplot as plt 
np.random.seed(0)

filename='/Users/naveen/Documents/Final_project/data_sets/finaltoday.csv' 
dataset=pd.read_csv(filename,low_memory=False)

datasets=np.array(dataset)
train_x=datasets[:np.shape(datasets)[0],:42]
train_y=datasets[:np.shape(datasets)[0],42]
print("No of samples(Number of frames" ,len(datasets))
print("No of features in input: ",len(train_x[0]))
classes=np.unique(train_y)
print("No of classes(waving , not waving and hand rising (two case) : ",classes)
X_train, X_test, Y_train, Y_test = train_test_split(train_x, train_y, test_size = 0.25, random_state = 0)	
print("Print training data and testing data length :" ,len(X_train),len(X_test))

logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
score=accuracy_score(Y_pred,Y_test)
print("Print Accuracy of predicting :",score)
print("Poses are not predicting correctly :")
for x in range(len(Y_test)):
	if Y_test[x]!=Y_pred[x]:
		print(Y_test[x],Y_pred[x],x)