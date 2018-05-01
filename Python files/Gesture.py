import numpy as np
# import matplotlib.pyplot as plt 
import pandas as pd
import pickle

# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
# from sklearn import linear_model, datasets
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import classification_report
# from sklearn.utils import shuffle
# from sklearn import preprocessing
# from sklearn.linear_model import LogisticRegression
# np.random.seed(0)
import csv
filename1 = '/Users/pk1601cs33/thank.sav'
# filename='/Users/pk1601cs33/kinect_project/naveencreatefile.csv'
# pp=float(sys.argv[0])
# print(pp)
import numpy as np
import sys

# val1 = float(sys.argv[1]) 
# val2 = float(sys.argv[2]) 
# val3 = float(sys.argv[3]) 
# val4 = float(sys.argv[4]) 
# val5 = float(sys.argv[5]) 
# val6 = float(sys.argv[6])
l=[] 
for i in range(1,43):
	l.append(float(sys.argv[i]))
# print(len(l))
# data=pd.read_csv(filename)
p=[]
p.append(l)
# print(len(data))
l=np.array(p)
# l=data[100:101,:42]
# p=data[100:101,42]
# print(l.dtype)
# print(l[22])
# l=l[10001:10100,:42]
# print(l,p)
loaded_model = pickle.load(open(filename1, 'rb'))

Y_pred = loaded_model.predict(l)
# print(int(Y_pred[0]))

if int(Y_pred[0])==0:
	print("STANDING")
if int(Y_pred[0])==1:
	print("RISING HAND")
if int(Y_pred[0])==3:
	print("TWO HAND ")
if int(Y_pred[0])==2:
	print("NAMASTEY")
