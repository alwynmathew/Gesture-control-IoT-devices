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
print(int(Y_pred[0]))
inputnum=-1
if Y_pred==1:
	inputnum=0
if Y_pred==3:
	inputnum=1;
# if int(Y_pred[0])==1||int(Y_pred[0])==2:
import socket
# s = socket.socket()         # Create a socket object
host = '172.16.29.251' # Get local machine name
#host=socket.gethostname()
port = 12346            # Reserve a port for your service.
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((host, port))
# Y_pred=1
s.sendall(b'%d' %inputnum)
data = s.recv(1024)
s.close()
print('rec', repr(data))
