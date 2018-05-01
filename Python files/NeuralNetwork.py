import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

filename='/Users/naveen/Documents/Final_project/data_sets/thankprakash.csv'
dataset=pd.read_csv(filename)
datasets=dataset
dataset=np.array(dataset)

data=dataset[:np.shape(dataset)[0],:42]
labels =dataset[:np.shape(dataset)[0],42]

# Split the dataset in two equal parts
train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size = 0.20, random_state = 0)

from keras.utils import to_categorical

print('Training dataset shape : ',train_x.shape ,train_y.shape)
print('Testing dataset shape : ',test_x.shape,test_y.shape)

#find the unique from the train labels
classes=np.unique(train_y)
nclasses=len(classes)
print('Print number of classes :',nclasses)
print('Print classes :',classes)
train_data=train_x
test_data=test_x
train_data=train_data.astype('float32')
test_data=test_data.astype('float32')

#Convert the labels from integer to categorical data
train_labels_one_hot=to_categorical(train_y)
test_labels_one_hot=to_categorical(test_y)
print('Original label 0: ',train_y[0])
print('After conversion to categorical :',train_labels_one_hot[0])

#Neural Network
#Sequential model, we can just stack up layers by adding the desired layer one by one
#We use the Dense layer, also called fully connected layer since 
#we are building a feedforward network in which all the neurons from one layer are connected to the neurons in the previous layer
# Dense layer, we add the ReLU activation function which is required to introduce non-linearity to the model
#The last layer is a softmax layer as it is a multiclass classification problem
#binary classification, we can use sigmoid.
#local data

dimData=42
#Create the Network
from keras.models import Sequential
from keras.layers import Dense

#Add Regularization to the model
# dropout, a fraction of neurons is randomly turned off during the training process, reducing the dependency on the training set by some amount.
from keras.layers import Dropout
model_reg = Sequential()
model_reg.add(Dense(512, activation='relu', input_shape=(dimData,)))
model_reg.add(Dropout(0.5))
model_reg.add(Dense(512, activation='relu'))
model_reg.add(Dropout(0.5))
model_reg.add(Dense(nclasses, activation='softmax'))

model_reg.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history_reg = model_reg.fit(train_data, train_labels_one_hot, batch_size=256, epochs=10, verbose=1, 
                            validation_data=(test_data, test_labels_one_hot))

# Predict the most likely class
# model_reg.predict_classes(test_data[[0],:])
# Predict the probabilities for each class 
print(model_reg.predict(test_data))
print(history_reg)

#Plot the Loss Curves
plt.figure(figsize=[8,6])
plt.plot(history_reg.history['loss'],'r',linewidth=3.0)
plt.plot(history_reg.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
 
#Plot the Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(history_reg.history['acc'],'r',linewidth=3.0)
plt.plot(history_reg.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
plt.show()
