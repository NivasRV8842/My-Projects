#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sklearn.datasets

breast_cancer=sklearn.datasets.load_breast_cancer()
print(breast_cancer)

X=breast_cancer.data
Y=breast_cancer.target
print(X)
print(Y)

print(X.shape,Y.shape)

import pandas as pd
data=pd.DataFrame(breast_cancer.data,columns=breast_cancer.feature_names)
data['Class']=breast_cancer.target
data.head()

data.describe()

print(data['Class'].value_counts())
print(breast_cancer.target_names)

data.groupby('Class').mean

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y)
print(Y.shape,Y_train.shape,Y_test.shape)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1)
print(Y.shape,Y_train.shape,Y_test.shape)
print(Y.mean(),Y_train.mean(),Y_test.mean())

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,stratify=Y)
print(Y.mean(),Y_train.mean(),Y_test.mean())

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=1)
print(X_train)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()

classifier.fit(X_train,Y_train)

from sklearn.metrics import accuracy_score

prediction_on_training_data=classifier.predict(X_train)
accuracy_on_training_data=accuracy_score(Y_train,prediction_on_training_data)
print("Accuracy on training Data:",accuracy_on_training_data)

prediction_on_test_data=classifier.predict(X_test)
accuracy_on_testing_data=accuracy_score(Y_test,prediction_on_test_data)
print("Accuracy on testing Data:",accuracy_on_testing_data)

#detectinng whether it is Benign or Malignant stage
input_data=(20.57,17.77,132.9,1326,0.08474,0.07864,0.0869,0.07017,0.1812,0.05667,0.5435,0.7339,3.398,74.08,0.005225,0.01308,0.0186,0.0134,0.01389,0.003532,24.99,23.41,158.8,1956,0.1238,0.1866,0.2416,0.186,0.275,0.08902)
#input is tuple-change it to numpy array
input_data_as_numpy_array=np.asarray(input_data)
print(input_data)

input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

prediction=classifier.predict(input_data_reshaped)
print(prediction)

if(prediction[0]==0):
  print("The Affected Person is in Malignant Stage")
else:
  print("The Affected Person is in Benign Stage")


# In[ ]:




