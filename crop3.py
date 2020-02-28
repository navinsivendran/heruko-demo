# Load libraries

#from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option

from pandas import read_csv
from pandas import set_option
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
url = 'dataset crop.csv'
dataset = read_csv(url, header=None)
# shape
print(dataset.shape)
# types
set_option('display.max_rows', 500)
print(dataset.dtypes)
# head
set_option('display.width', 100)
print(dataset.head(20))

# descriptions, change precision to 3 places
set_option('precision', 3)
print(dataset.describe())
array = dataset.values
X = array[:,0:9].astype(float)
Y = array[:,9]
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)
#from sklearn.preprocessing import MinMaxScaler

from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, Y_train)
print('Accuracy of SVM classifier on training set: {:.2f}'
     .format(svm.score(X_train, Y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'
     .format(svm.score(X_test, Y_test)))

import pickle
pickle.dump(svm, open('crop3.pkl', 'wb'))
model = pickle.load(open('crop3.pkl','rb'))
result1= model.predict([[-0.5,	-0.3,	-0.7,	-0.7,	1.8,44,61,	57,	38	]])
print(result1)
