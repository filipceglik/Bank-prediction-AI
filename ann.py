# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_Gender = LabelEncoder()
X[:, 2] = labelencoder_X_Gender.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#importing keras

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

#Initialising Networt
#this is network object
classifier = Sequential()

#Adding the input layer and hidden layer

classifier.add(Dense(6, kernel_initializer= "uniform", activation='relu', input_dim=11))
classifier.add(Dropout(rate = 0.1))

#Adding second hidden layer

classifier.add(Dense(6, kernel_initializer= "uniform", activation='relu'))
classifier.add(Dropout(rate = 0.1))
#Adding output layer

classifier.add(Dense(1, kernel_initializer= "uniform", activation='sigmoid'))


classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])


# Fitting classifier to the Training set
classifier.fit(X_train, y_train, batch_size=10, epochs=100)




# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

"""
Predict for
Country: France
Credit score: 600
Gender: Male
Age: 40
Tenure: 3
Ballance: 60000
Number of products: 2
Credit card: yes
Active client: yes
Est. Salary: 50000
"""

new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Evaluating ANN

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(6, kernel_initializer= "uniform", activation='relu', input_dim=11))
    classifier.add(Dense(6, kernel_initializer= "uniform", activation='relu'))
    classifier.add(Dense(1, kernel_initializer= "uniform", activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, nb_epoch = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)

mean = accuracies.mean()
variance = accuracies.std()

#Tuning the ANN

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(6, kernel_initializer= "uniform", activation='relu', input_dim=11))
    classifier.add(Dense(6, kernel_initializer= "uniform", activation='relu'))
    classifier.add(Dense(1, kernel_initializer= "uniform", activation='sigmoid'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
#accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
parameters = {'batch_size': [25, 32],
              'nb_epoch': [500, 100],
              'optimizer': ['rmsprop', 'adam']}
grid_search = GridSearchCV(estimator = classifier,param_grid = parameters, scoring = 'accuracy', cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_param = grid_search.best_params_
best_acc = grid_search.best_score_







