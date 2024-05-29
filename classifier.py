# Develop a classifier model to determine the values of the variable 'Revenue' 
# You can use any algorithms addressed in the class. 
# You have to turn in the source codes, the developed model, 
# and the accuracy of the developed model.  

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from keras.models import Sequential
from keras.layers import Dense,Dropout
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, roc_auc_score,classification_report

df = pd.read_csv('./data/shopping_data.csv')

df = pd.get_dummies(df, drop_first=True)
# Load and preprocess the data
# Assuming that 'df' is the preprocessed DataFrame and 'Revenue' is the target variable
X = df.drop('Revenue', axis=1)
y = df['Revenue']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# smote = SMOTE(random_state=42,sampling_strategy='minority')
# X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
std_scalar=StandardScaler()
std_scaled=std_scalar.fit(X_train).transform(X_train)
test_scaled=std_scalar.transform(X_test)

# Logistic  Regression
# model = LogisticRegression()

#Random Forest
# model = RandomForestClassifier(n_estimators=100)

# KNN Classifier

# for each in range(1,15):
#     model=KNeighborsClassifier(n_neighbors=each)
#     model.fit(std_scaled, y_train)
#     predictions = model.predict(test_scaled)

#     # Compute and print the accuracy of the model
#     accuracy = accuracy_score(y_test, predictions)
#     print("Accuracy of the developed model: {:.2f}%".format(accuracy * 100))

# Support Vector Machine
# model=SVC(random_state=42)

# Naive Bayes
# model= GaussianNB()

# XGBoost
    
param_grid={
    'n_estimators': [100, 200, 300],
    'max_depth': [2, 3, 4],
    'learning_rate': [0.01, 0.1, 0.2]
}
model=xgb.XGBClassifier()
grid_search=GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy')
# Fit the model to the training data
grid_search.fit(std_scaled, y_train)
best_params=grid_search.best_params_
print("Best Parameters : ",best_params)
# Make predictions on the test data
model=xgb.XGBClassifier(**best_params)
model.fit(std_scaled, y_train)
predictions = model.predict(test_scaled)

# Compute and print the accuracy of the model
accuracy = accuracy_score(y_test, predictions)
print("Accuracy of the developed model: {:.2f}%".format(accuracy * 100))


# Sequential Model Neural Network

# model = Sequential()
# model.add(Dense(32, input_dim=std_scaled.shape[1], activation='relu'))
# model.add(Dense(16, activation='relu',kernel_regularizer='l2'))
# model.add(Dense(1, activation='sigmoid'))

# # Compile the model
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# # Fit the model to the training data
# model.fit(std_scaled, y_train, epochs=50, batch_size=32, validation_data=(test_scaled, y_test))

# # Make predictions on the test data
# predictions = model.predict(test_scaled)
# predictions=np.round(predictions).flatten()
# # Compute and print the accuracy of the model
# accuracy = accuracy_score(y_test, predictions)
# print("Accuracy of the developed model: {:.2f}%".format(accuracy * 100))

# model.fit(std_scaled, y_train)
# predictions = model.predict(test_scaled)

# # Compute and print the accuracy of the model
# accuracy = accuracy_score(y_test, predictions)
# print("Accuracy of the developed model: {:.2f}%".format(accuracy * 100))

print(classification_report(y_test, predictions))
