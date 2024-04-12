import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
# credit.policy, purpose,int.rate, installment, log.annual.inc,
# dti, fico,days.with.cr.line, revol.bal, revol.util, inq.last.6mths,
# delinq.2yrs, pub.rec,not.fully.paid

# Load the data
data = pd.read_csv('./data/loan_data.csv')
data = pd.get_dummies(data, drop_first=True)

# Split into features and target
X = data.drop('not.fully.paid', axis=1)
y = data['not.fully.paid']
plt.figure(figsize=(8, 6))
sns.countplot(x='not.fully.paid', data=data)
plt.title('Class Distribution')
plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Oversample the minority class using SMOTE
smote = SMOTE(random_state=42,sampling_strategy='minority')
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
# Create a logistic regression model
model = LogisticRegression()
# Train the model 
model.fit(X_train_smote, y_train_smote) 
# model.fit(X_train, y_train)
# Make predictions on the testing data
predictions = model.predict(X_test)
# Print a classification report 
print(classification_report(y_test, predictions))

# Concatenate the DataFrames
oversampled_data = pd.concat([X_train_smote, y_train_smote], axis=1)
# non_oversample_data=pd.concat([X_train, y_train], axis=1)
pred_prob = model.predict_proba(X_test)

fpr,tpr,threshold=roc_curve(y_test, pred_prob[:,1],pos_label=1)
random_probs=[0 for i in range(len(y_test))]
pfpr,ptpr,_=roc_curve(y_test, random_probs,pos_label=1)
auc_score=roc_auc_score(y_test, pred_prob[:,1])
print('AUC value : ',auc_score)


# Plot the distribution of the 'not.fully.paid' classes after SMOTE
plt.figure(figsize=(8, 6))
# sns.countplot(x='not.fully.paid', data=oversampled_data)
sns.countplot(x='not.fully.paid', data=oversampled_data)
plt.title('Class Distribution After SMOTE')
plt.show()

# plt.style.use('seaborn')
plt.plot(fpr,tpr,linestyle='--',color='orange',label='Logistic Regression')
plt.plot(pfpr,ptpr,linestyle='--',color='blue')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')

plt.legend(loc='best')
plt.show()