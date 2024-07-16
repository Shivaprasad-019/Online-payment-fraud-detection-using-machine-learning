PROJECT NAME:ONLINE PAYMENT FRAUD DETECTION USING MACHINE LEARNING

Abstract:

As online transactions become more prevalent, the risk of fraudulent activities also increases, posing 
significant security threats. The increased usage of online payments is leading to a rise in fraud. Fraud 
detection is an important component of online payment systems since it serves to protect both customers 
and merchants from financial damages. This project aims to address online payment fraud detection using 
advanced machine learning techniques. By analyzing transaction data—such as transaction type, amount, 
sender and receiver details, and account balances. Our approach involves training the model on a diverse 
dataset to recognize patterns and anomalies indicative of fraud. We will explore various machine learning 
algorithms. The performance of these algorithms will be evaluated using metrics like accuracy, precision, 
recall, and F1-score to identify the most effective model. The ultimate goal is to enhance the security of 
online payment systems, fostering trust and confidence among users. By effectively detecting and 
mitigating fraudulent transactions, we aim to protect individual users and bolster the integrity of digital 
financial systems. This will support the continued growth and adoption of online payment methods, 
ensuring safer digital transactions for all stakeholders.

Keywords:

 Machine learning Techniques, fraudulent transactions, Safer digital Transactions, Security 
enhancement

Source code:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
data=pd.read_csv("online (1).csv")
data.head()
data.isnull().sum()
data.shape
data.type.value_counts()
data.info()
data.isFraud.value_counts()
data['type']=data['type'].map({'PAYMENT':1,'CASH_IN':2,'CASH_OUT':3,'TRANSFER':4,'DEBIT':5}
)
data['isFraud']=data['isFraud'].map({0:'No Fraud',1:'Fraud'})
print(data.head())
data.plot()
sb.pairplot(data, hue="isFraud")
plt.show()
plt.hist(data['isFraud'])
plt.show()
x = np.array(data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
y = np.array(data[["isFraud"]])
print(x)
print(y)
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)
model=RandomForestClassifier()
model.fit(xtrain,ytrain)
ypred=model.predict(xtest)
ypred
accuracy_score(ytest,ypred)
confusion_matrix(ytest,ypred)
print(classification_report(ytest,ypred))
model1=SVC()
model1.fit(xtrain,ytrain)
ypred1=model1.predict(xtest)
ypred1
accuracy_score(ytest,ypred1)
confusion_matrix(ytest,ypred1)
print(classification_report(ytest,ypred1))
model3=KNeighborsClassifier(n_neighbors=7)
model3.fit(xtrain,ytrain)
ypred3=model3.predict(xtest)
ypred3
model4=AdaBoostClassifier(n_estimators=7)
model4.fit(xtrain,ytrain)
ypred4=model4.predict(xtest)
ypred4
accuracy_score(ytest,ypred4)
confusion_matrix(ytest,ypred4)
print(classification_report(ytest,ypred4))
model5=DecisionTreeClassifier()
model5.fit(xtrain,ytrain)
ypred5=model5.predict(xtest)
ypred5
accuracy_score(ytest,ypred5)
confusion_matrix(ytest,ypred5)
print(classification_report(ytest,ypred5))
model6=LogisticRegression()
model6.fit(xtrain,ytrain)
ypred6=model6.predict(xtest)
accuracy_score(ytest,ypred6)
confusion_matrix(ytest,ypred6)
print(classification_report(ytest,ypred6))

Platform to implement the project:

JUPYTER NOTEBOOK
