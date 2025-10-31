import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk

df = pd.read_excel("breast_cancer.xlsx")
df = df.iloc[:, :-1]
print(df.shape)
df.head()

x=df.iloc[:,2:].values
y=df.diagnosis.values
print(x[:2])
print(y[:5])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,random_state=500)

x_train.shape #(455,30)
x_test.shape#(114, 30)
y_train.shape
y_test.shape


(y_train == 'M').sum()
(y_train=='B').sum()
278/len(y_train)


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
baseline_pred=["B"] *len(y_train) 
accuracy_score(y_train,baseline_pred) 
confusion_matrix(y_train,baseline_pred)
print(classification_report(y_train,baseline_pred))


from sklearn.naive_bayes import GaussianNB
nb_model=GaussianNB()
nb_model.fit(x_train,y_train)
print(x_train)
nb_model.score(x_train,y_train)
nb_model.score(x_test,y_test)

#confusion_matrix for training data
confusion_matrix(y_train,nb_model.predict(x_train))
#confusion_matrix for test data
confusion_matrix(y_test,nb_model.predict(x_test))


print(classification_report(y_train,nb_model.predict(x_train)))
print(classification_report(y_test,nb_model.predict(x_test)))
