# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import Necessary Libraries and Load iris Data set

2.Create a DataFrame from the Dataset

3.Add Target Labels to the DataFrame

4.Split Data into Features (X) and Target (y)

5.Split Data into Training and Testing Sets

6.Initialize the SGDClassifier Model

7.Train the Model on Training Data

8.Make Predictions on Test Data

9.Evaluate Accuracy of Predictions

10.Generate and Display Confusion Matrix

11.Generate and Display Classification Report
## Program // Output:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Vedagiri Indu Sree
RegisterNumber:  212223230236
*/
```
```
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
iris=load_iris()

df=pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target']=iris.target
print(df.head())

x=df.drop('target',axis=1)
y=df['target']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
sgd_clf=SGDClassifier(max_iter=1000,tol=1e-3)
sgd_clf.fit(x_train,y_train)
y_pred=sgd_clf.predict(x_test)

accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.3f}")
```
![image](https://github.com/user-attachments/assets/0d0f358b-9229-4593-8269-1ef0f4e85d87)

```
cm=confusion_matrix(y_test,y_pred)
print("Confufion Matrix:")
print(cm)
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
print("Name:Vedagiri Indu Sree")
print("Reg No:212223230236")
```
![image](https://github.com/user-attachments/assets/717f9f73-642d-4788-a0c1-ad1001d3d8c2)

![image](https://github.com/user-attachments/assets/cfad1ed6-bac4-44d0-990b-916dbfe100cf)

## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
