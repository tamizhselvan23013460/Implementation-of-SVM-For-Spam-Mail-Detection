# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import required libraries for encoding detection, data handling, text processing, machine learning, and evaluation.

2.Open the dataset file (spam.csv) in binary mode and detect its character encoding using chardet.

3.Load the CSV file into a pandas DataFrame using the detected encoding (e.g., 'windows-1252').

4.Display the dataset using head(), info(), and check for any missing values with isnull().sum().

5.Assign the label column (v1) to variable x and the message text column (v2) to variable y.

6.Split the dataset into training and testing sets using an 80-20 split with train_test_split().

7.Convert the text data into numerical format using CountVectorizer to prepare for model training.

8.Initialize the Support Vector Machine classifier (SVC) and train it on the vectorized training data.

9.Predict the labels of the test set using the trained SVM model.

10.Evaluate the model’s performance by calculating and printing the accuracy score using accuracy_score().

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: TAMIZHSELVAN B
RegisterNumber:  2122232302225
import chardet
file='spam.csv'
with open (file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result
import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')
data.head()
data.info()
data.isnull().sum()
x=data["v2"].values
y=data["v1"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
*/
```

## Output:
### Reading a dataset
![image](https://github.com/user-attachments/assets/92ed5dc9-83d3-46d9-97ee-5c645a392f82)


### data.head()
![image](https://github.com/user-attachments/assets/e470c1b5-a03a-4d84-856e-cfa814735540)


### data.info()
![image](https://github.com/user-attachments/assets/c1055a54-f374-474b-b417-95892445a39f)


### data.isnull().sum()
![image](https://github.com/user-attachments/assets/f8026648-2ff9-46b1-b02a-61194191e7e7)

### Y_pred 

![image](https://github.com/user-attachments/assets/9c23dd20-59f9-4245-ba42-9bf6b99dd01c)



### Calculating accuracy:
![image](https://github.com/user-attachments/assets/eb649cb4-702a-4d6e-ac08-356e2c22cfd5)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
