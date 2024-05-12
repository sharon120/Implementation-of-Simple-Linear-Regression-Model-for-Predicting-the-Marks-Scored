# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
/*
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.
*/
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Sharon Harshini L M
RegisterNumber: 212223040193

1.import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv("student_scores.csv")
df.head()

2.print(df.tail())

3.X=df.iloc[:,:-1].values
(X)

4.Y=df.iloc[:,-1].values
(Y)

5.from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
print(Y_pred)

6.print(Y_test)

7.plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

8.plt.scatter(X_test,Y_test,color="orange")
plt.plot(X_test,regressor.predict(X_test),color="red")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

9.mse=mean_squared_error(Y_test,Y_pred)
print("MSE = ",mse)

mae=mean_absolute_error(Y_test,Y_pred)
print("MAE = ",mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)
*/
```

## Output:
df.head()

![1](https://github.com/AkilaMohan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149555539/e26cc60c-9bda-4ccf-9968-183c43c9b767)

df.tail()

![2](https://github.com/AkilaMohan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149555539/45ce47ae-ec6e-45de-90aa-d087f8a94b0e)

Array of Values X

![image](https://github.com/AkilaMohan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149555539/bd0cbf3d-a37b-4f29-b50f-c7e6767768e4)

Array values of Y

![3](https://github.com/AkilaMohan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149555539/55fd63f2-5b4e-4d63-afec-307d175cb246)

Values of Y prediction

![4-2](https://github.com/AkilaMohan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149555539/cceab26b-c966-4cdb-9f11-d5dc756a494b)

Values of Y test

![5](https://github.com/AkilaMohan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149555539/3cba67eb-2b0e-42bc-bc1a-789af4cc1386)

Training set graph

![6](https://github.com/AkilaMohan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149555539/b10bdea6-1b8d-469f-b96c-ba7483ac7672)

Test set graph

![7](https://github.com/AkilaMohan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149555539/07716b41-2968-44aa-b48a-3dae9b337c8e)

Values of MSE,MAE and RMSE

![8](https://github.com/AkilaMohan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149555539/6fa43b0f-6519-4337-873b-49ace90c5365)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
