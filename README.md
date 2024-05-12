# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## Aim:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
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

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Sharon Harshini L M
RegisterNumber: 212223040193

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("/content/student_scores.csv")
df.head()

x=df.iloc[:,:-1].values
x

y=df.iloc[:,1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

y_train
y_pred

plt.scatter(x_train,y_train,color="yellow")
plt.plot(x_train,regressor.predict(x_train),color="purple")
plt.title("Hours VS Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print("MSE= ",mse)
mae=mean_absolute_error(y_test,y_pred)
print("MAE = ",mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)

```

## Output:
df.head()

![1](https://github.com/sharon120/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149555539/74bd1c81-f9a3-44aa-ba32-4ba6d837a0cd)

df.tail()

![2](https://github.com/sharon120/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149555539/8a85a802-5e9e-4af6-9411-37a1802ce561)

Array Values of X

![image](https://github.com/sharon120/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149555539/b5815546-ce58-4b55-888b-ec343f90856f)

Array Values of Y

![3](https://github.com/sharon120/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149555539/cf25e0cb-2999-49ac-9623-63c3fb5689f7)

Values of Y Prediction

![4-2](https://github.com/sharon120/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149555539/17a39ba7-36a4-4cb6-8188-a2b97c3f24db)

Values of Y test

![5](https://github.com/sharon120/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149555539/389e65bf-9aee-4810-a920-ad764012f2c6)

Training set graph

![6](https://github.com/sharon120/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149555539/2fad3d6c-f24c-4eaf-9cad-e2b9acfb7739)

Test set graph

![7](https://github.com/sharon120/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149555539/41087550-2fcf-4f90-9c8b-2f44d06b9014)

Values of MSE,MAE and RMSE

![8](https://github.com/sharon120/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149555539/1526af70-9fc9-4214-83a3-9ea313a3b598)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
