# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## Aim:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Import necessary libraries (e.g., pandas, numpy,matplotlib).
2. Load the dataset and then split the dataset into training and testing sets using sklearn library.
3. Create a Linear Regression model and train the model using the training data (study hours as input, marks scored as output).
4. Use the trained model to predict marks based on study hours in the test dataset.
5. Plot the regression line on a scatter plot to visualize the relationship between study hours and marks scored.


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

![3](https://github.com/sharon120/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149555539/a79d91e0-9b90-42e8-9314-acab124b9a4a)
![1](https://github.com/sharon120/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149555539/cac65fb0-d3b8-4ff2-85bb-713840984d2f)
![2](https://github.com/sharon120/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149555539/a20158d1-2c07-4e09-976b-75f797caac73)
![4](https://github.com/sharon120/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149555539/1c67da0c-4ff8-4c47-b196-e0a507ed0567)
![5](https://github.com/sharon120/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149555539/558d85d7-7d6f-44b0-8f18-e29943a4d0f5)
![6](https://github.com/sharon120/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149555539/b156e9d6-df58-443f-a0b5-f614cc2c7d3d)
![7](https://github.com/sharon120/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149555539/90cdeb65-e11c-46c6-b155-9609aeb50fe0)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
