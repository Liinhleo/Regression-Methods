import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#read dataset
dataset=pd.read_csv("Salary_Data.csv")
print(dataset.shape)

# Split dataset to input X and outcome Y
X=np.array(dataset.iloc[:,0].values).reshape(-1,1)
Y=np.array(dataset.iloc[:,1].values)


# Split datatset to trainning data & testing data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.8,random_state=0)

# Visualize training data
plt.scatter(X_train,Y_train,color ="red")
plt.title("Salary vs Experiment")
plt.xlabel("Experiment (years)")
plt.ylabel("Salary (dollars/year)")
plt.show()

# Trainning data => module regressor 
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

#  Visualize prediction
Y_train_pred = regressor.predict(X_train)
plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,Y_train_pred,color="blue")
plt.title("Salary vs Experiment")
plt.xlabel("Experiment (years)")
plt.ylabel("Salary (dollars/year)")
plt.show()

#  Visualize testing
Y_test_pred=regressor.predict(X_test)
plt.scatter(X_test,Y_test,color="red")
plt.plot(X_test,Y_test_pred,color="blue")
plt.scatter(X_test,Y_test_pred,color="black")
plt.title("Salary vs Experiment")
plt.xlabel("Experiment (years)")
plt.ylabel("Salary (dollars/year)")
plt.show()

# Compare 1 datapoint in testing data
def compare (i_example):
    x = X_test[i_example:i_example +1]
    y = Y_test[i_example]
    y_pred = regressor.predict(x)
    print(x,y,y_pred)

for i in range(len(X_test)):
    compare(i)