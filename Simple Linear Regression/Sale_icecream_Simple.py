import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Read dataset
datatrain=pd.read_csv("SalesData_Train05.csv")
datatest=pd.read_csv("SalesData_Test05.csv")

# Split dataset to input X and outcome Y
X_train = np.array(datatrain.iloc[:,0:1]).reshape(-1,1)
Y_train = datatrain.iloc[:,-1].values

X_test = np.array(datatest.iloc[:,0:1]).reshape(-1,1)
Y_test = datatest.iloc[:,-1].values

# Trainning data => module regressor 
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

# Visualize training data
plt.figure(figsize=(18,6))
plt.subplot(1,3,1)
plt.scatter(X_train,Y_train,color="red")
plt.title("Revenue prediction")
plt.xlabel("Temperature")
plt.ylabel("Revenue")

# Visualize prediction
Y_train_pred = regressor.predict(X_train)
plt.subplot(1,3,2)
plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,Y_train_pred,color="blue")
plt.title("Revenue prediction")
plt.xlabel("Temperature")
plt.ylabel("Revenue")

#  Visualize testing
Y_test_pred=regressor.predict(X_test)
plt.subplot(1,3,3)
plt.scatter(X_test,Y_test,color="red")
plt.plot(X_test,Y_test_pred,color="blue")
plt.scatter(X_test,Y_test_pred,color="black")
plt.title("Revenue prediction")
plt.xlabel("Temperature")
plt.ylabel("Revenue")

plt.savefig( "Revenue5.png")
plt.show()

# Evaluation
 # Evaluation
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

print("RMSE: ",sqrt(mean_squared_error(Y_test, Y_test_pred)))
print("r2_score (R2): ", r2_score(Y_test, Y_test_pred))

