import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

#read dataset
datatrain=pd.read_csv("Position_SalariesTrain.csv")
datatest=pd.read_csv("Position_SalariesTest.csv")

# Split dataset to input X_train/X_test and outcome Y_train/Y_test
X_train=datatrain.iloc[:,1:-1].values
Y_train=datatrain.iloc[:,-1].values
X_test=datatest.iloc[:,1:-1].values
Y_test=datatest.iloc[:,-1].values


# ======================Preprocessing ================
# Caculate x,x^2,x^3,x^4 
poly_transform = PolynomialFeatures(degree=5)
X_train_poly = poly_transform.fit_transform(X_train)

# Training model
poly_lin_reg=LinearRegression()
poly_lin_reg.fit(X_train_poly,Y_train)

# =========================Visualize====================
# Visualize training data
plt.figure(figsize=(22,6))
plt.subplot(1,3,1)
plt.scatter(X_train,Y_train,color ="red")
plt.title("Position Level vs Salary")
plt.xlabel("Position Level")
plt.ylabel("Salary (dollars/year)")

# Visualize train 2 prediction (Da thuc)
plt.subplot(1,3,2)
X_train_dummy=np.arange(0,11,0.1).reshape(-1,1)
X_train_dummy_poly=poly_transform.transform(X_train_dummy)
Y_train_dummy_poly_pred=poly_lin_reg.predict(X_train_dummy_poly)
plt.scatter(X_train,Y_train,color ="red")
plt.plot(X_train_dummy,Y_train_dummy_poly_pred,color="blue")
plt.title("Position Level vs Salary")
plt.xlabel("Position Level")
plt.ylabel("Salary (dollars/year)")

# Visualize test 2 prediction (Da thuc)
plt.subplot(1,3,3)
X_test_dummy=np.arange(0,11,0.1).reshape(-1,1)
X_test_dummy_poly=poly_transform.transform(X_test_dummy)
Y_test_dummy_poly_pred=poly_lin_reg.predict(X_test_dummy_poly)
plt.scatter(X_train,Y_train,color ="red")
plt.scatter(X_test,Y_test,color ="black")
plt.plot(X_test_dummy,Y_test_dummy_poly_pred,color="blue")
plt.title("Position Level vs Salary")
plt.xlabel("Position Level")
plt.ylabel("Salary (dollars/year)")
plt.savefig("visualize.png")
plt.show()

#Compare
def compare(i_example):
    x=X_test[i_example:i_example+1]
    x_poly = poly_transform.transform(x)
    y=Y_test[i_example]
    y_pred = poly_lin_reg.predict(x_poly)
    print ("y=" , y)
    print ("y_pred=" , y_pred)


for i in range(len(X_test)):
    compare(i)
