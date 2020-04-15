import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Read dataset
datatrain=pd.read_csv("BostonTrain.csv")
datatest=pd.read_csv("BostonTest.csv")

for i in range(13):
    # Split dataset to input X and outcome Y
    X_train = np.array(datatrain.iloc[:, i]).reshape(-1,1)
    Y_train = datatrain.iloc[:,-1].values

    X_test = np.array(datatest.iloc[:, i]).reshape(-1,1)
    Y_test = datatest.iloc[:,-1].values

    # Get name of column
    i_title = datatrain.columns[i]

    # Trainning data => module regressor 
    regressor = LinearRegression()
    regressor.fit(X_train,Y_train)

    # Visualize training data
    plt.figure(figsize=(18,6))
    plt.subplot(1,3,1)
    plt.scatter(X_train,Y_train,color="red")
    plt.title("Boston house prices")
    plt.xlabel(i_title)
    plt.ylabel("MEVD")

    # Visualize prediction
    Y_train_pred = regressor.predict(X_train)
    plt.subplot(1,3,2)
    plt.scatter(X_train,Y_train,color="red")
    plt.plot(X_train,Y_train_pred,color="blue")
    plt.title("Boston house prices")
    plt.xlabel(i_title)
    plt.ylabel("MEVD")

    #  Visualize testing
    Y_test_pred=regressor.predict(X_test)
    plt.subplot(1,3,3)
    plt.scatter(X_test,Y_test,color="red")
    plt.plot(X_test,Y_test_pred,color="blue")
    plt.scatter(X_test,Y_test_pred,color="black")
    plt.title("Boston house prices")
    plt.xlabel(i_title)
    plt.ylabel("MEVD")
    
    plt.savefig( i_title + " vs MEVD.png")
    plt.show()

    # Evaluation
    print("R square coef (" + i_title + ") =", r2_score(Y_test, Y_test_pred))
   
