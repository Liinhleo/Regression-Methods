import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from numpy import sqrt

for i in range(1,6):
    i_new = str(i)
    # Read dataset
    dataset = pd.read_csv("Bike_Train0" + i_new + ".csv")
    dataset1 = pd.read_csv("Bike_Test0" + i_new + ".csv")
    
    # Split dataset to input - X and outcome - Y
    X_train = dataset.iloc[:, [9, 10, 12]].values
    Y_train = dataset.iloc[:, 14].values
    
    X_test = dataset1.iloc[:, [9, 10, 12]].values
    Y_test = dataset1.iloc[:, 14].values

    # Training data
    poly_lin_reg = LinearRegression()
    poly_lin_reg.fit(X_train, Y_train)
    
    Y_train_pred = poly_lin_reg.predict(X_train)
    Y_test_pred = poly_lin_reg.predict(X_test)
    
    # Evaluation
    print("r2 score (Train0" + i_new + ")", poly_lin_reg.score(X_train, Y_train))
    print("r2 score (Test0" + i_new + ")", poly_lin_reg.score(X_test, Y_test))
    print("RMSE (Train0" + i_new + ")", sqrt(metrics.mean_squared_error(Y_train, Y_train_pred)))
    print("RMSE (Test0" + i_new + ")", sqrt(metrics.mean_squared_error(Y_test, Y_test_pred)))

    # save model to disk
    import pickle
   
    #Save model into .sav file
    filename = "D:\\Machine Learning & Statistic\\Bike_Hire\\multitrain0"+str(i)+".sav" #Name the model
    pickle.dump(poly_lin_reg, open(filename, 'wb'))

    # load the model from disk
    print("=========================")
    print(Y_test_pred)
    loaded_model = pickle.load(open( "D:\\Machine Learning & Statistic\\Bike_Hire\\multitrain0"+str(i)+".sav", 'rb'))
    result = loaded_model.predict(X_test)
    print(result)
    print("=========================")
    