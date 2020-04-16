import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


#read dataset
for i in range(1,6):
    train_name = 'SalesData_Train0'+str(i)+'.csv'
    test_name = 'SalesData_Test0'+str(i)+'.csv'
    datatrain = pd.read_csv(train_name)
    datatest = pd.read_csv(test_name)

    # Split dataset to input X_train/X_test and outcome Y_train/Y_test
    X_train = np.array(datatrain["Temperature"]).reshape(-1,1)
    Y_train = datatrain.iloc[:,-1].values

    X_test = np.array(datatest["Temperature"]).reshape(-1,1)
    Y_test = datatest.iloc[:,-1].values

    # ======================Preprocessing ================
    # Caculate x,x^2,x^3,x^4 
    poly_transform = PolynomialFeatures(degree=5)
    X_train_poly = poly_transform.fit_transform(X_train)
    X_test_poly =poly_transform.fit_transform(X_test)

    # Training model
    poly_lin_reg=LinearRegression()
    poly_lin_reg.fit(X_train_poly,Y_train)

    Y_test_poly_pred=poly_lin_reg.predict(X_test_poly)
    # =========================Visualize====================
    # Visualize training data
    plt.figure(figsize=(22,6))
    plt.subplot(1,3,1)
    plt.scatter(X_train,Y_train,color ="red")
    plt.title("Revenue prediction")
    plt.xlabel("Temperature")
    plt.ylabel("Revenue")

    # Visualize train 2 prediction (Da thuc)
    plt.subplot(1,3,2)
    X_train_dummy=np.arange(0,40,0.1).reshape(-1,1)
    X_train_dummy_poly=poly_transform.transform(X_train_dummy)
    Y_train_dummy_poly_pred=poly_lin_reg.predict(X_train_dummy_poly)
    plt.scatter(X_train,Y_train,color ="red")
    plt.plot(X_train_dummy,Y_train_dummy_poly_pred,color="blue")
    plt.title("Revenue prediction")
    plt.xlabel("Temperature")
    plt.ylabel("Revenue")

    # Visualize test 2 prediction (Da thuc)
    plt.subplot(1,3,3)
    X_test_dummy=np.arange(0,40,0.1).reshape(-1,1)
    X_test_dummy_poly=poly_transform.transform(X_test_dummy)
    Y_test_dummy_poly_pred=poly_lin_reg.predict(X_test_dummy_poly)
    plt.scatter(X_train,Y_train,color ="red")
    plt.scatter(X_test,Y_test,color ="black")
    plt.plot(X_test_dummy,Y_test_dummy_poly_pred,color="blue")
    plt.title("Revenue prediction")
    plt.xlabel("Temperature")
    plt.ylabel("Revenue")
    plt.savefig("poli_train0 "+ str(i) + ".png")
    plt.show()

    # save model to disk
    import pickle
    from math import sqrt
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error

    #Save model into .sav file
    filename = "D:\\Machine Learning & Statistic\\Sale_Icecream\\politrain0"+str(i)+".sav" #Name the model
    pickle.dump(poly_lin_reg, open(filename, 'wb'))

    # load the model from disk
    print("=========================")
    print(Y_test_poly_pred)
    loaded_model = pickle.load(open("D:\\Machine Learning & Statistic\\Sale_Icecream\\politrain0"+str(i)+".sav", 'rb'))
    result = loaded_model.predict(X_test_poly)
    print(result)
    print("=========================")

    # Evaluation
    print("RMSE: ",sqrt(mean_squared_error(Y_test, Y_test_poly_pred)))
    print("R2: ", r2_score(Y_test, Y_test_poly_pred))

