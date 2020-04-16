import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd

#read dataset
datatrain=pd.read_csv("kc_house_Train.csv")
datatest=pd.read_csv("kc_house_Test.csv")

#=================================================
# # Get 5 row head/tail
# dataset.head(5)
# dataset.tail(5)

# # Find maximum value in column dataframe and get all row values using
# dataset[dataset.bedrooms==dataset.bedrooms.max()]
# df_max=pd.DataFrame()
# df_min=pd.DataFrame()
# for i in dataset.columns:
#     df_max=df_max.append(pd.DataFrame(dataset[dataset[i]==dataset[i].max()]))
#     df_min=df_min.append(pd.DataFrame(dataset[dataset[i]==dataset[i].max()]))
# df_max
# df_min
#=================================================

independent_var_col = [4,5,6,7,8,13,14]

for i in independent_var_col:
    # # Split dataset to input X_train/X_test and outcome Y_train/Y_test
    X_train = np.array(datatrain.iloc[:, i].values).reshape(-1,1)
    Y_train = datatrain['price']
    X_test = np.array(datatest.iloc[:, i].values).reshape(-1,1)
    Y_test = datatest['price']

    #get name of column
    i_title = datatrain.columns[i]

    # Visualize training data
    plt.figure(figsize=(19,6))
    plt.subplot(1,3,1)
    plt.scatter(X_train,Y_train,color ="red")
    plt.title("House Sales in King County, USA")
    plt.xlabel(i_title )
    plt.ylabel("Price")

    # Trainning data => module regressor 
    regressor = LinearRegression()
    regressor.fit(X_train,Y_train)

    # Visualize prediction
    plt.subplot(1,3,2)
    Y_train_pred = regressor.predict(X_train)
    plt.scatter(X_train,Y_train,color="red")
    plt.plot(X_train,Y_train_pred,color="blue")
    plt.title("House Sales in King County, USA")
    plt.xlabel(i_title )
    plt.ylabel("Price")

    #  Visualize testing
    plt.subplot(1,3,3)
    Y_test_pred=regressor.predict(X_test)
    plt.scatter(X_test,Y_test,color="red")
    plt.plot(X_test,Y_test_pred,color="blue")
    plt.scatter(X_test,Y_test_pred,color="black")
    plt.title("House Sales in King County, USA")
    plt.xlabel(i_title )
    plt.ylabel("Price")
    plt.savefig( i_title + " vs price.png")
    plt.show()

    # Evaluation
    from math import sqrt
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error

    print("RMSE: ",sqrt(mean_squared_error(Y_test, Y_test_pred)))
    print("MSE: ",mean_squared_error(Y_test, Y_test_pred))
    print("MAE: ",mean_absolute_error(Y_test, Y_test_pred))
    print("r2_score (R2): ", r2_score(Y_test, Y_test_pred))


# import seaborn as sb
# f,ax=plt.subplots(figsize=(20,20))
# sb.heatmap(datatrain.corr(), annot = True)

# plt.show()
# datatrain.hist(bins = 20, figsize = (20,20), color = 'g')
# plt.show()