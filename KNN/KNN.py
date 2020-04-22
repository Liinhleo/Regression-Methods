import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

# Read dataset
datatrain = pd.read_csv("Social_Network_Ads_Train.csv")
datatest = pd.read_csv("Social_Network_Ads_Test.csv")

# Split dataset to X input and outcome Y
X_train = datatrain.iloc[:,[3,4]].values
Y_train = datatrain.iloc[:,5].values

X_test = datatest.iloc[:,[3,4]].values
Y_test = datatest.iloc[:,5].values

# Standalize data (mean = 0 , p.sai = 1)
SC = StandardScaler()
X_train = SC.fit_transform(X_train)
X_test = SC.fit_transform(X_test)

# Visualize data
def VisualizeDataset (X_,Y_):
    X1 = X_[:,0]
    X2 = X_[:,1]
    for i, label in enumerate(np.unique(Y_)):
        plt.scatter(X1[Y_==label],X2[Y_==label],color=ListedColormap(("red","green"))(i))

# Visualize Datatrain
VisualizeDataset(X_train,Y_train)
plt.show()

# Visualize Datatest
VisualizeDataset(X_test,Y_test)
plt.show()

def VisualizeResult(model, X_):
        X1 = X_[:, 0]
        X2 = X_[:, 1]
        X1_range = np.arange(start= X1.min()-1, stop= X1.max()+1,step = 0.01)
        X2_range = np.arange(start= X2.min()-1, stop= X2.max()+1,step = 0.01)
        X1_matrix, X2_matrix = np.meshgrid(X1_range, X2_range)
        X_grid= np.array([X1_matrix.ravel(), X2_matrix.ravel()]).T
        Y_grid= model.predict(X_grid).reshape(X1_matrix.shape)
        plt.contourf(X1_matrix, X2_matrix, Y_grid, alpha = 0.5,cmap = ListedColormap(("red", "green")))

a = {3,5,7,9}
for i in a:
    #Huấn luyện mô hình
    classifier =KNeighborsClassifier(n_neighbors=i)
    classifier.fit(X_train,Y_train)

    # Confusion matrix
    cm=confusion_matrix(Y_train,classifier.predict(X_train))
    print(cm)
    cm=confusion_matrix(Y_test,classifier.predict(X_test))
    print(cm)  

    #Visualize datatrain result
    VisualizeResult(classifier, X_train)
    VisualizeDataset(X_train,Y_train)
    plt.show()    
 
    #Visualize datatest result
    VisualizeResult(classifier, X_test)
    VisualizeDataset(X_test, Y_test)
    plt.show()   



