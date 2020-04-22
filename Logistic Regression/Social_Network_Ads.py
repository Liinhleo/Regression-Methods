import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

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

# Function visualize datapoint
def VisualizingDataset (X_,Y_):
    X1 = X_[:,0]
    X2 = X_[:,1]
    for i, label in enumerate(np.unique(Y_)):
        plt.scatter(X1[Y_==label],X2[Y_==label],
        color = ListedColormap(("red","green"))(i),
        label=label)
    plt.title('Classifier')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    
# Using class LogisticRegression to trainning model
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,Y_train)

# Function visualize result
def VisualizingResult(model,X_):
    X1=X_[:,0]
    X2=X_[:,1]
    X1_range = np.arange(start=X1.min()-1, stop=X1.max()+1, step=0.01)
    X2_range = np.arange(start=X2.min()-1, stop=X2.max()+1, step=0.01)
    X1_matrix, X2_matrix = np.meshgrid(X1_range,X2_range)
    X_grid = np.array([X1_matrix.ravel(), X2_matrix.ravel()]).T
    Y_grid = model.predict(X_grid).reshape(X1_matrix.shape)
    plt.contourf(X1_matrix, X2_matrix, Y_grid, alpha = 0.5,cmap = ListedColormap(("red", "green")))
    plt.title('Classifier')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()

# Visulize trainning model
VisualizingDataset(X_train,Y_train)
plt.show() 

# Visualizing the Training set results
VisualizingResult(classifier, X_train)
VisualizingDataset(X_train,Y_train)
plt.show()

# Visualizing the Training set results
VisualizingResult(classifier, X_test)
VisualizingDataset(X_test,Y_test)
plt.show()

# Using function confusion_matrix to compute matrix
cm = confusion_matrix(Y_train, classifier.predict(X_train))
print("\nTraindata\n", cm)
cm1 = confusion_matrix(Y_test, classifier.predict(X_test))
print("\nTestdata \n",cm1)