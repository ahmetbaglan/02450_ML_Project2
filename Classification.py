
# coding: utf-8

# In[16]:


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
sns.set(style="whitegrid", palette="muted")
current_palette = sns.color_palette()
from scipy.io import loadmat
import scipy.linalg as linalg
import numpy as np
from scipy.stats import norm
from sklearn import tree
import graphviz
import sklearn.linear_model as lm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn import model_selection



# Let's load the data

# In[2]:


df = pd.read_csv('OurDatabase.csv')


# Let's see column names

# In[3]:


df.columns


# Let's actually see the data 

# In[4]:


df


# # Creating X and Y

# Let's get the X matrix and show shapes

# In[5]:


l = list(df.columns)
attributesToShow = l[5:-2]
X = df.as_matrix(columns = attributesToShow)
N = len(df)
X.shape


# Let's standirdize our X matrix

# In[6]:


stdVector = np.array(df[attributesToShow].std())
stdVector = np.expand_dims(stdVector,axis = 0)
X = X / stdVector
X = X - np.ones((N,1))*X.mean(0)


# In[7]:


y_adv = df['isDevelopedEconomy'].as_matrix().copy()
y_adv[y_adv=='d'] = 1
y_adv[y_adv=='n'] = 0
y_adv=y_adv.astype('int')


# In[8]:


y_cont = df['Continent'].as_matrix().copy()
continents = list(set(y_cont))
for i in continents:
    y_cont[y_cont == i] = continents.index(i)
y_cont=y_cont.astype('int')


# # Decision Trees


# Okey

treeModels = [{'purity':p,'minSplit':m} for p in ['gini','entropy'] for m in range(2,100)]


options_decision_tree = ['gini', 'entropy']
options_knn = [2,4,5,6,7,8,9]


K_fold = 5
CV_out = model_selection.KFold(K_fold, shuffle=True)

Error_train = np.empty((K_fold, 3))
Error_test = np.empty((K_fold, 3))
k = 0
# You can set dist=1 to obtain manhattan distance (cityblock distance).
dist = 2

for train_index, test_index in CV_out.split(X, y_cont):
    internal_cross_validation = 5
    CV_in = model_selection.KFold(internal_cross_validation, shuffle=True)
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y_cont[train_index]
    X_test = X[test_index]
    y_test = y_cont[test_index]
    param_err_dec_tree = []

    for o in treeModels:
        err_inner = 0
        for train_index_inner, test_index_inner in CV_in.split(X_train, y_train):
            X_train_inner = X_train[train_index_inner]
            y_train_inner = y_train[train_index_inner]
            X_test_inner = X[test_index]
            y_test_inner = y_cont[test_index]
            dtc = tree.DecisionTreeClassifier(criterion=o['purity'],min_samples_split = o['minSplit'] )
            dtc = dtc.fit(X_train_inner, y_train_inner)
            y_est_inner = dtc.predict(X_test_inner)
            err_inner += sum(np.abs(y_est_inner != y_test_inner))
        param_err_dec_tree.append(err_inner)
    best_param_dec_tree = treeModels[np.argmin(param_err_dec_tree)]
    best_dtc = tree.DecisionTreeClassifier(criterion=best_param_dec_tree['purity'], min_samples_split= best_param_dec_tree['minSplit'])
    best_dtc = best_dtc.fit(X_train, y_train)
    y_est = best_dtc.predict(X_test)
    err_test =  sum(np.abs(y_est != y_test))
    Error_test[k,0] = float(err_test) / y_test.shape[0]


    param_err_knn = []
    for o in options_knn:
        err_inner = 0
        for train_index_inner, test_index_inner in CV_in.split(X_train, y_train):
            X_train_inner = X_train[train_index_inner]
            y_train_inner = y_train[train_index_inner]
            X_test_inner = X[test_index]
            y_test_inner = y_cont[test_index]
            # Fit classifier and classify the test points
            knclassifier = KNeighborsClassifier(n_neighbors=o)
            knclassifier.fit(X_train_inner, y_train_inner)
            y_est_inner = knclassifier.predict(X_test_inner)
            err_inner += sum(np.abs(y_est_inner != y_test_inner))
        param_err_knn.append(err_inner)

    best_param_knn = options_knn[np.argmin(param_err_knn)]
    best_knn = KNeighborsClassifier(n_neighbors=best_param_knn)
    best_knn = best_knn.fit(X_train, y_train)
    y_est = best_knn.predict(X_test)
    err_test = sum(np.abs(y_est != y_test))
    Error_test[k, 1] = float(err_test) / y_test.shape[0]

    # LOGISTIC REGRESSION
    model = lm.logistic.LogisticRegression()
    model = model.fit(X_train, y_train)
    y_est = model.predict(X_test)
    err_test = sum(np.abs(y_est != y_test))
    Error_test[k, 2] = float(err_test) / y_test.shape[0]
    k+=1

for i in range(3):
    print(Error_test[:,i].mean())

