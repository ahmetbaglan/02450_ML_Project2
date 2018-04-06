import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
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


df = pd.read_csv('OurDatabase.csv')

l = list(df.columns)
attributesToShow = l[5:-2]
X = df.as_matrix(columns = attributesToShow)
N = len(df)
X.shape

stdVector = np.array(df[attributesToShow].std())
stdVector = np.expand_dims(stdVector,axis = 0)
X = X / stdVector
X = X - np.ones((N,1))*X.mean(0)

