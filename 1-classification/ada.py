import numpy as np
from numpy import mean
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import pickle
from joblib import dump
from joblib import load

pre = np.loadtxt('predict.txt', delimiter=' ')

Xpre=pre[:,0:13]

with open('class.pickle', 'rb') as f:
    mod=pickle.load(f)

Ypre = mod.predict(Xpre)

print(Ypre)
