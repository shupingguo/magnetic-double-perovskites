import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump
from joblib import load
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import zero_one_loss
from sklearn.metrics import mean_squared_error
from matplotlib import rcParams
import matplotlib.font_manager as font_manager
import matplotlib
import pickle

pre = np.loadtxt('predict.txt', delimiter=' ')

Xpre=pre[:,0:38]


with open('90-ada.pickle', 'rb') as f:
    mod=pickle.load(f)

Ypre = mod.predict(Xpre)

print(Ypre)

