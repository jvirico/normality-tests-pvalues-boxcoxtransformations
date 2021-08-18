import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import style
#import seaborn as sns

import statsmodels.api as sm
from scipy import stats

plt.style.use('ggplot')

import warnings
#warnings.filterwarnings('ignore')

# Data
'''
url = ('https://raw.githubusercontent.com/JoaquinAmatRodrigo/' +
       'Estadistica-machine-learning-python/master/data/Howell1.csv')
raw_data = pd.read_csv(url)
raw_data.to_csv('./data/raw_data.csv')
print(raw_data.info())
raw_data.head(4)

filtered_data = raw_data[(raw_data.age > 15) & (raw_data.male == 0)]
weight = filtered_data['weight']
weight.to_csv('./data/filtered_data_weight.csv')
'''

data = pd.read_csv('./data/filtered_data_weight.csv')
print(data.head(10))