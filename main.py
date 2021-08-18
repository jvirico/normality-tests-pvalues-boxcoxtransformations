import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import style
#import seaborn as sns

import statsmodels.api as sm
from scipy import stats

plt.style.use('ggplot')

import warnings
warnings.filterwarnings('ignore')

# Data
url = ('https://raw.githubusercontent.com/JoaquinAmatRodrigo/' +
       'Estadistica-machine-learning-python/master/data/Howell1.csv')
datos = pd.read_csv(url)
print(datos.info())
datos.head(4)