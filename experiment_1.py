import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import style
#import seaborn as sns

import statsmodels.api as sm
from scipy import stats

plt.style.use('seaborn-darkgrid')

#import warnings
#warnings.filterwarnings('ignore')

'''
    Two datasets are explored:
        Dataset 1 - Men and women's height and weight recordings
                    Source: https://raw.githubusercontent.com/JoaquinAmatRodrigo/Estadistica-machine-learning-python/master/data/Howell1.csv
        Dataset 2 - Solar Energy process data
                    Source: https://support.minitab.com/en-us/minitab/19/help-and-how-to/quality-and-process-improvement/control-charts/how-to/box-cox-transformation/before-you-start/example/

'''

#### Dataset 1
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

# Filtering the data
raw_data = pd.read_csv('./data/raw_data.csv')
print(raw_data.head(10))
filtered_data = raw_data.copy()
filtered_data = filtered_data[(filtered_data.age > 15) & filtered_data.male == 1]
#print(filtered_data.head(5))
#all_data = raw_data['weight']
data = filtered_data['weight']
print(data)

### Graphical representations of data
#
# we fit mean and variance to the data
mu, sigma = stats.norm.fit(data)
print('Mean: %s and Std: %s' % (mu,sigma))
# theorical values of normal distribution in the observed range
print(min(data))
print(max(data))
x_hat = np.linspace(min(data),max(data), num=100)
y_hat = stats.norm.pdf(x_hat, mu, sigma)

fig, ax = plt.subplots(figsize=(7,4))
ax.plot(x_hat, y_hat, linewidth=2, label='normal')
ax.hist(x=data, density=True, bins=40, color="#3182bd", alpha=0.5)
ax.plot(data, np.full_like(data, -0.01), '|k', markeredgewidth=1)
ax.set_title('Weight distribution (men older than 15 years old)')
ax.set_xlabel('weight')
ax.set_ylabel('Probaility Distribution')
ax.legend()
plt.show()

# using a boxplot
fig, ax = plt.subplots(figsize=(7,4))
ax.set_title('Weight distribution (men older than 15 years old)')
ax.boxplot(data)
plt.show()

### Using Q-Q plot
fig, ax = plt.subplots(figsize=(7,4))
sm.qqplot(data,fit=True,line='q',alpha=0.4,lw=2,ax=ax)
ax.set_title('Q-Q plot of weights of men older than 15 years old')
ax.tick_params(labelsize=7)
plt.show()

## Analytical methods
# Kurtosis analysis
print('Kurtosis: %s' % stats.kurtosis(data))
# Skewness
print('Skewness: %s' % stats.skew(data))

## Hypothesis Contrast
# Shapiro-Wilk test
shapiro_test = stats.shapiro(data)
print(shapiro_test)
# D'Agostino's K-squared test
k2, p_value = stats.normaltest(data)
print('Statistic = %s, p-value= %s' % (k2,p_value))



