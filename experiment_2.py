import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import statsmodels.api as sm
from scipy import stats

plt.style.use('seaborn-darkgrid')


def DisplayAnalysis(data,desc='Energy per hour',var='energy'):
    #### Normality analysis
    fig, axes = plt.subplots(2,2)
    ax = axes.ravel()
    # we fit mean and variance to the data
    mu, sigma = stats.norm.fit(data)
    print('Mean: %s and Std: %s' % (mu,sigma))
    # theorical values of normal distribution in the observed range
    print(min(data))
    print(max(data))
    x_hat = np.linspace(min(data),max(data), num=100)
    y_hat = stats.norm.pdf(x_hat, mu, sigma)

    ax[0].plot(x_hat, y_hat, linewidth=2, label='normal')
    ax[0].hist(x=data, density=True, bins=40, color="#3182bd", alpha=0.5)
    ax[0].plot(data, np.full_like(data, -0.01), '|k', markeredgewidth=1)
    ax[0].set_title(desc)
    ax[0].set_xlabel(var)
    ax[0].set_ylabel('Probaility Distribution')
    ax[0].legend()

    ## Q-Q plot
    sm.qqplot(data,fit=True,line='q',alpha=0.4,lw=2,ax=ax[1])
    ax[1].set_title('Q-Q plot of %s' % var)
    ax[1].tick_params(labelsize=7)
    # Boxplot
    ax[2].set_title(desc)
    ax[2].boxplot(data)

    # Analytical methods
    kurt = stats.kurtosis(data)
    skewness = stats.skew(data)
    # Hypothesis Contrast
    shapiro_test = stats.shapiro(data)
    k2, p_value = stats.normaltest(data)

    ax[3].text(0.1, 0.8, 'Kurtosis = %4.2f'% kurt, dict(size=9))
    ax[3].text(0.1, 0.7, 'Skewness = %4.2f'% skewness, dict(size=9))
    ax[3].text(0.1, 0.6, 'Shapiro test p-value = %4.5f'% shapiro_test[1], dict(size=9))
    ax[3].text(0.1, 0.5, "D'Agostino test p_value = %4.5f"% p_value, dict(size=9))
    ax[3].axis('off')
    ax[3].grid(False)

    plt.show()


#### Dataset 2
raw_data = pd.read_csv('./data/energy_per_hour.csv',sep=';',header=0)
print(raw_data)
energy = raw_data['Energy']
print(energy)

# Display normality analysis of raw data
DisplayAnalysis(energy)

## 
## Applying manual transformations
# y = sqrt(x)
energy_sqrt = energy.copy()
energy_sqrt = energy_sqrt.apply(np.sqrt)
DisplayAnalysis(energy_sqrt,desc='Sqrt of energy',var='sqrt(energy)')
# y = 1/x
energy_inv = energy.copy()
energy_inv = energy_inv.apply(lambda x: 1/x)
DisplayAnalysis(energy_inv,desc='Inverse of energy',var='inv(energy)')
# y = Ln(x)
energy_natlog = energy.copy()
energy_natlog = energy_natlog.apply(lambda x: np.log(x))
DisplayAnalysis(energy_natlog,desc='Natural Log of energy',var='Ln(energy)')
# y = x^2
energy_squared = energy.copy()
energy_squared = energy_squared.apply(lambda x: x**2)
DisplayAnalysis(energy_squared,desc='Square of energy',var='energy^2')

## Applying Box-cox to transform the data to a closer normal dist.
energy_boxcox = energy.copy()
energy_boxcox, _ = stats.boxcox(energy_boxcox)
DisplayAnalysis(energy_boxcox, desc='BoxCox transformed energy', var='BoxCox(energy)')