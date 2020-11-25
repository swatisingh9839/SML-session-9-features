import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import sklearn.linear_model as skl_lm

from matplotlib.patches import Arc

import itertools
import math

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False


# Read in the data
#happy = pd.read_csv("data/happy.csv",delimiter=';') 
happy = pd.read_csv('https://uu-sml.github.io/course-sml-public/data/happy.csv', delimiter=';')
happy.head()

# Rename some columns
happy.rename(columns = {
    'Perceptions of corruption':'Corruption',
    'Log GDP per capita': 'LogGDP',
    'Healthy life expectancy at birth': 'LifeExp',
    'Freedom to make life choices': 'Freedom',
}, inplace = True) 

# In this exercise we will just analyse one year. 2017.
df = happy[happy['Year'] == 2017].dropna()
df.head()

#Fit model.
X_train = df[['Social support']]
y_train = df['Life Ladder']
model = skl_lm.LinearRegression(fit_intercept=False)
model.fit(X_train, y_train)

# Print the solution
print(f'The coefficient is: {model.coef_[0]:.3f}')

#Compute predictions.
x = np.arange(0.25, 1, step=0.01)
X_test = x.reshape(-1, 1)
y_test = model.predict(X_test)

# Plot social support and life ladder data
plt.plot('Social support', 'Life Ladder', 'o', data=df, color='gray')
countries = ['United Kingdom','Croatia', 'Benin', 'Finland',
             'Afghanistan']
for country in countries:
    ci = np.where(df['Country name'] == country)[0][0]
    plt.plot(df.iloc[ci]['Social support'],
             df.iloc[ci]['Life Ladder'], 'ko')
    plt.annotate(country,
                 xy=(df.iloc[ci]['Social support'],
                     df.iloc[ci]['Life Ladder']),
                 xytext=(3, 3),  # 3 points offset
                 textcoords="offset points",
                 ha='left', va='bottom')

# Plot model
plt.plot(x, y_test, 'k')
plt.ylabel('Life Satisfaction (y)')
plt.xlabel('Social support (s)')
plt.show()


factors = ['LogGDP', 'Social support', 'LifeExp',  'Freedom',
           'Generosity', 'Corruption']

# Fit regression model
X = df[factors]
y = df['Life Ladder']
model = skl_lm.LinearRegression()
model.fit(X, y)

# Print the solution
print('The coefficients are:', model.coef_)
print(f'The offset is: {model.intercept_:.3f}')

# Compute predictions
y_hat = model.predict(X)

# Compute AIC
def aic(model, y, y_hat):
    # Numbers of parameters of linear regression model
    # Variance of Gaussian noise is a parameter as well!
    k = model.coef_.size + model.get_params()['fit_intercept'] + 1
    
    # Compute maximum log-likelihood
    n = y.size
    mse = np.mean((y - y_hat)**2)
    loglik = - n / 2 * (1 + math.log(2 * math.pi) + np.log(mse))
    
    return 2 * (k - loglik)

print(f'The AIC is: {aic(model, y, y_hat):.3f}')
