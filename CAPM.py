#!/usr/bin/env python
# coding: utf-8

# # Capital Asset Pricing Model

# ## Milestone 1

# In[65]:


get_ipython().run_line_magic('reload_ext', 'dotenv')
get_ipython().run_line_magic('dotenv', '')
get_ipython().run_line_magic('matplotlib', 'inline')

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import yfinance as yf
from fredapi import Fred

sns.set()

FRED_API_KEY = os.getenv('FRED_API_KEY')


# Download the closing prices of the desired symbols.

# In[44]:


stocks_symbols = ['AAPL', 'IBM', 'MSFT', 'INTC', '^GSPC']
stocks = yf.download(stocks_symbols, start='2021-01-01', end='2021-04-01')
stocks = stocks['Close']
stocks = stocks.dropna()
stocks = stocks.rename(columns={'^GSPC': 'GSPC'})
stocks.describe()


# Search for risk free rate data in the Federal Reserve Economic Data:

# In[45]:


fred = Fred(api_key=FRED_API_KEY)
fred.search('risk free')


# We are interested in `DGS3MO`, as it is a government-issued and widely applicable rate.

# In[46]:


risk_free = fred.get_series('DGS3MO')
risk_free = risk_free['2021-01-01':'2021-04-01']


# ## Milestone 2

# The purpose of this milestone is to calculate excess returns. To do that, we need:
# - calculate stock returns as percentage
# - convert risk free rate to daily value
# - subtract each other
# 
# But first, some visualizations. Here are the trends of the downloaded stocks:

# In[47]:


fig, axes = plt.subplots(5, 1, sharex=True, figsize=(9, 8))
for i in range(5):
    sns.lineplot(data=stocks.iloc[:, i], ax=axes[i])


# Let's also visualize the correlation between stocks. See how similar are the trends for Intel and SP500. Apple on the other hand is quite different.

# In[48]:


sns.heatmap(stocks.corr(), annot=True)


# Let's start now start the journey to calculate excess returns. We start with the stock returns. Pandas provide a function `pct_change` to calculate those. We will just need to drop the first value as it will be `NaN` (as it cannot be compared to prior element).

# In[49]:


stock_returns = stocks.pct_change()
stock_returns = stock_returns.dropna()
stock_returns


# We've already downloaded `risk_free` data in Milestone 1. It spans over 3 months. Let's make it a daily rate and also delete the last row, because it goes one day over the end of our stocks data.

# In[50]:


risk_free_daily = risk_free / 90
risk_free_daily = risk_free_daily.dropna()
risk_free_daily = risk_free_daily.iloc[:-1]
sns.lineplot(data=risk_free_daily)


# In[51]:


excess_returns = stock_returns.sub(risk_free_daily, axis=0)
excess_returns


# ## Milestone 3

# In this milestone, we calculate the CAPM of our stocks with the help of the `statsmodels` library.

# In[68]:


endog = excess_returns['AAPL']
exog = sm.add_constant(excess_returns['GSPC'])
CAPM_AAPL = sm.OLS(endog, exog).fit()


# In[74]:


X = excess_returns['IBM']
y = sm.add_constant(excess_returns['GSPC'])
CAPM_IBM = sm.OLS(X, y).fit()


# In[79]:


X = excess_returns['INTC']
y = sm.add_constant(excess_returns['GSPC'])
CAPM_INTC = sm.OLS(X, y).fit()


# In[80]:


X = excess_returns['MSFT']
y = sm.add_constant(excess_returns['GSPC'])
CAPM_MSFT = sm.OLS(X, y).fit()


# ## Useful links
# 
# - [Risk Free Rate and Fama French factors](https://bizlib247.wordpress.com/2013/01/18/risk-free-rate-and-fama-french-factors/)
# - [`endog`, `exog`, what’s that?](https://www.statsmodels.org/stable/endog_exog.html)
# - [Ordinary Least Squares linear regression](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html#statsmodels.regression.linear_model.OLS)
# - [How to Version Control Jupyter Notebooks](https://nextjournal.com/schmudde/how-to-version-control-jupyter)
# 
