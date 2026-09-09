#!/usr/bin/env python
# coding: utf-8

# # Housing Price Prediction Case Study

# ## Multiple Linear Regression
# 
# ### Problem Statement:
# 
# Consider a real estate company that has a dataset containing the prices of properties in the Delhi region. It wishes to use the data to optimise the sale prices of the properties based on important factors such as area, bedrooms, parking, etc.
# 
# Essentially, the company wants —
# 
# 
# - To identify the variables affecting house prices, e.g. area, number of rooms, bathrooms, etc.
# 
# - To create a linear model that quantitatively relates house prices with variables such as number of rooms, area, number of bathrooms, etc.
# 
# - To know the accuracy of the model, i.e. how well these variables can predict house prices.
# 
# ### Data
# Use housing dataset.

# ## Reading and Understanding the Data

# In[1]:


# Supress Warnings

import warnings
warnings.filterwarnings('ignore')

# Import the numpy and pandas package

import numpy as np
import pandas as pd

# Data Visualisation

import matplotlib.pyplot as plt 
import seaborn as sns


# In[2]:


housing = pd.DataFrame(pd.read_csv("Housing.csv"))


# In[3]:


# Check the head of the dataset
housing.head()


# ## Data Inspection

# In[4]:


housing.shape


# In[5]:


housing.info()


# In[6]:


housing.describe()


# ## Data Cleaning

# In[7]:


# Checking Null values
housing.isnull().sum()*100/housing.shape[0]
# There are no NULL values in the dataset, hence it is clean.


# In[8]:


# Outlier Analysis
fig, axs = plt.subplots(2,3, figsize = (10,5))
plt1 = sns.boxplot(housing['price'], ax = axs[0,0])
plt2 = sns.boxplot(housing['area'], ax = axs[0,1])
plt3 = sns.boxplot(housing['bedrooms'], ax = axs[0,2])
plt1 = sns.boxplot(housing['bathrooms'], ax = axs[1,0])
plt2 = sns.boxplot(housing['stories'], ax = axs[1,1])
plt3 = sns.boxplot(housing['parking'], ax = axs[1,2])

plt.tight_layout()


# In[9]:


# Outlier Treatment
# Price and area have considerable outliers.
# We can drop the outliers as we have sufficient data.


# In[10]:


# outlier treatment for price
plt.boxplot(housing.price)
Q1 = housing.price.quantile(0.25)
Q3 = housing.price.quantile(0.75)
IQR = Q3 - Q1
housing = housing[(housing.price >= Q1 - 1.5*IQR) & (housing.price <= Q3 + 1.5*IQR)]


# In[11]:


# outlier treatment for area
plt.boxplot(housing.area)
Q1 = housing.area.quantile(0.25)
Q3 = housing.area.quantile(0.75)
IQR = Q3 - Q1
housing = housing[(housing.area >= Q1 - 1.5*IQR) & (housing.area <= Q3 + 1.5*IQR)]


# In[12]:


# Outlier Analysis
fig, axs = plt.subplots(2,3, figsize = (10,5))
plt1 = sns.boxplot(housing['price'], ax = axs[0,0])
plt2 = sns.boxplot(housing['area'], ax = axs[0,1])
plt3 = sns.boxplot(housing['bedrooms'], ax = axs[0,2])
plt1 = sns.boxplot(housing['bathrooms'], ax = axs[1,0])
plt2 = sns.boxplot(housing['stories'], ax = axs[1,1])
plt3 = sns.boxplot(housing['parking'], ax = axs[1,2])

plt.tight_layout()


# ## Exploratory Data Analytics
# 
# Let's now spend some time doing what is arguably the most important step - **understanding the data**.
# - If there is some obvious multicollinearity going on, this is the first place to catch it
# - Here's where you'll also identify if some predictors directly have a strong association with the outcome variable

# ### Visualising Numeric Variables
# 
# Let's make a pairplot of all the numeric variables

# In[13]:


sns.pairplot(housing)
plt.show()


# #### Visualising Categorical Variables
# 
# As you might have noticed, there are a few categorical variables as well. Let's make a boxplot for some of these variables.

# In[14]:


plt.figure(figsize=(20, 12))
plt.subplot(2,3,1)
sns.boxplot(x = 'mainroad', y = 'price', data = housing)
plt.subplot(2,3,2)
sns.boxplot(x = 'guestroom', y = 'price', data = housing)
plt.subplot(2,3,3)
sns.boxplot(x = 'basement', y = 'price', data = housing)
plt.subplot(2,3,4)
sns.boxplot(x = 'hotwaterheating', y = 'price', data = housing)
plt.subplot(2,3,5)
sns.boxplot(x = 'airconditioning', y = 'price', data = housing)
plt.subplot(2,3,6)
sns.boxplot(x = 'furnishingstatus', y = 'price', data = housing)
plt.show()


# We can also visualise some of these categorical features parallely by using the `hue` argument. Below is the plot for `furnishingstatus` with `airconditioning` as the hue.

# In[15]:


plt.figure(figsize = (10, 5))
sns.boxplot(x = 'furnishingstatus', y = 'price', hue = 'airconditioning', data = housing)
plt.show()


# ## Data Preparation

# - You can see that your dataset has many columns with values as 'Yes' or 'No'.
# 
# - But in order to fit a regression line, we would need numerical values and not string. Hence, we need to convert them to 1s and 0s, where 1 is a 'Yes' and 0 is a 'No'.

# In[16]:


# List of variables to map

varlist =  ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']

# Defining the map function
def binary_map(x):
    return x.map({'yes': 1, "no": 0})

# Applying the function to the housing list
housing[varlist] = housing[varlist].apply(binary_map)


# In[17]:


# Check the housing dataframe now

housing.head()


# ### Dummy Variables

# The variable `furnishingstatus` has three levels. We need to convert these levels into integer as well. 
# 
# For this, we will use something called `dummy variables`.

# In[18]:


# Get the dummy variables for the feature 'furnishingstatus' and store it in a new variable - 'status'
status = pd.get_dummies(housing['furnishingstatus'])


# In[19]:


# Check what the dataset 'status' looks like
status.head()


# Now, you don't need three columns. You can drop the `furnished` column, as the type of furnishing can be identified with just the last two columns where — 
# - `00` will correspond to `furnished`
# - `01` will correspond to `unfurnished`
# - `10` will correspond to `semi-furnished`

# In[20]:


# Let's drop the first column from status df using 'drop_first = True'

status = pd.get_dummies(housing['furnishingstatus'], drop_first = True)


# In[21]:


# Add the results to the original housing dataframe

housing = pd.concat([housing, status], axis = 1)


# In[22]:


# Now let's see the head of our dataframe.

housing.head()


# In[23]:


# Drop 'furnishingstatus' as we have created the dummies for it

housing.drop(['furnishingstatus'], axis = 1, inplace = True)


# In[24]:


housing.head()


# ### Splitting the Data into Training and Testing Sets

# In[25]:


from sklearn.model_selection import train_test_split

# We specify this so that the train and test data set always have the same rows, respectively
np.random.seed(0)
df_train, df_test = train_test_split(housing, train_size = 0.7, test_size = 0.3, random_state = 100)


# ### Rescaling the Features 
# 
# As you saw in the demonstration for Simple Linear Regression, scaling doesn't impact your model. Here we can see that except for `area`, all the columns have small integer values. So it is extremely important to rescale the variables so that they have a comparable scale. If we don't have comparable scales, then some of the coefficients as obtained by fitting the regression model might be very large or very small as compared to the other coefficients. This might become very annoying at the time of model evaluation. So it is advised to use standardization or normalization so that the units of the coefficients obtained are all on the same scale. As you know, there are two common ways of rescaling:
# 
# 1. Min-Max scaling 
# 2. Standardisation (mean-0, sigma-1) 
# 
# This time, we will use MinMax scaling.

# In[26]:


from sklearn.preprocessing import MinMaxScaler


# In[27]:


scaler = MinMaxScaler()


# In[28]:


# Apply scaler() to all the columns except the 'yes-no' and 'dummy' variables
num_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking','price']

df_train[num_vars] = scaler.fit_transform(df_train[num_vars])


# In[29]:


df_train.head()


# In[30]:


df_train.describe()


# In[31]:


# Let's check the correlation coefficients to see which variables are highly correlated

plt.figure(figsize = (16, 10))
sns.heatmap(df_train.corr(), annot = True, cmap="YlGnBu")
plt.show()


# As you might have noticed, `area` seems to the correlated to `price` the most. Let's see a pairplot for `area` vs `price`.

# ### Dividing into X and Y sets for the model building

# In[32]:


y_train = df_train.pop('price')
X_train = df_train


# ## Model Building

# This time, we will be using the **LinearRegression function from SciKit Learn** for its compatibility with RFE (which is a utility from sklearn)

# ### RFE

# Recursive feature elimination

# In[33]:


# Importing RFE and LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


# In[34]:


# Running RFE with the output number of the variable equal to 10
lm = LinearRegression()
lm.fit(X_train, y_train)


# In[35]:


rfe = RFE(estimator=lm, n_features_to_select=6)             # running RFE
rfe = rfe.fit(X_train, y_train)


# In[36]:


list(zip(X_train.columns,rfe.support_,rfe.ranking_))


# In[37]:


col = X_train.columns[rfe.support_]
col


# In[38]:


X_train.columns[~rfe.support_]


# ### Building model using statsmodel, for the detailed statistics

# In[39]:


# Creating X_test dataframe with RFE selected variables
X_train_rfe = X_train[col]


# In[40]:


# Adding a constant variable 
import statsmodels.api as sm  
X_train_rfe = sm.add_constant(X_train_rfe)


# In[41]:


lm = sm.OLS(y_train,X_train_rfe).fit()   # Running the linear model


# In[42]:


#Let's see the summary of our linear model
print(lm.summary())


# In[43]:


# Calculate the VIFs for the model
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[44]:


vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# ## Residual Analysis of the train data

# So, now to check if the error terms are also normally distributed (which is infact, one of the major assumptions of linear regression), let us plot the histogram of the error terms and see what it looks like.

# In[45]:


y_train_price = lm.predict(X_train_rfe)


# In[46]:


res = (y_train_price - y_train)


# In[47]:


# Importing the required libraries for plots.
import matplotlib.pyplot as plt
import seaborn as sns


# In[48]:


# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((y_train - y_train_price), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)                         # X-label


# In[49]:


plt.scatter(y_train,res)
plt.show()


# In[50]:


# There may be some relation in the error terms.


# ## Model Evaluation

# #### Applying the scaling on the test sets

# In[51]:


num_vars = ['area','stories', 'bathrooms', 'airconditioning', 'prefarea','parking','price']


# In[52]:


df_test[num_vars] = scaler.fit_transform(df_test[num_vars])


# #### Dividing into X_test and y_test

# In[53]:


y_test = df_test.pop('price')
X_test = df_test


# In[54]:


# Adding constant variable to test dataframe
X_test = sm.add_constant(X_test)


# In[55]:


# Now let's use our model to make predictions.


# In[56]:


# Creating X_test_new dataframe by dropping variables from X_test
X_test_rfe = X_test[X_train_rfe.columns]


# In[57]:


# Making predictions
y_pred = lm.predict(X_test_rfe)


# In[58]:


from sklearn.metrics import r2_score 
r2_score(y_test, y_pred)


# In[59]:


# Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test,y_pred)
fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('y_pred', fontsize=16)                          # Y-label


# 
# We can see that the equation of our best fitted line is:
# 
# $ price = 0.35  \times  area + 0.20  \times  bathrooms + 0.19 \times stories+ 0.10 \times airconditioning + 0.10 \times parking + 0.11 \times prefarea $
#
