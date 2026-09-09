# Housing-Analysis
Housing Price Prediction — Multiple Linear Regression
A machine learning project that predicts house prices in the Delhi region using Multiple Linear Regression, based on property features such as area, number of bedrooms, bathrooms, stories, and amenities.

Problem Statement
A real estate company wants to understand which factors most strongly influence property prices and use that understanding to optimize pricing decisions. This project aims to:

Identify the variables that significantly affect house prices (e.g. area, number of rooms, bathrooms)
Build a linear model that quantitatively relates these variables to price
Evaluate how accurately the model predicts prices on unseen data
Dataset
545 property records with the following features:

Feature	Description
price	Target variable — price of the house
area	Area of the property (sq. ft.)
bedrooms, bathrooms, stories	Numerical count features
mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea	Binary (yes/no) features
parking	Number of parking spaces
furnishingstatus	Categorical — furnished / semi-furnished / unfurnished
Approach
Data Exploration & Cleaning — inspected data types, checked for missing values, visualized distributions and outliers.
Feature Engineering — converted binary yes/no columns to 1/0; created dummy variables for furnishingstatus.
Train-Test Split & Scaling — split the data and applied min-max scaling to numerical features.
Feature Selection — used Recursive Feature Elimination (RFE) along with manual p-value and VIF (Variance Inflation Factor) checks to select significant, non-multicollinear predictors.
Model Building — fit an Ordinary Least Squares (OLS) regression model using statsmodels.
Model Evaluation — analyzed residual distribution and homoscedasticity, and evaluated performance using R² on the test set.
Tech Stack
Python
pandas, NumPy
scikit-learn
statsmodels
Matplotlib, Seaborn
Project Structure
.
├── housing-price-prediction-linear-regression.ipynb   # Full notebook with EDA, modeling, and evaluation
├── housing_regression.py                               # Plain Python script version
├── Housing.csv                                          # Dataset
└── README.md
Code
# Imports
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
housing = pd.DataFrame(pd.read_csv("Housing.csv"))
housing.head()
housing.shape
housing.info()
housing.describe()

# Check for missing values
housing.isnull().sum() * 100 / housing.shape[0]   # No nulls

# Outlier Analysis
fig, axs = plt.subplots(2, 3, figsize=(10, 5))
sns.boxplot(housing['price'], ax=axs[0, 0])
sns.boxplot(housing['area'], ax=axs[0, 1])
sns.boxplot(housing['bedrooms'], ax=axs[0, 2])
sns.boxplot(housing['bathrooms'], ax=axs[1, 0])
sns.boxplot(housing['stories'], ax=axs[1, 1])
sns.boxplot(housing['parking'], ax=axs[1, 2])
plt.tight_layout()

# Outlier treatment for price
Q1 = housing.price.quantile(0.25)
Q3 = housing.price.quantile(0.75)
IQR = Q3 - Q1
housing = housing[(housing.price >= Q1 - 1.5*IQR) & (housing.price <= Q3 + 1.5*IQR)]

# Outlier treatment for area
Q1 = housing.area.quantile(0.25)
Q3 = housing.area.quantile(0.75)
IQR = Q3 - Q1
housing = housing[(housing.area >= Q1 - 1.5*IQR) & (housing.area <= Q3 + 1.5*IQR)]

# EDA
sns.pairplot(housing)
plt.show()

plt.figure(figsize=(20, 12))
for i, col in enumerate(['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'furnishingstatus'], 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x=col, y='price', data=housing)
plt.show()

# Convert binary yes/no columns to 1/0
varlist = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
def binary_map(x):
    return x.map({'yes': 1, "no": 0})
housing[varlist] = housing[varlist].apply(binary_map)

# Create dummy variables for furnishingstatus
status = pd.get_dummies(housing['furnishingstatus'], drop_first=True)
housing = pd.concat([housing, status], axis=1)
housing.drop(['furnishingstatus'], axis=1, inplace=True)

# Train-test split
from sklearn.model_selection import train_test_split
np.random.seed(0)
df_train, df_test = train_test_split(housing, train_size=0.7, test_size=0.3, random_state=100)

# Rescaling features (Min-Max scaling)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
num_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'price']
df_train[num_vars] = scaler.fit_transform(df_train[num_vars])

# Correlation heatmap
plt.figure(figsize=(16, 10))
sns.heatmap(df_train.corr(), annot=True, cmap="YlGnBu")
plt.show()

y_train = df_train.pop('price')
X_train = df_train

# Recursive Feature Elimination (RFE)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train, y_train)

rfe = RFE(estimator=lm, n_features_to_select=6)
rfe = rfe.fit(X_train, y_train)

col = X_train.columns[rfe.support_]

# Build OLS model with statsmodels
import statsmodels.api as sm
X_train_rfe = X_train[col]
X_train_rfe = sm.add_constant(X_train_rfe)

lm = sm.OLS(y_train, X_train_rfe).fit()
print(lm.summary())

# Check VIF for multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif['Features'] = X_train_rfe.columns
vif['VIF'] = [variance_inflation_factor(X_train_rfe.values, i) for i in range(X_train_rfe.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by="VIF", ascending=False)

# Residual analysis
y_train_price = lm.predict(X_train_rfe)
res = y_train_price - y_train

fig = plt.figure()
sns.distplot((y_train - y_train_price), bins=20)
fig.suptitle('Error Terms', fontsize=20)
plt.xlabel('Errors', fontsize=18)

plt.scatter(y_train, res)
plt.show()

# Apply scaling to test set
num_vars = ['area', 'stories', 'bathrooms', 'airconditioning', 'prefarea', 'parking', 'price']
df_test[num_vars] = scaler.fit_transform(df_test[num_vars])

y_test = df_test.pop('price')
X_test = df_test
X_test = sm.add_constant(X_test)

X_test_rfe = X_test[X_train_rfe.columns]

# Predict on test set
y_pred = lm.predict(X_test_rfe)

# Evaluate model
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

# Plot actual vs predicted
fig = plt.figure()
plt.scatter(y_test, y_pred)
fig.suptitle('y_test vs y_pred', fontsize=20)
plt.xlabel('y_test', fontsize=18)
plt.ylabel('y_pred', fontsize=16)
plt.show()
Final fitted equation: price = 0.35 × area + 0.20 × bathrooms + 0.19 × stories + 0.10 × airconditioning + 0.10 × parking + 0.11 × prefarea

Code & Output
1. Load the Data
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

housing = pd.DataFrame(pd.read_csv("Housing.csv"))
housing.head()
      price  area  bedrooms  bathrooms  stories mainroad guestroom basement  \
0  13300000  7420         4          2        3      yes        no       no   
1  12250000  8960         4          4        4      yes        no       no   
2  12250000  9960         3          2        2      yes        no      yes   
3  12215000  7500         4          2        2      yes        no      yes   
4  11410000  7420         4          1        2      yes       yes      yes   

  hotwaterheating airconditioning  parking prefarea furnishingstatus  
0              no             yes        2      yes        furnished  
1              no             yes        3       no        furnished  
2              no              no        2      yes   semi-furnished  
3              no             yes        3      yes        furnished  
4              no             yes        2       no        furnished  
2. Inspect the Data
housing.shape
(545, 13)
housing.info()
<class 'pandas.DataFrame'>
RangeIndex: 545 entries, 0 to 544
Data columns (total 13 columns):
 #   Column            Non-Null Count  Dtype
---  ------            --------------  -----
 0   price             545 non-null    int64
 1   area              545 non-null    int64
 2   bedrooms          545 non-null    int64
 3   bathrooms         545 non-null    int64
 4   stories           545 non-null    int64
 5   mainroad          545 non-null    str  
 6   guestroom         545 non-null    str  
 7   basement          545 non-null    str  
 8   hotwaterheating   545 non-null    str  
 9   airconditioning   545 non-null    str  
 10  parking           545 non-null    int64
 11  prefarea          545 non-null    str  
 12  furnishingstatus  545 non-null    str  
dtypes: int64(6), str(7)
memory usage: 55.5 KB
3. Feature Selection with RFE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train, y_train)

rfe = RFE(estimator=lm, n_features_to_select=6)
rfe = rfe.fit(X_train, y_train)

list(zip(X_train.columns, rfe.support_, rfe.ranking_))
[('area', True, 1),
 ('bedrooms', False, 7),
 ('bathrooms', True, 1),
 ('stories', True, 1),
 ('mainroad', False, 5),
 ('guestroom', False, 6),
 ('basement', False, 4),
 ('hotwaterheating', False, 2),
 ('airconditioning', True, 1),
 ('parking', True, 1),
 ('prefarea', True, 1),
 ('semi-furnished', False, 8),
 ('unfurnished', False, 3)]
col = X_train.columns[rfe.support_]
col
Index(['area', 'bathrooms', 'stories', 'airconditioning', 'parking',
       'prefarea'], dtype='object')
4. Build the OLS Regression Model
import statsmodels.api as sm

X_train_rfe = sm.add_constant(X_train[col])
lm = sm.OLS(y_train, X_train_rfe).fit()
print(lm.summary())
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                  price   R-squared:                       0.611
Model:                            OLS   Adj. R-squared:                  0.605
Method:                 Least Squares   F-statistic:                     92.83
Prob (F-statistic):                                                  1.31e-69
No. Observations:                 361   AIC:                            -431.5
Df Residuals:                     354   BIC:                            -404.3
Df Model:                           6                                         
Covariance Type:            nonrobust                                         
===================================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
-----------------------------------------------------------------------------------
const               0.1097      0.015      7.442      0.000       0.081       0.139
area                0.3502      0.037      9.361      0.000       0.277       0.424
bathrooms           0.2012      0.033      6.134      0.000       0.137       0.266
stories             0.1884      0.026      7.219      0.000       0.137       0.240
airconditioning     0.0965      0.016      ...    0.000       ...         ...
5. Check Multicollinearity (VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif = vif.sort_values(by="VIF", ascending=False)
vif
          Features   VIF
0            const  4.51
1             area  1.24
4  airconditioning  1.20
3          stories  1.17
5          parking  1.14
2        bathrooms  1.12
6         prefarea  1.05
All VIF values are well below 5, confirming low multicollinearity among the selected predictors.

6. Evaluate on Test Data
from sklearn.metrics import r2_score

y_pred = lm.predict(X_test_rfe)
r2_score(y_test, y_pred)
0.579
The final model achieves an R² of ~0.61 on training data and ~0.58 on test data, using six significant, non-multicollinear predictors: area, bathrooms, stories, airconditioning, parking, and prefarea.

How to Run
pip install pandas numpy scikit-learn statsmodels matplotlib seaborn
python housing_regression.py
Or open housing-price-prediction-linear-regression.ipynb in Jupyter to run interactively.

Key Learnings
Practical, end-to-end experience building a regression model from raw data to validated predictions
Handling multicollinearity using VIF
Feature selection trade-offs with RFE
Interpreting model coefficients and residual diagnostics
##author Ahmad Muneeb
