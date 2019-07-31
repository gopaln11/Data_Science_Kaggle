
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from scipy.stats import skew


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
test_submit = test.loc[:,:]
train.head()
test.head()

train.columns
train.isnull().sum().sort_values(ascending = False)
train.corr()

train_corr = train.corr()
train_corr['SalePrice'].loc[(train_corr.SalePrice >= .5) | (train_corr.SalePrice <= -.5)].sort_values(ascending = False)

#Feature 1 OverallQual
train.OverallQual.value_counts()


ax = train.hist('SalePrice',by = 'OverallQual',figsize = (6,8), bins = 5)
ax[0][0].set_ylabel('count by SalePrice bins for each Qlf.')



#Feature 2 GrLivArea
train.GrLivArea.value_counts()
train['GrLivArea'].describe()
#binx = np.linspace(train.SalePrice.min(),train.SalePrice.max(),20)
#train.hist('SalePrice',by = 'GrLivArea',bin = binx)



bins = np.linspace(train.SalePrice.min(),train.SalePrice.max(),20)
ax = train.groupby(pd.cut(train.SalePrice,20)).count()['GrLivArea'].plot('bar',rot =60,figsize=(8,6))
ax.set_xlabel('SalePrice binned')
ax.set_ylabel('count of GrLivArea in that SalePrice range',rotation=90)


ax = train.groupby(pd.cut(train.SalePrice, bins =20)).mean()['GrLivArea'].plot('bar',rot =60,figsize=(8,6))
ax.set_xlabel('SalePrice binned')
ax.set_ylabel('mean sq living area in that SalePrice range',rotation=90)


# from below we understand last but two bins have no values
train.groupby(pd.cut(train.SalePrice, bins =20)).count()['GrLivArea']
#train['GrLivArea'].loc[(train.SalePrice > 646985) &  (train.SalePrice < 718995)]


#Feature 3 GarageCars
train.GarageCars.describe()

train.GarageCars.value_counts()

train.hist('SalePrice',by='GarageCars', figsize=(6,8))

#Feature 4 GarageArea
train.GarageArea.describe()


GarageArea_uniquevalues = len(train.GarageArea.unique())
print('Unique values in GarageArea feature is %d' % GarageArea_uniquevalues)


train.groupby(train.GarageCars).mean()['GarageArea'].plot('bar')

train.groupby(pd.cut(train.GarageArea,25)).mean()['GarageCars'].plot('bar')

train.groupby(pd.cut(train.GarageArea,25)).mean()['SalePrice'].plot('bar')
#from below records in last 3 bins needs more investigation


# In[22]:

train['grp'] = pd.cut(train.GarageArea,25)
train['grp'].head()
train.groupby(['grp','GarageCars']).GarageArea.size()
# FROM BELOW THERE ARE FEW INSTANCE of 4 cars with less GR space, OUTLIERS


# In[23]:

train.TotalBsmtSF.describe()


# In[24]:

ax = train.TotalBsmtSF.plot('box')
ax.set_ylim(700,1500)


# In[25]:

train.groupby(pd.cut(train.TotalBsmtSF,25)).SalePrice.count()


# In[26]:

train.groupby(pd.cut(train.TotalBsmtSF,25)).SalePrice.mean()


# In[27]:

train.groupby(pd.cut(train.TotalBsmtSF,25)).SalePrice.mean().plot('bar')


# In[28]:

train.groupby(pd.cut(train['1stFlrSF'],25)).SalePrice.mean().plot('bar')


# In[29]:

train.groupby(pd.cut(train['1stFlrSF'],20)).SalePrice.count()


# In[30]:

ax = train.groupby(train.FullBath).SalePrice.mean().plot('bar')
ax.set_title('Average SalePrice by FullBath')


# In[31]:

train.FullBath.value_counts()


# In[32]:

train.groupby(train.TotRmsAbvGrd).SalePrice.mean().plot('bar')


# In[33]:

train.groupby(train.TotRmsAbvGrd).SalePrice.count().plot('bar')


# In[34]:

train.groupby(['FullBath','TotRmsAbvGrd']).SalePrice.mean().plot('bar',figsize=(8,5))


# In[35]:

train.groupby(pd.cut(train.YearBuilt,25)).count()['SalePrice']


# In[36]:

fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(1,2,1)
bx = train.groupby(pd.cut(train.YearBuilt,25)).mean()['SalePrice'].plot('bar')
bx.set_title('SalePrice by year built')
ax = fig.add_subplot(1,2,2)
bx = train.groupby(pd.cut(train.YearRemodAdd,25)).mean()['SalePrice'].plot('bar')
bx.set_title('SalePrice by YearRemodAdd')
plt.show()


# In[70]:

train_target = train[['SalePrice']]
train = train[['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF',
                   'FullBath','TotRmsAbvGrd','YearBuilt','YearRemodAdd']]

test = test[['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF',
                   'FullBath','TotRmsAbvGrd','YearBuilt','YearRemodAdd']]


# In[71]:

train.isnull().sum()


# In[39]:

#From bwlow train dataset looks good and all set as input for model
train.head()


# In[72]:

test.isnull().sum()


# In[42]:

#Pull the record with NULL value to identify replacement value
test[test['GarageCars'].isnull()]


# In[43]:

train.pivot_table(index='OverallQual',values=['GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF',
                                              'FullBath','TotRmsAbvGrd','YearBuilt','YearRemodAdd'],
                  aggfunc=np.mean)

#using the information from pivot table, filling NULL with 2,600 and 1108.514107 for respective fields
test['GarageCars'] = test.GarageCars.fillna(2)
test['GarageArea'] = test.GarageArea.fillna(600)
test['TotalBsmtSF'] = test.TotalBsmtSF.fillna(1108.514107)
test.head()
test.info()

X_train,X_test,y_train,y_test = train_test_split(train,train_target,test_size=.3,random_state=42)
lr = linear_model.LinearRegression()
lr = lr.fit(X_train,y_train)

lr.score(X_test,y_test)

#Model Prediction of the test data from competition that is cleansed  
test_submit['SalePrice'] = lr.predict(test)

test.head()

test_submit[['Id','SalePrice']].to_csv('H:\Python exercises\Exercise2_AmesIowa_housing\Ames_housing_V1.csv',index=False)

#Model prediction of the test data from train test split
y_pred = lr.predict(X_test)
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(1,1,1)
ax.set_xscale('log')
ax.set_yscale('log')
plt.scatter(y_pred,y_test,c= ['b','r'])

Model_mse = np.mean((y_pred-y_test)**2)
Model_RMSE = np.sqrt(Model_mse)
print(Model_RMSE)

train_data_Skew = train.apply(lambda x: skew(x))
np.array(lr.coef_)

alpha_space = np.logspace(-4, 0, 50)
alpha_space.size


