# Setup
#!pip3 install lightgbm

# -----
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
##
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV  
from sklearn.metrics import roc_curve, roc_auc_score,auc
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
###
import pickle
import cdsw


train = pd.read_csv('Santander_Customer_Transc/data/train.csv')
test = pd.read_csv('Santander_Customer_Transc/data/test.csv')

#display few records from train and test dataset
train.head()
test.head()

#drop columns that are not required
train = train.drop(['ID_code'],axis = 1)
test = test.drop(['ID_code'],axis = 1)

#Validate by running below statements
train.head()
train.shape
test.head()
test.shape

#Target distribution suggests imbalance in label
train['target'].value_counts().plot.bar()

#Feature Engineering
#Let us see different column datatypes in the dataset
#train.dtypes.value_counts()
train.dtypes.unique()

# all the features are of numerical datatype, no categorical values, except the index column. which is fine
train.select_dtypes(include= ['object']).head()

# let us check for missing values (NULLS) in our dataset
# From below the dataset look complete
train.isnull().sum().sort_values(ascending = False).value_counts()

'''
CORR() WILL BE EFFECTIVE FOR REGRESSION PROBLEMS ONLY (clarification and research required)
# To keep things simple, let us generate correlation matrix and ignore features that are weekly correlated with Target
train_corr = train.corr()
train_corr = train_corr['target'].abs().sort_values(ascending=False)

#histogram of correlation values
pd.cut(train_corr,bins=[0, 0.01,0.02,0.03,0.04,0.05,0.06,1],include_lowest=True).value_counts().sort_values(ascending=False).plot.bar()

'''


#Scatter plot of two features with max and min correlation with target.
#Shows the labels are distributed equaly and no clear decision boundaries
#KNN may not be beter model as data points are equally distributed
# with a huge class imbalance which might result in BIAS

plt.scatter(train.var_81, train.var_30, alpha=0.2,c=train.target, cmap='viridis')
plt.scatter(train.var_81, train.var_139, alpha=0.2,c=train.target, cmap='viridis')
plt.scatter(train.var_81, train.var_12, alpha=0.2,c=train.target, cmap='viridis')



#Split Features and Target label from train dataset
X = train.drop(['target'],axis=1)
y = train['target']
#Standardising/Normalising faetures

seed = 123

scaler = StandardScaler()

X = scaler.fit_transform(X)
X_val = scaler.transform(test)

#Standardized data split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,stratify=y, random_state=seed)


############
##LOG REG
############

#Use GridSearchCV on Logistic Regression model to identify optimal C vaule 

lgr  = LogisticRegression(random_state = seed,class_weight='balanced',multi_class='ovr')
grid = {'C':np.logspace(-3,3,7)}
##For integer CV inputs with binary labels STRTIFY is applied by default
lgr_cv = GridSearchCV(lgr,grid,cv=5, scoring='roc_auc',n_jobs=-1)
lgr_cv.fit(X,y)

lgr_cv.best_params_
#{'C': 0.001}

#Let's use optimal C from GridSearch to fit and predict a logreg model
#using Standardized data
log_reg_mod1 = LogisticRegression(C= 0.001,random_state = seed,class_weight='balanced',multi_class='ovr')
log_reg_mod1.fit(X_train,y_train)
y_pred_proba = log_reg_mod1.predict_proba(X_test)[:,1]
roc_auc_score(y_test,y_pred_proba)

#model score is 0.8574026468862973 using standardized data

'''
#Lets look into the co-eff importance in this model
mod1_feature_importance = pd.Series(log_reg_mod1.coef_[0],index=train.columns[1:])

mod1_feature_importance = mod1_feature_importance.abs().sort_values(ascending=False)

'''
############
##LOG REG AdaBoosting
############
ad_lr_mod2 = AdaBoostClassifier(base_estimator = log_reg_mod1,learning_rate = 0.1,\
                                n_estimators=200,random_state=seed)
ad_lr_mod2.fit(X_train,y_train)
y2_pred_proba = ad_lr_mod2.predict_proba(X_test)[:,1]
roc_auc_score(y_test,y2_pred_proba)
#0.85810337889358568


############
##Decision Tree Classifier
############

#using GridSearchCV to find optimal parameters
'''
#Parameters passed as dict of list values
dt_parms = {'criterion': ['gini'],
 'max_depth': [4,6,7,8,10],
 'max_features': ['log2'],
 'max_leaf_nodes': [10,15,30,45],
 'min_samples_leaf': [.12,.15],
 'class_weight': ['balanced'],
 'random_state': [seed]}

dt_mod3 = DecisionTreeClassifier()
dt_cv = GridSearchCV(dt_mod3,param_grid = dt_parms,cv=5)
dt_cv.fit(X_train,y_train)
dt_cv.best_params_
'''
# Assign parameter values identified from GRIDSEARCH above
dt_parms = {'class_weight': 'balanced',
 'criterion': 'gini',
 'max_depth': 4,
 'max_features': 'log2',
 'max_leaf_nodes': 10,
 'min_samples_leaf': 0.12,
 'random_state': seed}

#0.55676613973399325

dt1_parms = {'class_weight': 'balanced',
 'criterion': 'gini',
 'max_depth': 10,
 'max_features': 'log2',
 'max_leaf_nodes': 10,
 'min_samples_leaf': 300,
 'random_state': seed}
#0.6053783056226183

dt_mod3 = DecisionTreeClassifier(**dt_parms)
dt_mod3.fit(X_train,y_train)
y3_pred_proba = dt_mod3.predict_proba(X_test)[:,1]
roc_auc_score(y_test,y3_pred_proba)

############
##Random Forests
############

rf_params = {'bootstrap': True,
 'class_weight': 'balanced_subsample',
 'criterion': 'gini',
 'max_depth': 11,
 'max_features': 'log2',
 'max_leaf_nodes': 800,
 'min_impurity_decrease': 0.0,
 'min_impurity_split': None,
 'min_samples_leaf': 300,
 'min_samples_split': 2,
 'min_weight_fraction_leaf': 0.0,
 'n_estimators': 100,
 'n_jobs': -1,
 'oob_score': True,
 'random_state': seed}
#rf_cv = GridSearchCV(rf,param_grid=rf_params, cv= 3)
#rf_cv.fit(X,y)
#rf_cv.best_params_


rf1 = RandomForestClassifier(**rf_params)
rf1.fit(X_train,y_train)
y_pred_proba = rf1.predict_proba(X_test)[:,1]

roc_auc_score(y_test,y_pred_proba)
#0.8352328958021874


#So far we have 
#model1 -ADABOOST -LOGREG with 0.85810337889358568 and
#model2 - RF with 0.8352328958021874 let us create
#model3 - Gradient boosting model using DT
#model4 - Adaboost model using SVM
#Finnally we will create a Ensemble voting classifier 
#using all the models

############
##Ensemble Voting Classifier
############

clfs =[('AdaBoost_LOGREG',ad_lr_mod2),('RandomForest',rf1)]
vc_final = VotingClassifier(estimators=clfs,voting='soft',n_jobs=-1,)
vc_final.fit(X_train,y_train)
y_pred_proba_final = vc_final.predict_proba(X_test)[:,1]
roc_auc_score(y_test,y_pred_proba_final)

import pickle
import cdsw
filename = 'Santander_Customer_Transc/data/vc_final_v1.pkl'
pickle.dump(vc_final, open(filename, 'wb'))
cdsw.track_file(filename)

############
##Kaggle submission
############

y_val_proba = vc_final.predict_proba(X_val)[:,1]

y_final = [1 if val >= 0.5 else 0 for val in y_val_proba]

test1 = pd.read_csv('Santander_Customer_Transc/data/test.csv')
submission_df = test1[['ID_code']]
submission_df["target"] = y_final
submission_df.to_csv('Santander_Customer_Transc/data/submission.csv', index = False)


'''
model_final = pickle.load(open('Santander_Customer_Transc/data/vc_final_v1.pkl', 'rb'))
y_val_proba = model_final.predict_proba(X_val)[:,1]
y_final = [1 if val >= 0.5 else 0 for val in y_val_proba]
test1 = pd.read_csv('Santander_Customer_Transc/data/test.csv')
submission_df = test1[['ID_code']]
submission_df["target"] = y_final
submission_df.to_csv('Santander_Customer_Transc/data/submission.csv', index = False)

'''


'''
mod2_feature_importance = pd.Series(rf.feature_importances_,index=train.columns[1:])
mod2_feature_importance = mod2_feature_importance.abs().sort_values(ascending=False)
# Output
import pickle
import cdsw
filename = 'Santander_Customer_Transc/data/RF_CV.pkl'
pickle.dump(rf, open(filename, 'wb'))
cdsw.track_file(filename)


'''


'''
Example 1
# Run GridSearchCV to identify best parms of logreg classifier. 
#this will create 7X2X10=140 instance of classifier 
lgr  = LogisticRegression(random_state = 0,class_weight='balanced')
grid = {'C':np.logspace(-3,3,7),'penalty':['l1','l2']}
lgr_cv = GridSearchCV(lgr,grid,cv=10)
lgr_cv.fit(X,y)

lgr_cv.best_params_
#{'C': 0.10000000000000001, 'penalty': 'l1'}

#USe this best parm value to fit and predict split data
# reslted in 0.86 auc score
'''

'''
Example 2
lgr  = LogisticRegressionCV(cv =5, random_state = 0, Cs=[0.01,.1,0.001],class_weight='balanced')
lgr.fit(X_train,y_train)
y_pred_prob = lgr.predict_proba(X_test)[:,1]
roc_auc_score(y_test,y_pred_prob).mean()
#0.86

'''


