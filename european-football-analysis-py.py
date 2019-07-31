
# coding: utf-8

# In[1]:

# import required Libraries
import numpy as np
import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt
#sns.set(context="paper", font="monospace")
get_ipython().magic('matplotlib inline')


# In[2]:

#Data load
DF = pd.read_csv('http://localhost:8888/files/OneNote%20Notebooks/Personal/Europe%20Football_EDA/FootballEurope.csv')


# In[3]:

#removing unwanted column and converting column names to a list 
DF = DF.drop(['Unnamed: 0','id'],axis=1)
DF['date'] = pd.to_datetime(DF['date'])
DF_cols = DF.columns.tolist()
#counting number of column data types  
DF.dtypes.value_counts()


# In[4]:

#Set date column as index and sort by index to group by EPL seasons
DF = DF.set_index('date').sort_index(axis=0)


# In[5]:

#Populate Season column with appropriate values
DF['Season'] = np.where(((DF.index > '2012-07-01') & (DF.index < '2013-06-30')), '1213',
                        np.where(((DF.index > '2013-07-01') & (DF.index < '2014-06-30')),'1314',
                                 np.where(((DF.index > '2014-07-01') & (DF.index < '2015-06-30')),'1415',
                                          np.where(((DF.index > '2015-07-01') & (DF.index < '2016-06-30')),'1516',
                                                   np.where(((DF.index > '2016-07-01') & (DF.index < '2017-06-30')),'1617','NAN')))))               


# In[6]:

#Create awayResult and homeResult columns and populate values
DF['awayResult'] = np.where((DF.awayGoalFT > DF.homeGoalFT),'W',
                           np.where((DF.awayGoalFT < DF.homeGoalFT),'L','D'))

DF['homeResult'] = np.where((DF.homeGoalFT > DF.awayGoalFT),'W',
                           np.where((DF.homeGoalFT < DF.awayGoalFT),'L','D'))


# In[7]:

#segregating columns based on text only types, half time and full time based
DF_cols_text = DF.select_dtypes(include=['object']).columns.tolist()
DF_cols_HT = [col for col in DF.columns if col.endswith('HT')]
DF_cols_FT = [col for col in DF.columns if col.endswith('FT')]

#segregating away team and home team features for numerical columns
DF_cols_FT_away = [col for col in DF_cols_FT if col.startswith('away')]
DF_cols_FT_home = [col for col in DF_cols_FT if col.startswith('home')]

#segregating away columns by position
DF_cols_FT_away_attack = ['awayPassSuccessFT','awayDribbleSuccessFT','awayShotsOnTargetFT','awayPassesKeyFT',                          'awayDribbledPastFT','awayDribblesAttemptedFT','awayPossessionFT','awayShotsTotalFT',                         'awayGoalFT','homeGoalFT']
DF_cols_FT_away_defence = ['awayDispossessedFT','awayShotsBlockedFT','awayDribblesWonFT','awayInterceptionsFT',                           'awayTackleSuccessFT','awayTacklesTotalFT','awayGoalFT','homeGoalFT']
DF_cols_FT_away_neutral = ['awayGoalFT','homeGoalFT','awayOffsidesCaughtFT','awayFoulsCommitedFT','awayCornersTotalFT',                          'awayAerialsTotalFT']
DF_cols_DIMS = ['awayTeam','homeTeam','awayResult','homeResult','Season']

#segregating home columns by position
DF_cols_FT_home_attack = ['homePassSuccessFT','homeDribbleSuccessFT','homeShotsOnTargetFT','homePassesKeyFT',                          'homeDribbledPastFT','homeDribblesAttemptedFT','homePossessionFT','homeShotsTotalFT',                         'homeGoalFT','awayGoalFT']


# In[8]:

#funtion to create correlation matix of individual EPL team based on features supplied
def corr_matrix(DF,corr_cols):
    DF_Team_attack = DF[corr_cols]
    DF_Team_attack_Matrix = DF_Team_attack.corr().abs().unstack()
    labels_todrop =  get_dup_cols(DF_Team_attack)
    DF_Team_attack_Matrix = DF_Team_attack_Matrix.drop(labels=labels_todrop).sort_values(ascending=False)
    DF_Team_attack_Matrix = DF_Team_attack_Matrix.reset_index()
    return DF_Team_attack_Matrix   


#Create a function to parse corelation matrix and keep only top absolute unique values
#remove diagonals and lower triangle values from corelation matrix
#NOTE: use only features with numerical values
#df_data = DF[DF_cols_HT]
def get_dup_cols(df_data):
    pairs_to_drop = set()
    df_data_cols = df_data.columns
    for i in range(0,df_data.shape[1]):
        for j in range(0,i+1):
            pairs_to_drop.add((df_data_cols[i],df_data_cols[j]))
    return pairs_to_drop


# In[9]:

#Function to create Dataframe of individual EPL team with "away" and "home" performance details
def EPL_individual_team_DF(DF,team_name):
    DF_team = DF.loc[(DF.division =='EPL') & ((DF.awayTeam == team_name) | (DF.homeTeam == team_name))]
    return DF_team

#Function to create Dataframe of individual EPL team with "away" performance details
def EPL_individual_team_away_DF(DF,team_name):
    DF_team = DF.loc[(DF.division =='EPL') & (DF.awayTeam == team_name)]
    return DF_team

#Function to create Dataframe of individual EPL team with "home" performance details
def EPL_individual_team_home_DF(DF,team_name):
    DF_team = DF.loc[(DF.division =='EPL') & (DF.homeTeam == team_name)]
    return DF_team


# In[10]:

#Get Man City away performance data
DF_ManCity_away = EPL_individual_team_away_DF(DF,'Man City')
#DF_ManCity.head()


# In[11]:

#Get Arsenal away performance data 
DF_Arsenal_away = EPL_individual_team_away_DF(DF,'Arsenal')
#Create Arsenal away attack DF with supporting columns
DF_Arsenal_away_attack = pd.concat((DF_Arsenal_away[DF_cols_FT_away_attack],DF_Arsenal_away[DF_cols_DIMS]),axis=1)
#DF_Arsenal_away_attack = DF_Arsenal_away_attack.assign(awayTeam=DF_Arsenal_away['awayTeam'],homeTeam = DF_Arsenal_away['homeTeam'],Season=DF_Arsenal_away['Season'])


# In[12]:

#Get Arsenal home performance data 
DF_Arsenal_home = EPL_individual_team_home_DF(DF,'Arsenal')
#Create Arsenal home attack DF with supporting columns
DF_Arsenal_home_attack = pd.concat((DF_Arsenal_home[DF_cols_FT_home_attack],DF_Arsenal_home[DF_cols_DIMS]),axis=1)


# In[13]:

#Arsenal data with points for time series analysis
DF_Arsenal = EPL_individual_team_DF(DF,'Arsenal')
DF_Arsenal_1213 = DF_Arsenal.loc[DF_Arsenal.Season=='1213']


# In[30]:

PTS_1213 = 0


# In[31]:


for idx, row in DF_Arsenal.iterrows():
    if (((DF_Arsenal.loc[idx,'homeTeam'] == 'Arsenal') & (DF_Arsenal.loc[idx,'homeResult'] == 'W')) |           ((DF_Arsenal.loc[idx,'awayTeam'] == 'Arsenal') & (DF_Arsenal.loc[idx,'awayResult'] == 'W'))):
        DF_Arsenal.loc[idx,'PTS'] = PTS_1213 + 3
    elif (((DF_Arsenal.loc[idx,'homeTeam'] == 'Arsenal') & (DF_Arsenal.loc[idx,'homeResult'] == 'D')) |           ((DF_Arsenal.loc[idx,'awayTeam'] == 'Arsenal') & (DF_Arsenal.loc[idx,'awayResult'] == 'D'))):
        DF_Arsenal.loc[idx,'PTS'] = PTS_1213 + 1
    else:
        DF_Arsenal.loc[idx,'PTS'] = PTS_1213
DF_Arsenal.PTS
    
    
    
    


# In[14]:

DF_Arsenal_1213


# In[15]:

#PLOT1
#Arsenal away performance plot with attacking features
DF_Arsenal_away_attack.pivot_table(index=['homeTeam'],                                         values=['awayPossessionFT','awayShotsOnTargetFT','awayGoalFT','homeGoalFT'],                                         aggfunc=np.mean).sort_values(by ='awayPossessionFT', ascending = False).                                         plot(kind='bar',rot=90,figsize=(10,8))


# In[16]:

#PLOT2
#Arsenal away performance plot on losing games with attacking features
DF_Arsenal_away_attack.loc[DF_Arsenal_away_attack.awayResult == 'L'].                                         pivot_table(index=['homeTeam'],                                         values=['awayPossessionFT','awayShotsOnTargetFT','awayGoalFT','homeGoalFT'],                                         aggfunc=np.mean).sort_values(by ='awayPossessionFT', ascending = False).                                         plot(kind='bar',rot=90,figsize=(10,8))


# In[17]:

#Arsenal home performance on losing games with attacking features
DF_Arsenal_home_attack.loc[(DF_Arsenal_home_attack.homeResult == 'L')].                                         pivot_table(index=['awayTeam'],                                         values=['homePossessionFT','homeShotsOnTargetFT','homeGoalFT','awayGoalFT'],                                         aggfunc=np.mean).sort_values(by ='homePossessionFT', ascending = False).                                         plot(kind='bar',rot=90,figsize=(10,8))


# In[18]:

#correlation matix of Arsenal attacking features at away
DF_Arsenal_away_attack_Matrix = corr_matrix(DF_Arsenal_away,DF_cols_FT_away_attack)
DF_Arsenal_away_attack_Matrix = DF_Arsenal_away_attack_Matrix.sort_values(['level_0','level_1'])


# In[19]:

#correlation matix of Man City attacking at away
DF_ManCity_away_attack_Matrix = corr_matrix(DF_ManCity_away,DF_cols_FT_away_attack)
#DF_ManCity_attack_Matrix = DF_ManCity_attack_Matrix.rename(index=str, columns = {'level_0':'ManCity_0','level_1':'ManCity_1',0:'ManCity'})
DF_ManCity_away_attack_Matrix = DF_ManCity_away_attack_Matrix.sort_values(['level_0','level_1'])


# In[20]:

#DF_Arsenal_attack_Matrix['ManCity_0'] = DF_ManCity_attack_Matrix['level_0'].values
#DF_Arsenal_attack_Matrix['ManCity_1'] = DF_ManCity_attack_Matrix['level_1'].values
DF_Arsenal_away_attack_Matrix['ManCity'] = DF_ManCity_away_attack_Matrix[0].values


# In[21]:

DF_Arsenal_away_attack_Matrix = DF_Arsenal_away_attack_Matrix.sort_values(0,ascending=False).rename(index=str, columns = {0:'Arsenal'})


# In[ ]:



