
# coding: utf-8

# In[ ]:

#Adding multiple columns from DF2 to DF1 usinf CONCAT and ASSIGN functions

DF_Arsenal_away_attack = pd.concat((DF_Arsenal_away[DF_cols_FT_away_attack],DF_Arsenal_away[DF_cols_DIMS]),axis=1)
#DF_Arsenal_away_attack = DF_Arsenal_away_attack.assign(awayTeam=DF_Arsenal_away['awayTeam'],homeTeam = DF_Arsenal_away['homeTeam'],Season=DF_Arsenal_away['Season'])


# In[5]:

#removing unwanted column and converting column names to a list 
DF = DF.drop('Unnamed: 0',axis=1)
DF_cols = DF.columns.tolist()


# In[4]:

#counting number of column data types  
DF.dtypes.value_counts()


# In[6]:

#segregating columns based on text only types, half time and full time based
DF_cols_text = DF.select_dtypes(include=['object']).columns.tolist()
DF_cols_HT = [col for col in DF.columns if col.endswith('HT')]
DF_cols_FT = [col for col in DF.columns if col.endswith('FT')]


# In[7]:

#validating whether all columns except ID column are processed
list(set(DF_cols) - set(DF_cols_HT + DF_cols_FT + DF_cols_text))


# In[9]:

#creating corelation matrix to draw insights from features  
#DF[DF_cols_FT_away].corr()
df_away_attach = DF[DF_cols_FT_away_attack].corr()


# In[10]:

plt.matshow(df_away_attach)


# In[ ]:

#from above awayPassesKeyFT and awayPassSuccessFT are well corelated lets dive deep to understand these features 
#and how they are diestributed
DF.plot(y='awayPassesKeyFT',x='awayPassSuccessFT',kind='Scatter')
DF.plot(y='awayPassesKeyFT',x='awayShotsOnTargetFT',kind='Scatter')
DF.plot(x='awayPassesKeyFT',y='awayGoalFT',kind='Scatter')


# In[ ]:

#plot bar chart(using PIVOT table) for each EPL teams away passes vs shots on target 
DF.loc[DF.division == 'EPL'].pivot_table(index=['awayTeam'],                                         values=['awayPassesKeyFT','awayShotsOnTargetFT','awayGoalFT','homeGoalFT'],                                         aggfunc=np.mean).sort_values(by ='awayPassesKeyFT', ascending = False).                                         plot(kind='bar',rot=90,figsize=(10,8))


# In[ ]:

c = DF[DF_cols_FT_away].corr().abs()
c.unstack().sort_values(kind="quicksort",ascending=False)


# In[ ]:

DF[DF_cols_FT_away].corr()['awayTackleSuccessFT']


# In[ ]:

DF.plot('awayTackleSuccessFT','awayDribbledPastFT',kind='Scatter')


# In[ ]:

#displaying column names after grouping
parts_by_year = sets.groupby(['year'],as_index = False).num_parts.mean()


# In[ ]:

#List Highest Correlation Pairs from a Large Correlation Matrix in Pandas?

df_data = DF[DF_cols_FT_away_attack]
def get_dup_cols(df_data):
    pairs_to_drop = set()
    df_data_cols = df_data.columns
    for i in range(0,df_data.shape[1]):
        for j in range(0,i+1):
            pairs_to_drop.add((df_data_cols[i],df_data_cols[j]))
            df_val.drop
    return pairs_to_drop
df_val = df_data.corr().abs().unstack()
labels_todrop = get_dup_cols(df_data)
df_val = df_val.drop(labels=labels_todrop).sort_values(ascending=False)

