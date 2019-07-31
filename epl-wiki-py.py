import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
%matplotlib inline
url = {}
url['1617'] = 'https://en.wikipedia.org/wiki/2016%E2%80%9317_Premier_League'
url['1516'] = 'https://en.wikipedia.org/wiki/2015%E2%80%9316_Premier_League'
url['1415'] = 'https://en.wikipedia.org/wiki/2014%E2%80%9315_Premier_League'
url['1314'] = 'https://en.wikipedia.org/wiki/2013%E2%80%9314_Premier_League'

#Function to extract PL data by year
def create_DF(key,value):
  
  #Get contents from WIKI using requests package
  web_res = requests.get(value)
  html_text= web_res.text
  soup = BeautifulSoup(html_text) 
  
  #manually identified table position in website and extract
  table_data = soup.find_all('table')[4]
  #Initialize and extract column names using header tag 
  col = []
  err_file = []         # to save error rows
  col = [i.get_text() for i in table_data.find('tr').find_all('th')]
  col = [re.sub('\n','',i) for i in col]
      
  # extracting only required columns
  col = col[1:10] 
  
  #formatting columns
  col[0] = 'Team'    
  col.append('Season')
  
  #Create PL_DF DataFrame to hold all years pl data
  PL_DF = pd.DataFrame(columns = col,index=None)
  
  #process one row at a time
  for cur_row in table_data.find_all('tr'):
    row = []
   
    #extract all columns of a row
    for cur_col in cur_row.find_all('td'):
      row.append(cur_col.text)
    
    # Process only required columns of a row
    row = row[:9]
    row.append(key)        #adding season details
    if len(row) == 10:
      PL_DF = PL_DF.append(pd.Series(row, index=col), ignore_index=True)      
    else:
      err_file.append(row)
    PL_DF['Pos'] = PL_DF.index+1
  return PL_DF      #returns Dataframe with passed years PL data from wiki

# get PL table data by year and append it to one DF
PL_YEARS = pd.DataFrame()
for key,value in url.items():
  PL_YEARS = PL_YEARS.append(create_DF(key,value),ignore_index=True)

# converting columns to integers and str
int_cols = ['Pld','W','D','L','GF','GA','Pts','Season','Pos']
str_cols = ['Team','GD']
PL_YEARS.loc[:,int_cols] = PL_YEARS.loc[:,int_cols].apply(pd.to_numeric)
PL_YEARS.loc[:,str_cols] = PL_YEARS.loc[:,str_cols].astype(str)
#PL_YEARS.set_index(['Season','Pos'])
#PL_YEARS.groupby('Team').sum().sort_values(by = 'Pts',ascending=False)

#PLot to analyze points over season used Position instead of Teams
#as team names varies by season

#PL_YEARS.pivot(index='Pos',columns='Season',values='Pts'). \
#    plot(kind='bar',xticks = np.arange(20),figsize= (10,8))
'''
#Graph1
PL_YEARS.groupby(['Season',pd.cut(PL_YEARS.Pos,5)]).mean()['Pts'].plot('bar')
'''
#Graph2
x_ticks = np.array(PL_YEARS.index.values)
y_ticks = np.linspace(0,100,num=10)
pl_fig = PL_YEARS.plot(x='Team',y=['Pts','W','D','L'],kind='line', \
              xticks=x_ticks,yticks=y_ticks,rot = 90,figsize=(14,10))
fig = pl_fig.get_figure()
fig.savefig(r"H:\Python exercises\plot.pdf") 


