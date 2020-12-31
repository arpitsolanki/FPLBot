#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/arpitsolanki/FPLBot/blob/main/Fantasy_League.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[65]:


import os
import pandas as pd
import requests
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xgboost as xgboost
import datetime

from sklearn.model_selection import train_test_split #Splitting data for model training
from sklearn.ensemble import RandomForestClassifier #RF
from sklearn.preprocessing import LabelEncoder#Label Encoder
from sklearn.preprocessing import OneHotEncoder#One Hot Encoder
from sklearn.metrics import confusion_matrix#Confusion Matrix
from sklearn.metrics import roc_curve#RoC Curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve#Metrics Functions


# In[66]:


#Define GW for predictions
GW=16

#Fantasy API Calls to pull historic player stats
url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
response = requests.get(url)
response = json.loads(response.content)

#Fixtures for upcoming game weeks
fix_url = 'https://fantasy.premierleague.com/api/fixtures?event='+str(GW)
fix_response = requests.get(fix_url)
fix_response = json.loads(fix_response.content)


# In[67]:


def get(url):
    response = requests.get(url)
    return json.loads(response.content)
                      


# In[68]:


players = response['elements']
teams = response['teams']
events = response['events']
#fixtures = fix_response['fixtures']

players_df = pd.DataFrame(players)
teams_df = pd.DataFrame(teams)
events_df = pd.DataFrame(events)
fixtures_df=pd.DataFrame(fix_response)
fixtures_df


# In[69]:


events_df


# In[70]:


#Dataset from Fivethirtyeight providing projected scores for the upcoming game week
spi_data = pd.read_csv('https://projects.fivethirtyeight.com/soccer-api/club/spi_matches_latest.csv')
spi_data=spi_data.loc[(spi_data['season']==2020) & (spi_data['league']=='Barclays Premier League')]
#spi_data

#Team mapping file
team_mapping_spi=pd.read_csv('https://raw.githubusercontent.com/arpitsolanki/FPLBot/main/team_mapping.csv')
#team_mapping_spi

#Add team_ids to the 538 dataset to make it compatible with FPL API datasets
spi_data_name=pd.merge(left=spi_data,right=team_mapping_spi,how='inner',left_on='team1',right_on='team')
spi_data_name=spi_data_name.rename(columns={"id": "team1_id"})

spi_data_name=pd.merge(left=spi_data_name,right=team_mapping_spi,how='inner',left_on='team2',right_on='team')
spi_data_name=spi_data_name.rename(columns={"id": "team2_id"})
#spi_data_name


# In[71]:


players_df


# In[72]:


players_df_filtered=players_df[['web_name','element_type','now_cost','selected_by_percent','points_per_game','team','total_points','minutes','goals_scored','assists','clean_sheets','goals_conceded','yellow_cards','red_cards','bonus','chance_of_playing_next_round']]
players_df_filtered=players_df_filtered.loc[players_df_filtered['minutes']>300]
players_df_filtered['now_cost_mil']=players_df_filtered['now_cost']/10
players_df_filtered['ppm']=players_df_filtered['total_points']/players_df_filtered['now_cost_mil']
players_df_filtered.sort_values(by='ppm',ascending=False,inplace=True)


# In[73]:


#players_df_filtered.sort_values(by='total_points',ascending=False)


# In[74]:


teams_df_filtered=teams_df[['id','name','short_name','played','points','position','win','draw','loss']]
teams_df_filtered_join=teams_df[['id','name']]


# In[75]:


fixtures_df_name=pd.merge(left=fixtures_df,right=teams_df_filtered_join,left_on='team_a',right_on='id',how='left')
fixtures_df_name=fixtures_df_name.rename(columns={"name": "away_team"})
fixtures_df_name=pd.merge(left=fixtures_df_name,right=teams_df_filtered_join,left_on='team_h',right_on='id',how='left')
fixtures_df_name=fixtures_df_name.rename(columns={"name": "home_team"})
fixtures_df_name


# In[76]:


df_l=[]
for i in range(GW):
  fix_url = 'https://fantasy.premierleague.com/api/fixtures?event='+str(i+1)
  fix_response = requests.get(fix_url)
  fix_response = json.loads(fix_response.content)
  fixtures_df=pd.DataFrame(fix_response)
  fixtures_df['gw']=i+1
  df_l.append(fixtures_df)
#  print(fix_url)

fixtures_df = pd.concat(df_l, axis=0, ignore_index=True)
fixtures_df_map=fixtures_df[['gw','kickoff_time']].copy()

fixtures_df_map['date']=pd.to_datetime(fixtures_df_map['kickoff_time'], errors='coerce').dt.date

del fixtures_df_map['kickoff_time']
fixtures_df_map=fixtures_df_map.drop_duplicates()
#fixtures_df_map


# In[77]:


fixtures_df_name_cur_gw=fixtures_df_name[['home_team','away_team','id','id_y']]
fixtures_df_name_cur_gw.columns=['home_team','away_team','home_id','away_id']


# In[78]:


spi_data_name_fil=spi_data_name[['team1','team2','team1_id','team2_id','proj_score1','proj_score2']]
spi_data_name_fil.columns=['team1','team2','home_id','away_id','proj_score1','proj_score2']

spi_gw_scores=pd.merge(left=fixtures_df_name_cur_gw,right=spi_data_name_fil,left_on=['home_id','away_id'],right_on=['home_id','away_id'],how='inner')


# In[79]:


players_df_filtered_teams=pd.merge(left=players_df_filtered,right=teams_df_filtered_join,how='inner',left_on='team',right_on='id')
#players_df_filtered_teams


# In[80]:


#Read historical fixtures data

team_list=['understat_Arsenal.csv','understat_Aston_Villa.csv','understat_Brighton.csv','understat_Burnley.csv','understat_Chelsea.csv','understat_Crystal_Palace.csv','understat_Everton.csv','understat_Fulham.csv','understat_Leeds.csv','understat_Leicester.csv','understat_Liverpool.csv','understat_Manchester_City.csv','understat_Manchester_United.csv','understat_Newcastle_United.csv','understat_Sheffield_United.csv','understat_Southampton.csv','understat_Tottenham.csv','understat_West_Bromwich_Albion.csv','understat_West_Ham.csv','understat_Wolverhampton_Wanderers.csv']
static='https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2020-21/understat/'
df_l=[]

for i in range(20):
  k=static+team_list[i]
  temp=pd.read_csv(k)
 # print(temp.shape[0])
  temp['team']=team_list[i]
  df_l.append(temp)

team_total_data = pd.concat(df_l, axis=0, ignore_index=True)

#team_total_data
#team_mapping_spi
team_total_data_id=pd.merge(left=team_total_data,right=team_mapping_spi,left_on='team',right_on='team_file_name',how='left')
team_total_data_id

team_total_data_id=team_total_data_id[['team_y','id','pts','date','h_a','xG','xGA','deep','deep_allowed','scored','missed']]
# #team_total_data_id
team_total_data_id['date_new']=pd.to_datetime(team_total_data_id['date'], errors='coerce').dt.date

del team_total_data_id['date']
team_total_data_id['cum_pts']=team_total_data_id.groupby(['team_y'])['pts'].cumsum()

#team_total_data_id

team_total_data_id_means = team_total_data_id.join(team_total_data_id.groupby(['id']).expanding().agg({'xG':'mean','xGA': 'mean', 'deep': 'mean', 'deep_allowed': 'mean','scored':'mean','missed':'mean'})
               .reset_index(level=0, drop=True)
               .add_suffix('_roll'))
team_total_data_id_means=team_total_data_id_means.groupby(['id', 'date_new']).last().reset_index()
team_total_data_id_means=team_total_data_id_means[['id','date_new','xG_roll','xGA_roll','deep_roll','deep_allowed_roll','scored_roll','missed_roll']]
team_total_data_id_means
#Create a league standings table
team_total_data_id=pd.merge(left=team_total_data_id,right=fixtures_df_map,left_on='date_new',right_on='date',how='inner')
team_total_data_id.sort_values(by='cum_pts',inplace=True,ascending=False)
team_total_data_id=team_total_data_id[['cum_pts','gw','id','date_new']]
team_total_data_id["rank"] = team_total_data_id.groupby("gw")["cum_pts"].rank("dense", ascending=False)
team_total_data_id.columns=['cum_pts','gw','id','date_new','rank']
#team_total_data_id

team_total_data_id=pd.merge(left=team_total_data_id,right=team_total_data_id_means,left_on=['id','date_new'],right_on=['id','date_new'],how='inner')
del team_total_data_id['date_new']
team_total_data_id['xG_diff']=team_total_data_id['xG_roll']-team_total_data_id['xGA_roll']
team_total_data_id['deep_diff']=team_total_data_id['deep_roll']/(team_total_data_id['deep_roll']+team_total_data_id['deep_allowed_roll'])
team_total_data_id['scored_diff']=team_total_data_id['scored_roll']/(team_total_data_id['scored_roll']+team_total_data_id['missed_roll'])
team_total_data_id=team_total_data_id[['cum_pts','gw','id','rank','xG_diff','deep_diff','scored_diff']]
team_total_data_id
#team_total_data_id.to_csv('team_total_data_id.csv')
#files.download('team_total_data_id.csv')


# In[81]:


#Read Gameweek history

#gw_list=['gw1','gw2','gw3','gw4','gw5','gw6','gw7','gw8','gw9','gw10','gw11','gw12','gw13','gw14','gw15']
gw_list=[]
for i in range(GW-1):
  gw_str='gw'+str(i+1)
  gw_list.append(gw_str)
print(gw_list)

static='https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2020-21/gws/'
df_l=[]

for i in range(len(gw_list)):
  k=static+gw_list[i]+'.csv'
  temp=pd.read_csv(k)
  temp['gw']=i+1
  df_l.append(temp)
# print(k)

weekly_data = pd.concat(df_l, axis=0, ignore_index=True)
#weekly_data

weekly_data_team=pd.merge(left=weekly_data,right=teams_df_filtered_join,left_on='opponent_team',right_on='id',how='inner')
weekly_data_team=weekly_data_team.rename(columns={"name_y": "opponent_team_name"})

weekly_data_team=pd.merge(left=weekly_data_team,right=teams_df_filtered_join,left_on='team',right_on='name',how='inner')
weekly_data_team=weekly_data_team.rename(columns={"name_x": "player_name",'id_y':'team_id'})
#weekly_data_team


# In[82]:


weekly_data_team.gw.max()


# In[83]:


#weekly_data_team.iloc[0,33]
#Add the home and away team columns
weekly_data_team['home_team'] = weekly_data_team['team_id']
weekly_data_team['away_team'] = weekly_data_team['team_id']

weekly_data_team.loc[weekly_data_team['was_home'] == True, 'home_team'] = weekly_data_team.loc[weekly_data_team['was_home'] == True, 'team_id']
weekly_data_team.loc[weekly_data_team['was_home'] == True, 'away_team'] = weekly_data_team.loc[weekly_data_team['was_home'] == True, 'opponent_team']

weekly_data_team.loc[weekly_data_team['was_home'] == False, 'home_team'] = weekly_data_team.loc[weekly_data_team['was_home'] == False, 'opponent_team']
weekly_data_team.loc[weekly_data_team['was_home'] == False, 'away_team'] = weekly_data_team.loc[weekly_data_team['was_home'] == False, 'team_id']
#weekly_data_team

weekly_data_team_spi=pd.merge(left=weekly_data_team,right=spi_data_name_fil,left_on=['home_team','away_team'],right_on=['home_id','away_id'],how='inner')
weekly_data_team_spi

#Weekly data remove players who haven't started yet
zero_min=weekly_data_team_spi.groupby(['player_name',]).agg({'minutes':'sum'}).reset_index()
zero_min=zero_min.loc[zero_min['minutes']==0]
#zero_min
#weekly_data_team_spi.player_name.isin(zero_min['player_name']).sum()

weekly_data_team_spi_zero=weekly_data_team_spi[~(weekly_data_team_spi.player_name.isin(zero_min['player_name']))]
# weekly_data_team_spi_zero['team_id']=0
# weekly_data_team_spi_zero.loc[:,'team_id']=weekly_data_team_spi_zero.loc[weekly_data_team_spi_zero['was_home']=True,'home_team']
#weekly_data_team_spi_zero


# In[84]:


#Most points in a gameweek
gp=weekly_data.groupby(['name','gw']).agg({'total_points':'sum'}).reset_index()
gp.sort_values(by='total_points',ascending=False)


# In[85]:


#Distribution of points scored by players
gp=weekly_data_team_spi_zero.groupby(['total_points']).size().reset_index(name='counts')
#gp.sort_values(by='counts',ascending=False)

fig=px.bar(gp,x='total_points',y='counts',title='Distribution of points scored in gameweek by players')
fig.show()


# In[86]:


#Points Distribution by Position
weekly_data_team_spi_zero['point_flag']=0

weekly_data_team_spi_zero.loc[weekly_data_team_spi_zero['total_points']>2,'point_flag']=1
#weekly_data_team_spi_zero

gp=weekly_data_team_spi_zero.groupby(['position','point_flag']).size().reset_index(name='counts')
gp

fig = make_subplots(rows=1, cols=4,specs=[[{'type':'domain'}, {'type':'domain'},{'type':'domain'},{'type':'domain'}]],subplot_titles=("GK","DEF","MID","FWD"))

fig.add_trace(go.Pie(labels=gp.loc[gp['position']=='GK','point_flag'], values=gp.loc[gp['position']=='GK','counts'], name="GK"),1, 1)
fig.add_trace(go.Pie(labels=gp.loc[gp['position']=='DEF','point_flag'], values=gp.loc[gp['position']=='DEF','counts'], name="DEF"),1, 2)
fig.add_trace(go.Pie(labels=gp.loc[gp['position']=='MID','point_flag'], values=gp.loc[gp['position']=='MID','counts'], name="MID"),1, 3)
fig.add_trace(go.Pie(labels=gp.loc[gp['position']=='FWD','point_flag'], values=gp.loc[gp['position']=='FWD','counts'], name="FW"),1, 4)

fig.update_traces(hole=.4, hoverinfo="label+percent+name")

fig.update_layout(
    title_text="Points Distribution by Position")

fig.show()


# In[87]:


#weekly_data_team_spi_zero


# In[88]:


player_summary=weekly_data_team_spi_zero[['player_name','round','team_id','total_points']]
team_gp=player_summary.groupby(['team_id','round']).agg({'total_points':'sum'}).reset_index()
team_gp.columns=['team_id','round','team_total_points']
team_gp

team_gp['team_total_points_cumsum'] = team_gp.groupby(['team_id'])['team_total_points'].cumsum()
team_gp
del team_gp['team_total_points']
# team_player_gp=player_summary.groupby(['team_id','player_name','round']).agg({'total_points':'sum'}).reset_index()
# team_player_gp.columns=['team_id','player_name','round','team_total_points']
# team_player_gp

# from google.colab import files

# team_player_gp.to_csv('team_player_gp.csv')
# files.download('team_player_gp.csv')


# In[89]:


#Feature Engineering
weekly_data_subset=weekly_data_team_spi_zero[['player_name','gw','position','team','round','id_x','team_id','xP','assists','bonus','clean_sheets','goals_scored','goals_conceded','minutes','creativity','ict_index','influence','selected','threat','yellow_cards','red_cards','was_home','home_team','away_team','proj_score1','proj_score2','total_points','point_flag']]
weekly_data_subset=weekly_data_subset.rename(columns={"id_x": "opponent_id"})

#Weekly Cumulative sums
weekly_data_subset.sort_values(by=['player_name','gw'],ascending=True,inplace=True)
gp_cols=weekly_data_subset[['player_name','gw','red_cards','yellow_cards','xP','total_points','bonus','clean_sheets','assists','minutes','influence','creativity','threat','ict_index']]
gp_cols=gp_cols.loc[gp_cols.gw>=GW-4]
gp_cumsum=gp_cols.groupby(['player_name','gw']).sum().groupby('player_name').cumsum().reset_index()

#Weekly Mean Sums
weekly_rolling_means = weekly_data_subset.join(weekly_data_subset.groupby(['player_name']).expanding().agg({'minutes':'mean','influence': 'mean', 'creativity': 'mean', 'threat': 'mean','ict_index':'mean','selected':'mean'})
               .reset_index(level=0, drop=True)
               .add_suffix('_roll'))
weekly_rolling_means=weekly_rolling_means.groupby(['round', 'player_name']).last().reset_index()
weekly_rolling_means=weekly_rolling_means[['player_name','gw','influence_roll', 'creativity_roll', 'threat_roll','ict_index_roll','minutes_roll','selected_roll']]

weekly_data_subset_new=weekly_data_subset[['player_name','gw','position','team','round','team_id','opponent_id','was_home','home_team','away_team','proj_score1','proj_score2','point_flag']].copy()

weekly_data_subset_new['join_key']=0
gw=weekly_data_subset_new['gw']
weekly_data_subset_new.loc[:,'join_key']=gw-1

#Merge cumulative sums
weekly_data_subset_new_agg=pd.merge(left=weekly_data_subset_new,right=weekly_rolling_means,left_on=['player_name','join_key'],right_on=['player_name','gw'],how='inner')

#Merge cumulative means
weekly_data_subset_new_agg=pd.merge(left=weekly_data_subset_new_agg,right=gp_cumsum,left_on=['player_name','join_key'],right_on=['player_name','gw'],how='inner')

#Merge team total ranking stats with home team
weekly_data_subset_new_agg_home=pd.merge(left=weekly_data_subset_new_agg,right=team_total_data_id,left_on=['team_id','join_key'],right_on=['id','gw'])
weekly_data_subset_new_agg_home=weekly_data_subset_new_agg_home.rename(columns={"rank": "team_rank",'cum_pts':'team_pts','xG_diff':'xG_diff_team','scored_diff':'scored_diff_team','deep_diff':'deep_diff_team'})

#Merge team total ranking stats with away team
weekly_data_subset_new_agg_pts=pd.merge(left=weekly_data_subset_new_agg_home,right=team_total_data_id,left_on=['opponent_id','join_key'],right_on=['id','gw'])
weekly_data_subset_new_agg_pts=weekly_data_subset_new_agg_pts.rename(columns={"rank": "opponent_rank",'cum_pts':'opponent_points','xG_diff':'xG_diff_opponent','scored_diff':'scored_diff_opponent','deep_diff':'deep_diff_opponent'})

#Calculate rank and projected score differences for the upcoming fixture
weekly_data_subset_new_agg_pts['rank_diff']=weekly_data_subset_new_agg_pts['team_rank']-weekly_data_subset_new_agg_pts['opponent_rank']
weekly_data_subset_new_agg_pts['proj_score_diff']=weekly_data_subset_new_agg_pts['proj_score1']-weekly_data_subset_new_agg_pts['proj_score2']

#Merge team total ranking stats with away team
weekly_data_subset_new_agg_pts_team=pd.merge(left=weekly_data_subset_new_agg_pts,right=team_gp,left_on=['team_id','join_key'],right_on=['team_id','round'],how='inner')

weekly_data_subset_new_agg_pts_team['percent_team_points']=weekly_data_subset_new_agg_pts_team['total_points']/weekly_data_subset_new_agg_pts_team['team_total_points_cumsum']

# ##del weekly_data_subset_new_agg_pts['gw']
del weekly_data_subset_new_agg_pts_team['gw_y']
del weekly_data_subset_new_agg_pts_team['gw_x']
del weekly_data_subset_new_agg_pts_team['id_x']
del weekly_data_subset_new_agg_pts_team['id_y']
del weekly_data_subset_new_agg_pts_team['join_key']
del weekly_data_subset_new_agg_pts_team['team_id']
del weekly_data_subset_new_agg_pts_team['opponent_id']

weekly_data_subset_new_agg_pts_team

#weekly_data_subset_new_agg_pts_team.to_csv('weekly_data_subset_new_agg_pts_team.csv')
#files.download('weekly_data_subset_new_agg_pts_team.csv')


# In[90]:


# weekly_data_subset=weekly_data_subset.sort_values(by=['player_name','gw']).reset_index(drop=True)
# #weekly_data_subset

# weekly_rolling_means = weekly_data_subset.join(weekly_data_subset.groupby(['player_name']).expanding().agg({'minutes':'mean','influence': 'mean', 'creativity': 'mean', 'threat': 'mean','ict_index':'mean'})
#                .reset_index(level=0, drop=True)
#                .add_suffix('_roll'))

# weekly_rolling_means=weekly_rolling_means.groupby(['round', 'player_name']).last().reset_index()
# weekly_rolling_means=weekly_rolling_means[['player_name','gw','influence_roll', 'creativity_roll', 'threat_roll','ict_index_roll','minutes_roll']]
# weekly_rolling_means
# # k.to_csv('k.csv')
# # files.download('k.csv')


# In[91]:


#Handling categorical variable
labelencoder = LabelEncoder()
weekly_data_subset_new_agg_pts_team['position_flag'] = labelencoder.fit_transform(weekly_data_subset_new_agg_pts_team['position'])

cols = pd.get_dummies(weekly_data_subset_new_agg_pts_team['position_flag'])
cols.columns = ['pos_0','pos_1','pos_2','pos_3']
weekly_data_subset_new_agg_pts_team1 = pd.concat([weekly_data_subset_new_agg_pts_team, cols], axis=1)

#weekly_data_subset_new_agg_pts_team1


# In[92]:


weekly_data_subset_new_agg_pts_team2=weekly_data_subset_new_agg_pts_team1.copy()
X=weekly_data_subset_new_agg_pts_team1.copy()
#Y=weekly_data_subset_new_agg['point_flag']
del X['player_name']
del X['position']
del X['team']
#del X['total_points']
#X_train,X_test,Y_train,Y_test=train_test_split(X, Y,test_size=0.3,random_state=1)

max_gw=X['round_x'].max()
print(max_gw)
#max_gw=14

X_train=X.loc[X['round_x']<max_gw]
X_test=X.loc[X['round_x']==max_gw]
Y_train=X_train['point_flag']
Y_test=X_test['point_flag']
del X_train['point_flag']
del X_test['point_flag']

del X_train['round_x']
del X_test['round_x']

del X_train['round_y']
del X_test['round_y']


# In[93]:


#X_test


# In[94]:



from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

model = RandomForestClassifier(n_estimators=100,max_depth=10, random_state=0,min_samples_split=5,class_weight='balanced_subsample')
model.fit(X_train,Y_train)#Fitting the model 
#Generating predictions from Random Fores Models
pred_rf=model.predict(X_test)
pred_rf_proba=model.predict_proba(X_test)

feat_importances = pd.Series(model.feature_importances_, index=X_train.columns)
feat_importances=feat_importances.sort_values()
feat_importances.plot(kind='barh',figsize=(16,16))#Plotting feature importance

print('Model Accuracy')
print(model.score(X_test,Y_test))


# In[95]:


import statistics

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X_train, Y_train, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
print('Mean ROC AUC: %.3f' % statistics.mean(scores))


# In[96]:


import plotly.express as px
import matplotlib.pyplot as plt

def plot_hist(pred):
  plt.hist(pred[:,1], density=True, bins=30)  # density=False would make counts
  plt.ylabel('Probability')
  plt.xlabel('Data')

plot_hist(pred_rf_proba)
print(pred_rf_proba[:,1].mean())


# In[97]:


#Confusion Matrix & True Positive Rate
calc_pred=pred_rf_proba[:,1]>0.16
calc_pred=calc_pred.astype(int)

conf_mat = confusion_matrix(Y_test, calc_pred)
df_confusion = pd.crosstab(Y_test, calc_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
print (df_confusion)

#print(conf_mat)

# Visualize it as a heatmap
import seaborn
import matplotlib.pyplot as plt

seaborn.heatmap(conf_mat)
plt.show()

FP = conf_mat.sum(axis=0) - np.diag(conf_mat)  
FN = conf_mat.sum(axis=1) - np.diag(conf_mat)
TP = np.diag(conf_mat)
TN = conf_mat.sum() - (FP + FN + TP)
TPR = TP/(TP+FN)

print("True Positive Rate",TPR)


# In[98]:


import sklearn.metrics as metrics
import matplotlib.pyplot as plt


def plot_curve(pred):
  fpr, tpr, threshold = metrics.roc_curve(Y_test, pred[:,1])
  roc_auc = metrics.auc(fpr, tpr)

# method I: plt
  plt.title('Receiver Operating Characteristic')
  plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
  plt.legend(loc = 'lower right')
  plt.plot([0, 1], [0, 1],'r--')
  plt.xlim([0, 1])
  plt.ylim([0, 1])
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')
  plt.show()
plot_curve(pred_rf_proba)


# In[99]:


# optimal_proba_cutoff = sorted(list(zip(np.abs(tpr - fpr), pred_rf_proba)), key=lambda i: i[0], reverse=True)[0][1]
# print(optimal_proba_cutoff)

# roc_predictions=calc_pred=pred_rf_proba[:,1]>0.26
# roc_predictions=roc_predictions.astype(int)

#roc_predictions = [1 if i >= optimal_proba_cutoff else 0 for i in predicted_proba[:, -1]]


# In[100]:


# print("Accuracy Score Before and After Thresholding: {}, {}".format(accuracy_score(Y_test, roc_predictions), accuracy_score(Y_test, roc_predictions)))
# print("Precision Score Before and After Thresholding: {}, {}".format(precision_score(Y_test, roc_predictions), precision_score(Y_test, roc_predictions)))
# print("Recall Score Before and After Thresholding: {}, {}".format(recall_score(Y_test, roc_predictions), recall_score(Y_test, roc_predictions)))


#  XGBoost

# In[101]:


from xgboost import XGBClassifier


model = XGBClassifier(max_depth=7,
   # 'max_leaves' : 2**4,
    alpha=0.1, 
   # 'lambda' : 0.2,
    subsample=0.8,
    learning_rate=0.1, #default = 0.3,
    colsample_bytree=0.7,
    eval_metric='auc', 
    objective = 'binary:logistic',
    grow_policy='lossguide',
    n_estimators=100)
model.fit(X_train, Y_train)
pred_xg_proba=model.predict_proba(X_test)

feat_importances = pd.Series(model.feature_importances_, index=X_train.columns)
feat_importances=feat_importances.sort_values()
feat_importances.plot(kind='barh',figsize=(16,16))#Plotting feature importance


# In[102]:


plot_curve(pred_xg_proba)
plot_hist(pred_xg_proba)

calc_pred=pred_xg_proba[:,1]>0.16
calc_pred=calc_pred.astype(int)

conf_mat = confusion_matrix(Y_test, calc_pred)
df_confusion = pd.crosstab(Y_test, calc_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
print (df_confusion)


# In[103]:


X_cols=weekly_data_subset_new_agg_pts_team2[['player_name','position','team','total_points','point_flag']].copy()
X_test_pred=pd.merge(X_test, X_cols, left_index=True, right_index=True)
#X_test_pred['predictions']=pred_rf_proba[:,1]
X_test_pred['predictions']=pred_xg_proba[:,1]


# In[104]:


X_test_pred.loc[X_test_pred['predictions']>0.5].sort_values(by='predictions',ascending=False).head(20)


# In[105]:


#from google.colab import files

X_test_pred.to_csv('df.csv')
#files.download('df.csv')


# In[106]:


weekly_data_subset['round'].max()


# In[107]:


#Prepare dataset for future game week predictions
fixtures_df1=fixtures_df[['event','team_h','team_a']]
max_week=weekly_data_subset['round'].max()
print(max_week)
pred_data=weekly_data_subset.loc[weekly_data_subset['round']==max_week]
fixtures_df1=fixtures_df1.loc[fixtures_df1['event']==max_week+1]

#fixtures_df1=fixtures_df1.loc[fixtures_df1['event']==max_week]

pred_data=pred_data[['player_name','position','team','team_id','round']].reset_index(drop=True)
pred_data
#fixtures_df

pred_data=pd.merge(left=pred_data,right=fixtures_df1,left_on='team_id',right_on='team_h',how='left')
pred_data=pd.merge(left=pred_data,right=fixtures_df1,left_on='team_id',right_on='team_a',how='left')

pred_data.loc[pred_data['team_h_x'].isna(),'team_h_x']=pred_data.loc[pred_data['team_h_x'].isna(),'team_h_y']
pred_data.loc[pred_data['team_a_x'].isna(),'team_a_x']=pred_data.loc[pred_data['team_a_x'].isna(),'team_a_y']
pred_data=pred_data[['player_name','position','team','team_id','team_h_x','team_a_x','round']]
pred_data=pred_data.rename(columns={"round": "gw","team_h_x": "home_team","team_a_x":"away_team"})

# #weekly_data_subset_new=weekly_data_subset[['player_name','gw','position','team','round','team_id','opponent_id','selected','was_home','home_team','away_team','proj_score1','proj_score2','point_flag']].copy()

# #Home Flag
pred_data['was_home']=0
pred_data.loc[pred_data['home_team']==pred_data['team_id'],'was_home']=1

#TOBECHANGED
pred_data['gw']=GW-1

pred_data['opponent_id']=0

pred_data['opponent_id']=pred_data['home_team']
pred_data.loc[pred_data['was_home']==1,'opponent_id']=pred_data.loc[pred_data['was_home']==1,'away_team']
#pred_data

pred_data_roll=pd.merge(left=pred_data,right=weekly_rolling_means,left_on=['player_name','gw'],right_on=['player_name','gw'],how='inner')

# #Merge cumulative means
pred_data_roll_cum=pd.merge(left=pred_data_roll,right=gp_cumsum,left_on=['player_name','gw'],right_on=['player_name','gw'],how='inner')
#pred_data_roll_cum

# # #Merge team total ranking stats with home team
pred_data_roll_cum_team=pd.merge(left=pred_data_roll_cum,right=team_total_data_id,left_on=['team_id','gw'],right_on=['id','gw'])
pred_data_roll_cum_team=pred_data_roll_cum_team.rename(columns={"rank": "team_rank",'cum_pts':'team_pts','xG_diff':'xG_diff_team','scored_diff':'scored_diff_team','deep_diff':'deep_diff_team'})

# # #Merge team total ranking stats with away team
pred_data_roll_cum_team=pd.merge(left=pred_data_roll_cum_team,right=team_total_data_id,left_on=['opponent_id','gw'],right_on=['id','gw'])
pred_data_roll_cum_team=pred_data_roll_cum_team.rename(columns={"rank": "opponent_rank",'cum_pts':'opponent_points','xG_diff':'xG_diff_opponent','scored_diff':'scored_diff_opponent','deep_diff':'deep_diff_opponent'})
pred_data_roll_cum_team

# #Merge team total ranking stats with away team
pred_data_roll_cum_team_pts=pd.merge(left=pred_data_roll_cum_team,right=team_gp,left_on=['team_id','gw'],right_on=['team_id','round'],how='inner')
pred_data_roll_cum_team_pts['percent_team_points']=pred_data_roll_cum_team_pts['total_points']/pred_data_roll_cum_team_pts['team_total_points_cumsum']
pred_data_roll_cum_team_pts['gw']=max_week+1

pred_data_roll_cum_team_pts_spi=pd.merge(left=pred_data_roll_cum_team_pts,right=spi_data_name_fil,left_on=['home_team','away_team'],right_on=['home_id','away_id'],how='inner')

# #Calculate rank and projected score differences for the upcoming fixture
pred_data_roll_cum_team_pts_spi['rank_diff']=pred_data_roll_cum_team_pts_spi['team_rank']-pred_data_roll_cum_team_pts_spi['opponent_rank']
pred_data_roll_cum_team_pts_spi['proj_score_diff']=pred_data_roll_cum_team_pts_spi['proj_score1']-pred_data_roll_cum_team_pts_spi['proj_score2']

pred_data_roll_cum_team_pts_spi


# In[108]:


pred_data_roll_cum_team_pts_spi['position_flag'] = labelencoder.fit_transform(pred_data_roll_cum_team_pts_spi['position'])
cols = pd.get_dummies(pred_data_roll_cum_team_pts_spi['position_flag'])
cols.columns = ['pos_0','pos_1','pos_2','pos_3']
pred_data_roll_cum_team_pts_spi1 = pd.concat([pred_data_roll_cum_team_pts_spi, cols], axis=1)
pred_data_roll_cum_team_pts_spi1

pred_data_roll_cum_team_pts_spi1.drop(['player_name','position','team','team1','team2','away_id','home_id','id_x','id_y','opponent_id','team_id','round'],axis=1,inplace=True)

#Reorder columns in the test data to be in the same order as train data
pred_data_roll_cum_team_pts_spi1=pred_data_roll_cum_team_pts_spi1[X_train.columns]
proba=model.predict_proba(pred_data_roll_cum_team_pts_spi1)
pred_data_roll_cum_team_pts_spi['predicted']=proba[:,1]


# In[109]:


pred_data_roll_cum_team_pts_spi.sort_values(by='predicted',ascending=False).head(15)


# In[110]:


pred_data_roll_cum_team_pts_spi.to_csv('pred.csv')
#files.download('pred.csv')


# In[111]:


pred_data_roll_cum_team_pts_spi.sort_values(by='predicted',ascending=False,inplace=True)

team_list=[]

gk=pred_data_roll_cum_team_pts_spi.loc[pred_data_roll_cum_team_pts_spi['position']=='GK',['player_name','position','team_id','gw']].head(1)
team_list.append(gk)

defender=pred_data_roll_cum_team_pts_spi.loc[pred_data_roll_cum_team_pts_spi['position']=='DEF',['player_name','position','team_id','gw']].head(3)
team_list.append(defender)
midfielder=pred_data_roll_cum_team_pts_spi.loc[pred_data_roll_cum_team_pts_spi['position']=='MID',['player_name','position','team_id','gw']].head(4)
team_list.append(midfielder)
forward=pred_data_roll_cum_team_pts_spi.loc[pred_data_roll_cum_team_pts_spi['position']=='FWD',['player_name','position','team_id','gw']].head(3)
team_list.append(forward)

team_list


# In[112]:



output=pd.concat(team_list)

output['web_name'] = output['player_name'].str.split(' ').str[1]
player_photo=players_df[['first_name','second_name','photo','code','team']].copy()
player_photo['web_name']=player_photo['first_name']+' '+player_photo['second_name']
player_photo
output_join=pd.merge(left=output,right=player_photo,left_on=['player_name','team_id'],right_on=['web_name','team'],how='left')
output_join=output_join[['player_name','position','photo','gw']]
output_join['new_photo']=output_join['photo'].str.split('.').str[0]
del output_join['photo']
output_join.to_csv('output.csv',index=False)
#files.download('output.csv')

#output_join


# In[113]:


#Appending to the predictions file
output_file=pd.read_csv('https://raw.githubusercontent.com/arpitsolanki/FPLBot/main/output.csv')
# output_file

final_output=output_join.append(output_file, ignore_index=True)
final_output.to_csv('output.csv',index=False)
#files.download('output.csv')


# In[114]:


#Past actual game week scores for my dream team
last_gw_team=output_file.loc[output_file.gw==GW-1]
last_gw_team

weekly_data_last=weekly_data_team[['player_name','total_points','gw','position']]
weekly_data_last=weekly_data_last.loc[weekly_data_last.gw==GW-1]

last_gw_team_pts=pd.merge(left=last_gw_team,right=weekly_data_last,left_on=['player_name','gw','position'],right_on=['player_name','gw','position'],how='inner')
last_gw_team_pts

last_gw_team_pts.to_csv('gw_points_history.csv',index=False)
#files.download('gw_points_history.csv')

#Dream Team past GW
weekly_data_last.sort_values(by=['total_points'],ascending=False,inplace=True)
weekly_data_last.head(10)

dream_team_list=[]
gk=weekly_data_last.loc[weekly_data_last['position']=='GK',['player_name','position','total_points']].head(1)
dream_team_list.append(gk)

defender=weekly_data_last.loc[weekly_data_last['position']=='DEF',['player_name','position','total_points']].head(3)
dream_team_list.append(defender)
midfielder=weekly_data_last.loc[weekly_data_last['position']=='MID',['player_name','position','total_points']].head(4)
dream_team_list.append(midfielder)
forward=weekly_data_last.loc[weekly_data_last['position']=='FWD',['player_name','position','total_points']].head(3)
dream_team_list.append(forward)

last_gw_dream_team=pd.concat(dream_team_list)
last_gw_dream_team.to_csv('last_gw_dream_team.csv',index=False)
#files.download('last_gw_dream_team.csv')


# In[115]:


avg_score_df=events_df.loc[events_df.id<=GW]
avg_score_df=avg_score_df[['id','average_entry_score']]

avg_score_df.to_csv('avg_score_df.csv',index=False)
#files.download('avg_score_df.csv')


# In[122]:


#output_join.iloc[0,2]


# img=output_join.iloc[0,2]
# img_path='https://resources.premierleague.com/premierleague/photos/players/110x140/p'+img+'.png'
# print(img_path)
# from skimage import io
# io.imshow(io.imread(img_path))
# #io.show()
#import plotly
import skimage
#skimage.io.show()
#from skimage import io
print(seaborn.__version__)

