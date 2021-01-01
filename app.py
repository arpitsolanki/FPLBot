%%writefile app.py
import streamlit as st
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def main():

  st.title("FPL Dream Team Predictions ")

  team=pd.read_csv('https://raw.githubusercontent.com/arpitsolanki/FPLBot/main/output.csv')
  team['new_photo']=team['new_photo'].astype(str)

  gw_list=team['gw'].unique().tolist()
  max_gw=team['gw'].max()
#  menu = ["Gameweek 15"]
  menu=gw_list
  choice = st.sidebar.selectbox('Gameweek Number',menu)
 # st.text(choice)
  team=team.loc[team.gw==int(choice)]

  st.subheader("Team Predictions for GameWeek "+str(choice))
  st.markdown("The aim of this application is to use Machine Learning to predict which players are most likely to score 2+ points during a gameweek. The model uses stats like player's historical performance along with score predictions and fixtures related information to identify top 11 players who are likely to be part of the dream team during the week")

  layout=[3,3,4,3]
  k=0
  for i in range(4):
    #if k==0:
    #  buffer, col1, buffer1 = st.beta_columns([1,1,1])
    #else :
    col= st.beta_columns(layout[i])

    for j in range(layout[i]):
#      col=lst[i]
      img=team.iloc[k,3]
      img_path='https://resources.premierleague.com/premierleague/photos/players/110x140/p'+img+'.png'
      img=io.imread(img_path)
      if k==0:
        col[1].image(img, caption=team.iloc[k,0],width=50)
        k=k+1
        break
      else:
        col[j].image(img, caption=team.iloc[k,0],width=50)
      
      k=k+1
  #Populate line chart for previous weeks

  if choice < max_gw:
    col1,col2= st.beta_columns((2,2))
    col1.subheader("Pts - Predicted vs Actual Team")

    last_gw_dream_team=pd.read_csv('https://raw.githubusercontent.com/arpitsolanki/FPLBot/main/last_gw_dream_team.csv')
    dream_team_points=last_gw_dream_team['total_points'].sum()

    gw_points_history=pd.read_csv('https://raw.githubusercontent.com/arpitsolanki/FPLBot/main/gw_points_history.csv')
    my_team_points=gw_points_history['total_points'].sum()

    avg_score_df=pd.read_csv('https://raw.githubusercontent.com/arpitsolanki/FPLBot/main/avg_score_df.csv')
    avg_score=avg_score_df.loc[avg_score_df.id==choice,'average_entry_score'].sum()
#    st.text(avg_score)

    d = {'cat': ['dream_team_points','gw_average_points','my_team_points'], 'pts': [dream_team_points,avg_score,my_team_points]}
    chart_data = pd.DataFrame(data=d)
    cat=['dream_team_pts','gw_avg_pts','my_team_pts']

    fig = go.Figure([go.Bar(x=cat, y=[dream_team_points,avg_score,my_team_points])])
    fig.update_layout(width=400,height=300,margin=dict(l=5, r=5, t=5, b=5))
    col1.plotly_chart(fig,width=350,height=300)
    
    col2.subheader("Points scored by predicted team")
    col2.write(gw_points_history[['player_name','total_points']])

    # st.bar_chart(chart_data)
#     alt.Chart(chart_data).mark_bar().encode(
#     x='cat',
#     y='pts'
# )
 # layout=[1,3,4,3]

if __name__ == '__main__':
	main()
