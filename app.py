import streamlit as st
import pandas as pd

#PAGE_CONFIG = {"page_title":"StColab.io","page_icon":":smiley:","layout":"centered"}
#st.beta_set_page_config(**PAGE_CONFIG)
def main():

  st.title("Fantasy Dream Team Predictions ")
  st.subheader("Predictions for GameWeek 15")
  team=pd.read_csv('https://raw.githubusercontent.com/arpitsolanki/FPLBot/main/output.csv')
  st.dataframe(team)

  menu = ["Gameweek 15"]
  choice = st.sidebar.selectbox('Menu',menu)
	# if choice == 'Home':
	# 	st.subheader("Streamlit From Colab")	
  
if __name__ == '__main__':
	main()
