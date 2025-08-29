# and we are using two datasets summary_analysis.xlsx and data_2023_final.csv
# but there is a mismatch in the player's name spelling in both the datasets

import streamlit as st
import pandas as pd
import numpy as np
import io 
#cleaning the dataset before loading to the streamlit app like removing extra spaces tabs new lines and converting the numeric columns to float
def load_and_clean_data(summary_content, overall_content):
    
    summary_df = pd.read_excel(io.BytesIO(summary_content), sheet_name='Sheet1')
    overall_df = pd.read_csv(io.BytesIO(overall_content))
    

    summary_df.columns = (summary_df.columns
                          .str.strip()
                          .str.replace('"', '', regex=False)
                          .str.replace('\t', '', regex=False)
                          .str.replace('\n', '', regex=False)
                          .str.replace('\r', '', regex=False)
                          .str.replace(r'\s+', ' ', regex=True)) 
    

    st.write("Cleaned Column Names:", list(summary_df.columns))
    
    # Numeric columns list from summary_analysis.xlsx
    numeric_cols = ['Total Balls Bowled', 'Total Runs', 'Sixes', 'Fours', 'Dot Balls', 
                    'Wide Balls', 'Leg Byes', 'Byes', 'No Balls', 'Penalties']
    

    summary_df.fillna(0, inplace=True)
    
    
    existing_numeric = [col for col in numeric_cols if col in summary_df.columns]
    if not existing_numeric:
        st.error("No matching numeric columns found")
        return None, None
    
    try:
        summary_df[existing_numeric] = summary_df[existing_numeric].astype(float)
    except ValueError as e:
        st.error(f"Error converting columns to float: {e}")
        return None, None
    
    overall_df.fillna(0, inplace=True)
    return summary_df, overall_df

# calculateing initial stats for players to like rate afterwards that ki which player is good in batting bowling like that 
def calculate_initial_stats(summary_df, overall_df):#in research paper like it is mentioned they calculated r(average runs per ball) and avg(average runs per wicket)
    player_stats = {}
    
    
    for _, row in overall_df.iterrows():
        player = row['Player_Name']
        player_stats[player] = {
            'type': [],#like getting all theses values from the data_2023_final dataset 
            'total_runs': row['Runs_Scored'],
            'total_balls_faced': row['Balls_Faced'],
            'total_dismissals': row['Matches_Batted'] - row['Not_Outs'],
            'total_runs_conceded': row['Runs_Conceded'],
            'total_balls_bowled': row['Balls_Bowled'],
            'total_wickets': row['Wickets_Taken'],
        }
        
        # like here i am having a doubt like to choose that a player is batsman or bowler or all rounder what should be the conditions like 
        if pd.notnull(row['BattingStyle']):# currently i simply used batting style column is not there for any player so its a batsman and its bowling style is not there so its a bowler and the player which has both is a all rounder 
            player_stats[player]['type'].append('batsman')#but like in our dataset we have filled all the players batting and bowling style so according that all the players are all rounders we have to choose something different to indentify its a bowler batter or all rounder
        if pd.notnull(row['BowlingStyle']):
            player_stats[player]['type'].append('bowler')
        
        # Calculated r and avg for batsmen
        if 'batsman' in player_stats[player]['type']:
            balls_faced = player_stats[player]['total_balls_faced']
            dismissals = player_stats[player]['total_dismissals']
            player_stats[player]['r_bat'] = player_stats[player]['total_runs'] / balls_faced if balls_faced > 0 else 0
            player_stats[player]['avg_bat'] = player_stats[player]['total_runs'] / dismissals if dismissals > 0 else 0
        
        # here for bowler
        if 'bowler' in player_stats[player]['type']:
            balls_bowled = player_stats[player]['total_balls_bowled']
            wickets = player_stats[player]['total_wickets']
            player_stats[player]['r_bowl'] = player_stats[player]['total_runs_conceded'] / balls_bowled if balls_bowled > 0 else 0
            player_stats[player]['avg_bowl'] = player_stats[player]['total_runs_conceded'] / wickets if wickets > 0 else 0
    
    return player_stats

# calculate_quality_index accoring to the paper 
def calculate_quality_index(player_stats, summary_df, iterations=3):
    
    for player in player_stats:
        if 'batsman' in player_stats[player]['type']:
            player_stats[player]['phi_bat'] = player_stats[player]['r_bat'] * player_stats[player]['avg_bat']
        if 'bowler' in player_stats[player]['type']:
            player_stats[player]['phi_bowl'] = player_stats[player]['r_bowl'] * player_stats[player]['avg_bowl']
    
    # Iterative update based on matchups with the opponents 
    for _ in range(iterations):
        for _, row in summary_df.iterrows():
            batsman = row['batter']
            bowler = row['bowler']
            runs = row['Total Runs']
            balls = row['Total Balls Bowled']
            
            if batsman in player_stats and bowler in player_stats and balls > 0:
                # for batsmen
                weight = player_stats[bowler].get('phi_bowl', 1)
                adjusted_r = (runs / balls) * weight
                player_stats[batsman]['r_bat'] = (player_stats[batsman]['r_bat'] + adjusted_r) / 2
                player_stats[batsman]['phi_bat'] = player_stats[batsman]['r_bat'] * player_stats[batsman]['avg_bat']
                
                # for bowler
                adjusted_avg = player_stats[bowler]['avg_bowl'] / weight
                player_stats[bowler]['avg_bowl'] = (player_stats[bowler]['avg_bowl'] + adjusted_avg) / 2
                player_stats[bowler]['phi_bowl'] = player_stats[bowler]['r_bowl'] * player_stats[bowler]['avg_bowl']
    
    return player_stats

# creating two types of embeddings level 1 and level 2 according to the paper 
def create_embeddings(summary_df, player_ratings):
    batsmen = summary_df['batter'].unique()
    bowlers = summary_df['bowler'].unique()
    
    # so this is creating level 1 embeddings
    level1 = pd.DataFrame(0, index=batsmen, columns=bowlers)
    for _, row in summary_df.iterrows():
        batsman = row['batter']
        bowler = row['bowler']
        if batsman in player_ratings and bowler in player_ratings:
            level1.at[batsman, bowler] = player_ratings[batsman].get('phi_bat', 0) - player_ratings[bowler].get('phi_bowl', 0)
    
    # this is creating level 2 embeddings by thresholding level 1
    level2 = level1.applymap(lambda x: 1 if x > 0 else 0)
    
    return level1, level2

# team recommendation based on the input given by the user 
def recommend_team(your_pool_list, opposition_list, composition, level1, level2):
    rankings = {}
    for player in your_pool_list:
        if player in level1.index:
            edges = []
            for opp in opposition_list:
                if opp in level1.columns and level2.at[player, opp] == 1:
                    edges.append(level1.at[player, opp])
            if edges:
                mean_edge = np.mean(edges)
                std_edge = np.std(edges) if len(edges) > 1 else 1
                rankings[player] = mean_edge / std_edge
    
    sorted_players = sorted(rankings, key=rankings.get, reverse=True)
    
    recommended_team = sorted_players[:sum(composition.values())]
    
    return recommended_team


st.title("GameSage Team Recommendation System")

st.header("Upload Datasets")
summary_upload = st.file_uploader("Upload summary_analysis.xlsx", type=['xlsx'])
overall_upload = st.file_uploader("Upload data_2023_final.csv", type=['csv'])

if summary_upload and overall_upload:
    summary_content = summary_upload.read()
    overall_content = overall_upload.read()
    
    summary_df, overall_df = load_and_clean_data(summary_content, overall_content)
    if summary_df is not None and overall_df is not None:
        st.success("Datasets uploaded and cleaned successfully!")
        
        player_stats = calculate_initial_stats(summary_df, overall_df)
        player_ratings = calculate_quality_index(player_stats, summary_df)
        level1, level2 = create_embeddings(summary_df, player_ratings)
        
        st.header("Enter Your Team Pool")
        your_pool = st.text_input("Your players", "Shubman Gill,Anuj Rawat,SE Rutherford")
        your_pool_list = [p.strip() for p in your_pool.split(',')]
        
        st.header("Enter Opposition Team (comma separated names)")
        opposition = st.text_input("Opposition players", "KK Ahmed,T Natarajan")
        opposition_list = [o.strip() for o in opposition.split(',')]
        
        st.header("Team Composition (e.g., batsmen:2, bowlers:1)")
        comp_input = st.text_input("Composition", "batsmen:2,bowlers:1")
        composition = {}
        for item in comp_input.split(','):
            key, val = item.split(':')
            composition[key.strip()] = int(val.strip())
        
        if st.button("Recommend Team"):
            recommended_team = recommend_team(your_pool_list, opposition_list, composition, level1, level2)
            st.header("Recommended Team")
            st.write(recommended_team)
else:
    st.warning("Please upload both datasets.")
