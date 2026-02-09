import os
import streamlit as st
import pandas as pd
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import re
import plotly.figure_factory as ff
from datetime import timedelta
import matplotlib.pyplot as plt
from rounds_helpers.parsers import (
    parse_timestamp, 
    parse_json_safely, 
    standardize_timestamps
)
from rounds_helpers.visuals import create_timeline_plots

# Load and preprocess data
@st.cache_data
def load_data(file_path='data/combining/combined'):
    file_path_game = os.path.join(file_path, 'game.csv')
    file_path_round = os.path.join(file_path, 'round.csv')
    file_path_batch = os.path.join(file_path, 'batch.csv')

    # Load game data
    df = pd.read_csv(file_path_game)
    df_round = pd.read_csv(file_path_round)
    df_batch = pd.read_csv(file_path_batch)
    return df, df_round, df_batch

@st.cache_data
def process_game_data(df_game, df_round, df_batch):
    # First filter games based on batch status
    ended_batch_ids = df_batch[df_batch['status'] == 'ended']['id'].unique()
    
    # Filter games where both batch is ended and game is ended
    df_game = df_game[
        (df_game['batchID'].isin(ended_batch_ids)) & 
        (df_game['ended'] == True)
    ]
    
    # Filter out initialSurvey and training rounds
    df_round_filtered = df_round[~df_round['name'].isin(['initialSurvey', 'training'])]
    
    # Group by gameId and create a dictionary of round information
    round_info = {}
    for game_id, game_rounds in df_round_filtered.groupby('gameID'):
        round_info[game_id] = {}
        for _, row in game_rounds.iterrows():
            round_id = row['roundId']
            round_info[game_id][f'round_{round_id}_startTime'] = pd.to_datetime(row['startLastChangedAt'], utc=True)
    
    # Convert startLastChangedAt to UTC datetime
    df_game['startLastChangedAt'] = pd.to_datetime(df_game['startLastChangedAt'], utc=True)
    
    # Add round start times to df_game
    for idx, row in df_game.iterrows():
        game_id = row['id']
        if game_id in round_info:
            for round_key, start_time in round_info[game_id].items():
                df_game.at[idx, round_key] = start_time

    action_columns = ['actions-0', 'actions-1', 'actions-2']
    comment_columns = ['comments-0', 'comments-1', 'comments-2']
    
    # First parse JSON safely
    for col in action_columns + comment_columns:
        df_game[col] = df_game[col].apply(parse_json_safely)
    
    # Apply timestamp standardization to parsed JSON
    for col in action_columns + comment_columns:
        df_game[col] = df_game[col].apply(standardize_timestamps)

    return df_game, action_columns, comment_columns


def main():
    df, df_round, df_batch = load_data()

    df, action_columns, comment_columns = process_game_data(df, df_round, df_batch)

    st.title("Game Analysis")

    # Sidebar for game selection
    game_ids = df['id'].unique()
    selected_game = st.sidebar.selectbox("Select a Game ID", game_ids)

    # Select all the game_ids and match on id in df_game
    df_selected = df[df['id'] == selected_game]

    st.write(df_selected)

    st.header(f"Game Overview: {selected_game}")

    st.subheader(f"Game Type: {df_selected['treatmentName'].iloc[0]}")

    # Aggregate metrics for the game
    col1, col2 = st.columns(2)
    with col1:
        total_comments = sum(len(df_selected[col].iloc[0]) for col in comment_columns)
        st.metric("Total Comments", total_comments)
    with col2:
        total_actions = sum(len(df_selected[col].iloc[0]) for col in action_columns)
        st.metric("Total Actions", total_actions)

    st.markdown("---")

    create_timeline_plots(df_selected)

if __name__ == "__main__":
    main()