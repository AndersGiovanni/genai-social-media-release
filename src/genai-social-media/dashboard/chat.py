import streamlit as st
import pandas as pd
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
from engagement_helpers.parsers import process_engagement_data
from treatments.parsers import (
    process_player_rounds, 
    match_selections_to_comments
)
  
from rounds_game import load_data, process_game_data
from treatments.treatment_info import display_chat_metrics
from topics_helpers.metrics import display_chat_metrics_by_topic, analyze_commenters_by_topic_distribution
from treatments.conversation_tree_for_regression import ConversationTree
import statsmodels.formula.api as smf
import statsmodels.api as sm
from typing import List

def main():
    """Main function to run the dashboard"""
    # Setup page
    
    df, df_round, df_batch = load_data()
    df_player_round = pd.read_csv('data/combining/combined/playerRound.csv')
    df, action_columns, comment_columns = process_game_data(df, df_round, df_batch)

    # Process data for both control and treatment groups
    processed_data, invalid_items = process_engagement_data(
        df,
        control_group='baseline_5'
    )

    player_round_data = process_player_rounds(processed_data, df_player_round)

    st.header("Treatment 2: Chat Analysis")
    st.markdown("Here we look at how the users used the open chat feature. It is worth noting, that they max use per user per round is 8.")

    chat_metrics = display_chat_metrics(player_round_data)

    st.divider()
    st.subheader("Chat Usage Metrics by Topic")
    st.markdown("In the metrics below, we look at how the AI chat is used in different topics of the game. "
                "We look at the total number of __prompts__ (when a user sends a message to the AI), "
                "the number of unique users, the number of users who have used the chat, and various engagement metrics.")
    topic_chat_metrics = display_chat_metrics_by_topic(player_round_data)

    producers = analyze_commenters_by_topic_distribution(player_round_data, topic_chat_metrics)

if __name__ == "__main__":
    main()