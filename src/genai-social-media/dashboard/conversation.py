from typing import List
import streamlit as st
import pandas as pd
from engagement_helpers.parsers import process_engagement_data
from treatments.parsers import (
    process_player_rounds, 
)
  
from rounds_game import load_data, process_game_data
from treatments.treatment_info import display_conversation_metrics
from topics_helpers.metrics import display_conversation_metrics_by_topic
from treatments.conversation_tree_for_regression import ConversationTree
import statsmodels.api as sm
import statsmodels.formula.api as smf
import json

st.set_page_config(layout="wide")

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

    st.header("Treatment 4: Conversation Starter Analysis")
    st.markdown("Here we look at how the users used the open conversation starter feature.")

    conversation_metrics = display_conversation_metrics(player_round_data)

    st.divider()
    st.subheader("Conversation Metrics by Topic")
    st.markdown("In the metrics below, we look at how the conversation starter is used in different topics of the game. ")
    conversation_metrics_by_topic = display_conversation_metrics_by_topic(player_round_data)

if __name__ == "__main__":
    main()