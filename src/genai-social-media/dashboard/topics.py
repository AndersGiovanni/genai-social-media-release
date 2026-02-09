import streamlit as st
import pandas as pd
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
from engagement_helpers.parsers import filter_treatment_users_by_treatment_usage, process_engagement_data
from treatments.parsers import (
    process_player_rounds, 
    match_selections_to_comments
)
from topics_helpers.metrics import display_topic_metrics, display_suggestions_metrics_by_topic
from topics_helpers.visuals import create_generation_length_comparison_plot, create_generation_likes_distribution_plot, create_generation_sentiment_comparison_plot
from treatments.treatment_info import display_suggestions_metrics
from rounds_game import load_data, process_game_data

def main():
    """Main function to run the dashboard"""
    # Setup page
    st.subheader("Topic Analysis Dashboard")
    st.markdown("""
    In this page we will look at the topic-specific engagement for the control and treatment groups. Below are some high level metrics.
    """)
    # Add treatment selector
    df, df_round, df_batch = load_data()
    df_player_round = pd.read_csv('data/combining/combined/playerRound.csv')
    df, action_columns, comment_columns = process_game_data(df, df_round, df_batch)


    
    # Process data for both control and treatment groups
    processed_data, invalid_items = process_engagement_data(
        df,
        control_group='baseline_5'
    )

    FILTER_DATA_BASED_ON_USERS_USING_TREATMENTS = False
    if FILTER_DATA_BASED_ON_USERS_USING_TREATMENTS:
        processed_data, users_using_treatments = filter_treatment_users_by_treatment_usage(processed_data)

    player_round_data = process_player_rounds(processed_data, df_player_round)

    detailed_data = display_suggestions_metrics(player_round_data, display=False) # SHOULD BE USED IN SUGGESTIONS

    df_generations_matched = match_selections_to_comments(processed_data, detailed_data)

    st.divider()

    topics_content, total_games = display_topic_metrics(df_generations_matched)

    # topic_suggestions_metrics = display_suggestions_metrics_by_topic(df_generations_matched) # SHOULD BE USED IN SUGGESTIONS

    st.subheader("Comparison of Generation Lengths")
    st.markdown("""
    We look at the distribution of comment lengths for comments made with and without AI-generations. We also include the control group for comparison.
    """)
    generation_length_plot = create_generation_length_comparison_plot(df_generations_matched)
    st.plotly_chart(generation_length_plot)

    generation_likes_distribution_plot = create_generation_likes_distribution_plot(df_generations_matched)
    st.subheader("Distribution of Like Types by Topic")
    st.markdown("""
    The plots show either raw counts or proportions of like types for comments, as well as seed content, for each topic. The "with" and "without" are for suggestions, where the "with" is for suggestions that were used.
                The proportions are within comment types. 
                I know the plots are a bit messy, but essentially what you can see here, is how reactions are distributed across different types of comments (with and without AI generations) for both control and treatment and across the different topics.
    """)
    st.plotly_chart(generation_likes_distribution_plot)

    st.divider()

    st.subheader("Sentiment Comparison Between Comments with and without Generations")
    st.markdown("""
    The plots show the mean sentiment scores for all groups, for each topic.
    The error bars represent the standard deviation of the sentiment scores. Each text gets a score for how positive, negative or neutral it is - this plot shows the mean of these scores.
    """)
    generation_sentiment_plot = create_generation_sentiment_comparison_plot(df_generations_matched)
    st.plotly_chart(generation_sentiment_plot)

if __name__ == "__main__":
    main()