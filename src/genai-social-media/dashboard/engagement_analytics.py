import streamlit as st
import pandas as pd
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
from engagement_helpers.visuals import (
    create_engagement_plots,
    create_metric_comparison_plots,
    create_comment_length_plot,
    create_likes_distribution_plot,
    create_sentiment_plot,
    process_time_series_data,
    create_toxicity_plot,
    create_time_series_plots
)
from engagement_helpers.parsers import filter_treatment_users_by_treatment_usage, process_engagement_data
from engagement_helpers.page_elements import display_comparative_metrics, shannon_entropy_comparison, shannon_entropy_comparison_pointplot
from rounds_game import load_data, process_game_data
from treatments.parsers import (
    process_player_rounds
)

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

    FILTER_DATA_BASED_ON_USERS_USING_TREATMENTS = False
    if FILTER_DATA_BASED_ON_USERS_USING_TREATMENTS:
        processed_data, users_using_treatments = filter_treatment_users_by_treatment_usage(processed_data)

    st.header("Engagement Analytics")
    st.markdown("""
    Here we will look at the engagement metrics for the control and treatment groups. We keep things pretty high level and keep details for other analysis. The differences are calculated against the control group.
    """)

    # # Display comparative metrics
    total_metrics, avg_metrics = display_comparative_metrics(processed_data)

    entropies = shannon_entropy_comparison(processed_data)

    stats_results_replicated = shannon_entropy_comparison_pointplot(processed_data)



    st.subheader("Actions and Comments per Round")
    st.markdown("""
    We just show the average and total number of actions and comments per round for both the control and treatment groups.
    """)
    # Get the figures
    [avg_actions_fig, avg_comments_fig], [total_actions_fig, total_comments_fig] = create_metric_comparison_plots(total_metrics, avg_metrics)
    
    # Display average metrics
    st.subheader("Average Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(avg_actions_fig, use_container_width=True)
    with col2:
        st.plotly_chart(avg_comments_fig, use_container_width=True)
    
    st.divider()

    # After displaying metrics
    overall_fig, round_figs = create_engagement_plots(processed_data)
    
    # Process and create time series visualization
    time_series_data, out_of_window = process_time_series_data(processed_data)
    metric_figs = create_time_series_plots(time_series_data, out_of_window)
    
    # Display time series plots
    st.subheader("Time Series Engagement")
    metrics = ['Actions', 'Comments', 'Total Engagement']
    for metric, fig in zip(metrics, metric_figs):
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    comment_length_fig = create_comment_length_plot(processed_data)
    st.subheader("Comment Length Distribution")
    st.plotly_chart(comment_length_fig, use_container_width=True)

    likes_distribution_fig = create_likes_distribution_plot(processed_data)
    st.subheader("Likes Distribution")
    st.plotly_chart(likes_distribution_fig, use_container_width=True)
    
    st.divider()

    st.subheader("Sentiment Distribution")
    sentiment_fig = create_sentiment_plot(processed_data)
    st.plotly_chart(sentiment_fig, use_container_width=True)

    st.subheader("Toxicity Distribution")
    toxicity_fig = create_toxicity_plot(processed_data)
    st.plotly_chart(toxicity_fig, use_container_width=True)

if __name__ == "__main__":
    main()