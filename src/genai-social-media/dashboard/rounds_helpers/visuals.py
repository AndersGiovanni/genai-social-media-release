import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# Helper function to check if timestamp is within bucket (with buffer)
def is_in_bucket(timestamp, start_time, buffer_seconds=30):
    """Helper function to check if timestamp is within bucket (with buffer)"""
    buffer = pd.Timedelta(seconds=buffer_seconds)
    start_with_buffer = start_time - buffer
    end_with_buffer = start_time + pd.Timedelta(minutes=10) + buffer
    return start_with_buffer <= timestamp <= end_with_buffer

# Helper function to process items and check bucket validity
def process_items_for_bucket(items, start_time, item_type):
    valid_items = []
    invalid_items = []
    
    for item in items:
        timestamp = pd.to_datetime(item['timestamp'])
        if is_in_bucket(timestamp, start_time):
            valid_items.append(item)
        else:
            invalid_items.append({
                'timestamp': timestamp,
                'expected_bucket': start_time,
                'item': item
            })
    
    return valid_items, invalid_items

# Helper function to find the best bucket for a set of items
def find_best_bucket(items, round_starts):
    bucket_counts = [0] * len(round_starts)
    
    for item in items:
        timestamp = pd.to_datetime(item['timestamp'])
        for i, start_time in enumerate(round_starts):
            if is_in_bucket(timestamp, start_time):
                bucket_counts[i] += 1
    
    # If any bucket has more than 70% of the items, choose that bucket
    total_items = len(items)
    for i, count in enumerate(bucket_counts):
        if total_items > 0 and (count / total_items) > 0.7:
            return i
            
    # If no clear winner, return the bucket with the most items
    return bucket_counts.index(max(bucket_counts))

def create_timeline_plots(df_selected):
    # Get all round start times
    round_starts = [
        df_selected[f'round_{i}_startTime'].iloc[0]
        for i in range(3)
    ]
    
    # First, determine which bucket each set of actions/comments belongs to
    all_actions = []
    all_comments = []
    for i in range(3):
        all_actions.append(df_selected[f'actions-{i}'].iloc[0])
        all_comments.append(df_selected[f'comments-{i}'].iloc[0])
    
    # Reassign actions and comments to correct buckets
    correct_bucket_actions = [[] for _ in range(3)]
    correct_bucket_comments = [[] for _ in range(3)]
    
    for actions in all_actions:
        if actions:  # Check if not empty
            best_bucket = find_best_bucket(actions, round_starts)
            correct_bucket_actions[best_bucket].extend(actions)
    
    for comments in all_comments:
        if comments:  # Check if not empty
            best_bucket = find_best_bucket(comments, round_starts)
            correct_bucket_comments[best_bucket].extend(comments)
    
    # Create total counts line charts ONCE before the round loop
    col1, col2 = st.columns(2)
    
    with col1:
        actions_fig = go.Figure()
        actions_counts = [len(actions) for actions in correct_bucket_actions]
        
        actions_fig.add_trace(go.Line(
            x=['Round 0', 'Round 1', 'Round 2'],
            y=actions_counts,
            marker_color='blue'
        ))
        
        actions_fig.update_layout(
            title='Total Actions per Round',
            yaxis_title='Number of Actions',
            height=300,
            showlegend=False,
            yaxis=dict(
                # range=[0, 40],
                # tickvals=[0, 10, 20, 30, 40],
                # ticktext=['0', '10', '20', '30', '40'],
            )
        )
        
        st.plotly_chart(actions_fig, use_container_width=True)
        
    with col2:
        comments_fig = go.Figure()
        comments_counts = [len(comments) for comments in correct_bucket_comments]
        
        comments_fig.add_trace(go.Line(
            x=['Round 0', 'Round 1', 'Round 2'],
            y=comments_counts,
            marker_color='green'
        ))
        
        comments_fig.update_layout(
            title='Total Comments per Round',
            yaxis_title='Number of Comments',
            height=300,
            showlegend=False,
            yaxis=dict(
                # range=[0, 40],
                # tickvals=[0, 10, 20, 30, 40],
                # ticktext=['0', '10', '20', '30', '40'],
            )
        )
        
        st.plotly_chart(comments_fig, use_container_width=True)
    
    # Create plots for each round
    figs = []
    all_invalid_items = []
    
    for round_num in range(3):
        round_start = round_starts[round_num]
        
        # Process actions and comments for this round
        valid_actions, invalid_actions = process_items_for_bucket(
            correct_bucket_actions[round_num], 
            round_start, 
            'action'
        )
        valid_comments, invalid_comments = process_items_for_bucket(
            correct_bucket_comments[round_num], 
            round_start, 
            'comment'
        )
        
        all_invalid_items.extend(invalid_actions)
        all_invalid_items.extend(invalid_comments)
        
        # Create figure
        fig = go.Figure()
        
        # Get unique user_ids
        user_ids = set(item['user_id'] for item in valid_actions + valid_comments)
        
        # Add traces for actions and comments
        for user_id in user_ids:
            user_actions = [a for a in valid_actions if a['user_id'] == user_id]
            user_comments = [c for c in valid_comments if c['user_id'] == user_id]
            
            # Convert timestamps to minutes from round start
            if user_actions:
                action_times = [(pd.to_datetime(a['timestamp']) - round_start).total_seconds() / 60 for a in user_actions]
                fig.add_trace(go.Scatter(
                    x=action_times,
                    y=[user_id] * len(action_times),
                    mode='markers',
                    name=f'Actions ({user_id})',
                    marker_symbol='triangle-up',
                    marker_size=12,
                    hovertemplate=(
                        "<b>Action</b><br>" +
                        "Type: %{customdata[0]}<br>" +
                        "Content ID: %{customdata[1]}<br>" +
                        "Time: %{customdata[2]}<br>" +
                        "User ID: %{customdata[3]}<extra></extra>"
                    ),
                    customdata=[[a['like_type'], a['content_id'], a['timestamp'], a['user_id']] for a in user_actions],
                    showlegend=True if user_id == list(user_ids)[0] else False
                ))
            
            if user_comments:
                comment_times = [(pd.to_datetime(c['timestamp']) - round_start).total_seconds() / 60 for c in user_comments]
                fig.add_trace(go.Scatter(
                    x=comment_times,
                    y=[user_id] * len(comment_times),
                    mode='markers',
                    name=f'Comments ({user_id})',
                    marker_symbol='circle',
                    marker_size=12,
                    hovertemplate=(
                        "<b>Comment</b><br>" +
                        "Content ID: %{customdata[0]}<br>" +
                        "Parent ID: %{customdata[1]}<br>" +
                        "Content: %{customdata[2]}<br>" +
                        "Time: %{customdata[3]}<br>" +
                        "User ID: %{customdata[4]}<extra></extra>"
                    ),
                    customdata=[[
                        c['content_id'],
                        c.get('parent_content_id', 'None'),
                        c['content'],
                        c['timestamp'],
                        c['user_id']
                    ] for c in user_comments],
                    showlegend=True if user_id == list(user_ids)[0] else False
                ))
        
        # Update layout
        fig.update_layout(
            title=f'Round {round_num} Timeline (Start: {round_start.strftime("%Y-%m-%d %H:%M:%S")})',
            xaxis_title='Minutes from Round Start',
            yaxis_title='User ID',
            xaxis=dict(
                range=[-0.5, 10.5],
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.2)',
                zeroline=True,
                zerolinecolor='rgba(0, 0, 0, 0.2)',
            ),
            height=400,
            showlegend=True,
            hovermode='closest',
            hoverlabel=dict(
                bgcolor="rgba(50, 50, 50, 0.9)",  # Dark semi-transparent background
                font=dict(
                    family="monospace",
                    size=14,
                    color="white"
                ),
                bordercolor="rgba(255, 255, 255, 0.3)",  # Light border
                align="left"
            )
        )
        figs.append(fig)
    
    # Display warnings for invalid items
    if all_invalid_items:
        st.warning(f"⚠️ Some items were found outside their expected time buckets: {len(all_invalid_items)}")
        for item in all_invalid_items:
            st.write(f"Item timestamp: {item['timestamp']}")
            st.write(f"Expected bucket start: {item['expected_bucket']}")
            st.write(f"Item details: {item['item']}")
            st.write("---")
    
    # Display plots
    for fig in figs:
        st.plotly_chart(fig, use_container_width=True)