import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from transformers import pipeline
import streamlit as st
import numpy as np
from engagement_helpers.visuals import calculate_bootstrap_ci

@st.cache_resource
def load_sentiment_pipeline():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    sentiment_task = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)
    return sentiment_task

@st.cache_data
def calculate_ci_stats(df_generations_matched):
    """
    Calculate means and bootstrap CIs for each topic-group combination
    """
    plot_data = []
    np.random.seed(42)  # For reproducibility
    
    # Process all groups
    for condition, condition_data in df_generations_matched.items():
        if not isinstance(condition_data, dict):
            continue
            
        for game_id, game in condition_data.items():
            # Get topic order mapping for this game
            if isinstance(game['topic_order'], dict):
                topic_order = game['topic_order']
            else:
                topic_order = {topic: i for i, topic in enumerate(game['topic_order'])}
            round_to_topic = {round_num: topic for topic, round_num in topic_order.items()}
            
            for round_num, round_comments in enumerate(game['comments']):
                topic = round_to_topic[round_num]
                for comment in round_comments:
                    if condition == 'suggestions_5':
                        used_generation = 'generation_info' in comment
                        group = f"{condition} - {'With' if used_generation else 'Without'} Generation"
                    else:
                        group = condition
                    
                    plot_data.append({
                        'group': group,
                        'topic': topic.capitalize(),
                        'words': len(comment['content'].split())
                    })
    
    # Create DataFrame
    df = pd.DataFrame(plot_data)
    
    # Calculate statistics with bootstrap CI
    stats = []
    n_bootstrap = 10000
    
    for topic in sorted(df['topic'].unique()):
        topic_df = df[df['topic'] == topic]
        
        for group in sorted(df['group'].unique()):
            group_data = topic_df[topic_df['group'] == group]['words']
            
            if not group_data.empty:
                # Bootstrap confidence intervals
                bootstrap_samples = np.random.choice(group_data, 
                                                   size=(n_bootstrap, len(group_data)), 
                                                   replace=True)
                bootstrap_means = np.mean(bootstrap_samples, axis=1)
                ci_lower, ci_upper = np.percentile(bootstrap_means, [2.5, 97.5])
                
                stats.append({
                    'topic': topic,
                    'group': group,
                    'mean': group_data.mean(),
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'n': len(group_data)
                })
    
    return pd.DataFrame(stats)

def create_generation_length_comparison_plot(df_generations_matched):
    """
    Create mean plots with confidence intervals comparing word counts.
    """
    # Get cached statistics
    stats_df = calculate_ci_stats(df_generations_matched)
    
    # Create figure
    fig = go.Figure()
    
    # Colors for topics
    topic_colors = {
        'Cats': '#2ecc71',    # Green
        'Politics': '#e74c3c', # Red
        'Oats': '#3498db'     # Blue
    }
    
    # Add points and error bars for each topic and treatment combination
    for topic in sorted(stats_df['topic'].unique()):
        topic_df = stats_df[stats_df['topic'] == topic]
        
        for group in sorted(stats_df['group'].unique()):
            group_data = topic_df[topic_df['group'] == group]
            
            if not group_data.empty:
                fig.add_trace(go.Scatter(
                    x=[f"{topic} - {group}"],
                    y=[group_data['mean'].iloc[0]],
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array=[group_data['ci_upper'].iloc[0] - group_data['mean'].iloc[0]],
                        arrayminus=[group_data['mean'].iloc[0] - group_data['ci_lower'].iloc[0]],
                        visible=True
                    ),
                    mode='markers',
                    marker=dict(
                        size=12,
                        color=topic_colors[topic],
                        line=dict(
                            width=3 if 'suggestions_5' in group and 'With Generation' in group else 1,
                            color='black'
                        )
                    ),
                    name=f"{topic} - {group}",
                    hovertemplate=(
                        "<b>Topic-Group:</b> %{x}<br>" +
                        "<b>Mean Words:</b> %{y:.1f}<br>" +
                        "<b>95% CI:</b> [%{customdata[0]:.1f}, %{customdata[1]:.1f}]<br>" +
                        "<b>n:</b> %{customdata[2]}<br>" +
                        "<extra></extra>"
                    ),
                    customdata=[[
                        group_data['ci_lower'].iloc[0],
                        group_data['ci_upper'].iloc[0],
                        group_data['n'].iloc[0]
                    ]]
                ))
    
    # Update layout
    fig.update_layout(
        title='Mean Comment Word Counts by Topic and Treatment (with 95% Bootstrap CI)',
        yaxis_title="Number of Words",
        xaxis_title="Topic - Treatment Group",
        height=600,
        showlegend=False,
        xaxis=dict(tickangle=45),
        margin=dict(b=150, r=150)  # Increased bottom margin for rotated labels
    )
    
    
    return fig


def create_generation_likes_distribution_plot(df_generations_matched):
    """
    Create plots comparing the distribution of like types between different treatments.
    """
    # Prepare data for plotting
    plot_data = {
        topic: {} for topic in ['Cats', 'Politics', 'Oats']
    }
    
    # Create a mapping using composite key (game_id, content_id, round_num)
    content_id_map = {}
    
    # Process all groups
    for condition, condition_data in df_generations_matched.items():
        if not isinstance(condition_data, dict):
            continue
            
        # Initialize condition in plot_data for all topics
        for topic in plot_data:
            if condition == 'suggestions_5':
                plot_data[topic][f'{condition} - With Generation'] = {}
                plot_data[topic][f'{condition} - Without Generation'] = {}
                plot_data[topic][f'{condition} - Seed Content'] = {}
            else:
                plot_data[topic][condition] = {}
                plot_data[topic][f'{condition} - Seed Content'] = {}
        
        # Process all games in each condition
        for game_id, game_data in condition_data.items():
            # Get topic order mapping for this game
            if isinstance(game_data['topic_order'], dict):
                topic_order = game_data['topic_order']
            else:
                topic_order = {topic: i for i, topic in enumerate(game_data['topic_order'])}
            round_to_topic = {round_num: topic for topic, round_num in topic_order.items()}
            
            # Process each round's comments to build the content_id mapping
            for round_num, round_comments in enumerate(game_data['comments']):
                current_topic = round_to_topic[round_num].capitalize()
                for comment in round_comments:
                    content_id = comment['content_id']
                    # Create composite key
                    comment_key = (game_id, content_id, round_num)
                    
                    if condition == 'suggestions_5':
                        used_generation = 'generation_info' in comment
                        group = f'{condition} - With Generation' if used_generation else f'{condition} - Without Generation'
                    else:
                        group = condition
                        
                    content_id_map[comment_key] = {
                        'group': group,
                        'topic': current_topic,
                        'condition': condition
                    }
            
            # Process each round's actions
            for round_num, round_actions in enumerate(game_data['actions']):
                current_topic = round_to_topic[round_num].capitalize()
                for action in round_actions:
                    if 'like_type' in action and 'content_id' in action:
                        content_id = action['content_id']
                        like_type = action['like_type']
                        
                        # Look up using composite key
                        comment_key = (game_id, content_id, round_num)
                        if comment_key in content_id_map:
                            comment_info = content_id_map[comment_key]
                            topic = comment_info['topic']
                            group = comment_info['group']
                        else:
                            # If not in mapping, it's seed content
                            topic = current_topic
                            group = f"{condition} - Seed Content"
                        
                        # Initialize like_type counter if needed
                        if like_type not in plot_data[topic][group]:
                            plot_data[topic][group][like_type] = 0
                        
                        # Increment the counter
                        plot_data[topic][group][like_type] += 1
    
    # Calculate proportions and prepare data for plotting
    plot_df = []
    for topic in plot_data:
        for group in plot_data[topic]:
            total = sum(plot_data[topic][group].values())
            for like_type, count in plot_data[topic][group].items():
                proportion = count / total if total > 0 else 0
                plot_df.append({
                    'topic': topic,
                    'group': group,
                    'like_type': like_type,
                    'count': count,
                    'proportion': proportion
                })
    
    # Create DataFrame
    df = pd.DataFrame(plot_df)
    
    # Create figure with six subplots (2 rows x 3 columns)
    fig = make_subplots(
        rows=2, 
        cols=3,
        subplot_titles=(
            'Cats - Raw Count', 'Politics - Raw Count', 'Oats - Raw Count',
            'Cats - Proportion', 'Politics - Proportion', 'Oats - Proportion'
        )
    )
    
    # Define subplot positions for each topic
    topic_positions = {
        'Cats': {'count': (1, 1), 'prop': (2, 1)},
        'Politics': {'count': (1, 2), 'prop': (2, 2)},
        'Oats': {'count': (1, 3), 'prop': (2, 3)}
    }

    # Generate colors for all groups dynamically
    unique_groups = sorted(df['group'].unique())
    colors = {}
    for i, group in enumerate(unique_groups):
        # Use a color palette that scales with the number of groups
        hue = i / len(unique_groups)
        colors[group] = f'hsl({hue * 360}, 70%, 50%)'
    
    # Add traces for each topic
    for topic in ['Cats', 'Politics', 'Oats']:
        topic_data = df[df['topic'] == topic]
        
        for group in unique_groups:
            group_data = topic_data[topic_data['group'] == group]
            
            if not group_data.empty:
                # Add raw count distribution
                fig.add_trace(
                    go.Bar(
                        x=group_data['like_type'],
                        y=group_data['count'],
                        name=group,
                        marker_color=colors[group],
                        showlegend=(topic == 'Cats')  # Show legend only for first topic
                    ),
                    row=topic_positions[topic]['count'][0],
                    col=topic_positions[topic]['count'][1]
                )
                
                # Add proportion distribution
                fig.add_trace(
                    go.Bar(
                        x=group_data['like_type'],
                        y=group_data['proportion'],
                        name=group,
                        marker_color=colors[group],
                        showlegend=False
                    ),
                    row=topic_positions[topic]['prop'][0],
                    col=topic_positions[topic]['prop'][1]
                )
    
    # Update layout
    fig.update_layout(
        # height=800,
        # width=1200,
        barmode='group',
        showlegend=True,
        legend_title_text='Treatment Type'
    )
    
    # Update specific axes
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Proportion", row=2, col=1)
    
    # Set fixed range for proportion plots (bottom row)
    fig.update_yaxes(range=[0, 1], row=2, col=1)
    fig.update_yaxes(range=[0, 1], row=2, col=2)
    fig.update_yaxes(range=[0, 1], row=2, col=3)
    
    return fig


@st.cache_data
def analyze_sentiment_batch(texts):
    """Analyze sentiment for a batch of texts"""
    sentiment_task = load_sentiment_pipeline()
    try:
        results = sentiment_task(texts, return_all_scores=True)
        return results
    except Exception as e:
        st.error(f"Error in batch sentiment analysis: {str(e)}")
        return None

def create_generation_sentiment_comparison_plot(df_generations_matched):
    """
    Create error bar plots comparing sentiment scores between different treatments,
    showing mean and confidence intervals for each group.
    """
    # Colors for consistency
    colors = {
        'control': '#1f77b4',  # blue
        'chat_5': '#2ca02c',   # green
        'conversation_5': '#ff7f0e',  # orange
        'feedback_5': '#d62728',  # red
        'suggestions_5': '#9467bd'  # purple
    }
    
    # Prepare data for plotting
    plot_data = []
    total_comments = 0
    
    # Count total comments first
    for condition, condition_data in df_generations_matched.items():
        if not isinstance(condition_data, dict):
            continue
        for game in condition_data.values():
            for round_comments in game['comments']:
                total_comments += len(round_comments)
    
    # Process data with progress tracking
    # with st.spinner("Analyzing sentiment, hang tight..."):
    progress_bar = st.progress(0)
    status_placeholder = st.empty()
    comments_processed = 0
    
    for condition, condition_data in df_generations_matched.items():
        if not isinstance(condition_data, dict):
            continue
            
        for game_id, game in condition_data.items():
            topic_order = {topic: i for i, topic in enumerate(game['topic_order'])} if isinstance(game['topic_order'], list) else game['topic_order']
            round_to_topic = {round_num: topic for topic, round_num in topic_order.items()}
            
            # Collect all comments for batch processing
            comments_batch = []
            comment_metadata = []
            
            for round_num, round_comments in enumerate(game['comments']):
                topic = round_to_topic[round_num]
                for comment in round_comments:
                    if condition == 'suggestions_5':
                        group = f"{condition} - {'With' if 'generation_info' in comment else 'Without'} Generation"
                    else:
                        group = condition
                        
                    comments_batch.append(comment['content'])
                    comment_metadata.append({
                        'group': group,
                        'topic': topic.capitalize()
                    })
            
            # Batch process sentiments
            if comments_batch:
                results = analyze_sentiment_batch(comments_batch)
                if results:
                    for i, result in enumerate(results):
                        for score_dict in result:
                            plot_data.append({
                                'group': comment_metadata[i]['group'],
                                'topic': comment_metadata[i]['topic'],
                                'sentiment_type': score_dict['label'].lower(),
                                'score': score_dict['score']
                            })
                
                comments_processed += len(comments_batch)
                progress = min(1.0, comments_processed / total_comments)
                progress_bar.progress(progress)
                status_placeholder.write(f"Processed {comments_processed}/{total_comments} comments...")
    
    status_placeholder.empty()
    st.success('Sentiment analysis completed!')
    
    # Create DataFrame
    df = pd.DataFrame(plot_data)
    
    # Create figure with subplots
    topics = sorted(df['topic'].unique())
    fig = make_subplots(
        rows=len(topics), cols=3,
        subplot_titles=[f"{topic} - {sentiment.capitalize()} Sentiment" 
                       for topic in topics 
                       for sentiment in ['negative', 'neutral', 'positive']],
        vertical_spacing=0.2,
        horizontal_spacing=0.1
    )
    
    # Generate colors for groups
    unique_groups = sorted(df['group'].unique())
    
    # Calculate CIs and plot
    sentiment_types = ['negative', 'neutral', 'positive']
    
    for row, topic in enumerate(topics, 1):
        topic_df = df[df['topic'] == topic]
        
        for col, sentiment_type in enumerate(sentiment_types, 1):
            sentiment_df = topic_df[topic_df['sentiment_type'] == sentiment_type]
            
            for group in unique_groups:
                group_data = sentiment_df[sentiment_df['group'] == group]
                
                if not group_data.empty:
                    mean, ci_lower, ci_upper = calculate_bootstrap_ci(group_data['score'])
                    x_pos = unique_groups.index(group) + 1
                    
                    fig.add_trace(
                        go.Scatter(
                            x=[x_pos],
                            y=[mean],
                            error_y=dict(
                                type='data',
                                array=[ci_upper - mean],
                                arrayminus=[mean - ci_lower],
                                visible=True
                            ),
                            name=group,
                            mode='markers',
                            marker=dict(
                                color=colors.get(group.split(' - ')[0], '#000000'),
                                size=10
                            ),
                            showlegend=(row == 1 and col == 1)
                        ),
                        row=row, col=col
                    )
    
    # Update layout
    fig.update_layout(
        title='Mean Sentiment Scores by Topic and Treatment (with 95% CI)',
        height=300 * len(topics),
        showlegend=True,
        margin=dict(l=50, r=150, t=100, b=50)
    )
    
    # Update axes
    for row in range(1, len(topics) + 1):
        for col in range(1, 4):
            fig.update_xaxes(
                showticklabels=False,
                range=[0, len(unique_groups) + 1],
                row=row, col=col
            )
            fig.update_yaxes(
                title_text="Score" if col == 1 else None,
                range=[0, 1],
                row=row, col=col
            )
    
    return fig