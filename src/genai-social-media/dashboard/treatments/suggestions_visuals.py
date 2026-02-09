from collections import defaultdict
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import numpy as np
from transformers import pipeline
from scipy.stats import chi2_contingency
from engagement_helpers.visuals import calculate_bootstrap_ci, analyze_sentiment_batch


@st.cache_resource
def load_sentiment_pipeline():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    sentiment_task = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)
    return sentiment_task


def create_generation_length_comparison_plot(df_generations_matched):
    """
    Create plots comparing comment lengths with bootstrapped confidence intervals.
    Only processes treatment group data.
    
    Args:
        df_generations_matched (dict): Dictionary containing processed control and treatment data
        
    Returns:
        plotly.graph_objects.Figure: Figure with bootstrapped CIs for word and character counts
    """
    # Prepare data for plotting
    plot_data = {'words': [], 'characters': []}
    
    # Only process treatment group
    treatment_data = df_generations_matched.get('suggestions_5', {})
    
    # Collect all comments across all games and rounds
    for game in treatment_data.values():
        for round_comments in game['comments']:
            for comment in round_comments:
                # Determine if comment used generation
                used_generation = 'generation_info' in comment
                group = 'With Generation' if used_generation else 'Without Generation'
                
                plot_data['words'].append({
                    'group': group,
                    'value': len(comment['content'].split())
                })
                plot_data['characters'].append({
                    'group': group,
                    'value': len(comment['content'])
                })
    
    # Create figure with two subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Word Count (with 95% Bootstrap CI)', 
                       'Character Count (with 95% Bootstrap CI)')
    )
    
    # Colors for consistency
    colors = {
        'With Generation': '#2ecc71',     # Green
        'Without Generation': '#e74c3c'    # Red
    }
    
    # Define offsets for each group
    group_offsets = {
        'With Generation': -0.1,
        'Without Generation': 0.1
    }
    
    # Calculate CIs for each metric and group
    for col_idx, metric in enumerate(['words', 'characters'], 1):
        for group in ['With Generation', 'Without Generation']:
            # Get data for this group
            group_data = pd.Series([
                item['value'] for item in plot_data[metric] 
                if item['group'] == group
            ])
            
            if not group_data.empty:
                # Calculate bootstrap CI
                mean, ci_lower, ci_upper = calculate_bootstrap_ci(group_data)
                
                # Add trace
                fig.add_trace(
                    go.Scatter(
                        x=[0 + group_offsets[group]],
                        y=[mean],
                        error_y=dict(
                            type='data',
                            symmetric=False,
                            array=[ci_upper - mean],
                            arrayminus=[mean - ci_lower],
                            width=10,
                        ),
                        mode='markers',
                        marker=dict(size=10, color=colors[group]),
                        name=group,
                        showlegend=(col_idx == 1)  # Only show legend for first column
                    ),
                    row=1, col=col_idx
                )
    
    # Update layout
    fig.update_layout(
        title='Comment Length Analysis: Generation vs No Generation (with 95% Bootstrap CI)',
        height=500,
        template='plotly_white',
        showlegend=True,
        legend_title_text='Comment Type'
    )
    
    # Update axes
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=2)
    fig.update_yaxes(title_text="Number of Words", row=1, col=1)
    fig.update_yaxes(title_text="Number of Characters", row=1, col=2)
    
    # Add sample size annotation
    with_gen_count = len([x for x in plot_data['words'] if x['group'] == 'With Generation'])
    without_gen_count = len([x for x in plot_data['words'] if x['group'] == 'Without Generation'])
    
    fig.add_annotation(
        text=f"Sample sizes:<br>With Generation: {with_gen_count}<br>Without Generation: {without_gen_count}<br>Total: {with_gen_count + without_gen_count}",
        xref="paper", yref="paper",
        x=1.02, y=1,
        showarrow=False,
        font=dict(size=10),
        align="left"
    )
    
    return fig

def create_generation_likes_distribution_plot(df_generations_matched):
    """
    Create plots comparing the distribution of like types between comments with and without generations.
    Includes bootstrap confidence intervals for proportions.
    """
    # Prepare data for plotting
    plot_data = {
        'With Generation': {},
        'Without Generation': {},
        'Seed Content': {}  # Added new category
    }
    
    # Only process treatment group
    treatment_data = df_generations_matched.get('suggestions_5', {})
    
    # Create a mapping of content_ids to generation status
    content_id_map = {}
    
    # Process all games in treatment
    for game_data in treatment_data.values():
        # Process each round's comments
        for round_comments in game_data['comments']:
            for comment in round_comments:
                content_id = comment['content_id']
                used_generation = 'generation_info' in comment
                content_id_map[content_id] = used_generation
        
        # Process each round's actions
        for round_actions in game_data['actions']:
            for action in round_actions:
                if 'like_type' in action and 'content_id' in action:
                    content_id = action['content_id']
                    like_type = action['like_type']
                    
                    if content_id in content_id_map:
                        group = 'With Generation' if content_id_map[content_id] else 'Without Generation'
                    else:
                        group = 'Seed Content'
                        
                    if like_type not in plot_data[group]:
                        plot_data[group][like_type] = 0
                    plot_data[group][like_type] += 1
    
    # Calculate proportions and prepare data for plotting
    plot_df = []
    for group in ['With Generation', 'Without Generation', 'Seed Content']:
        total = sum(plot_data[group].values())
        for like_type, count in plot_data[group].items():
            proportion = count / total if total > 0 else 0
            plot_df.append({
                'group': group,
                'like_type': like_type,
                'count': count,
                'proportion': proportion
            })
    
    # Create DataFrame
    df = pd.DataFrame(plot_df)
    
    # Prepare data for bootstrapping
    bootstrap_data = {
        'With Generation': [],
        'Without Generation': [],
        'Seed Content': []
    }
    
    # Process all games in treatment
    for game_data in treatment_data.values():
        # Process each round's actions
        for round_actions in game_data['actions']:
            for action in round_actions:
                if 'like_type' in action and 'content_id' in action:
                    content_id = action['content_id']
                    like_type = action['like_type']
                    
                    if content_id in content_id_map:
                        group = 'With Generation' if content_id_map[content_id] else 'Without Generation'
                    else:
                        group = 'Seed Content'
                    
                    bootstrap_data[group].append(like_type)
    
    # Function to calculate proportions for a sample
    def calculate_proportions(sample):
        total = len(sample)
        if total == 0:
            return {}
        counts = defaultdict(int)
        for item in sample:
            counts[item] += 1
        return {k: v/total for k, v in counts.items()}
    
    # Perform bootstrap
    n_bootstrap = 1000
    bootstrap_results = {
        group: {
            'samples': [],
            'ci_lower': {},
            'ci_upper': {}
        } for group in bootstrap_data.keys()
    }
    
    for group, actions in bootstrap_data.items():
        if actions:  # Only bootstrap if we have data
            for _ in range(n_bootstrap):
                # Resample with replacement
                bootstrap_sample = np.random.choice(actions, size=len(actions), replace=True)
                proportions = calculate_proportions(bootstrap_sample)
                bootstrap_results[group]['samples'].append(proportions)
            
            # Calculate confidence intervals for each like type
            all_like_types = set().union(*bootstrap_results[group]['samples'])
            for like_type in all_like_types:
                values = [sample.get(like_type, 0) for sample in bootstrap_results[group]['samples']]
                bootstrap_results[group]['ci_lower'][like_type] = np.percentile(values, 2.5)
                bootstrap_results[group]['ci_upper'][like_type] = np.percentile(values, 97.5)
    
    # Create figure with two subplots
    fig = make_subplots(
        rows=1, 
        cols=2,
        subplot_titles=('Raw Count Distribution', 'Proportion Distribution with 95% CI')
    )
    
    # Colors for consistency
    colors = {
        'With Generation': '#2ecc71',     # Green
        'Without Generation': '#e74c3c',   # Red
        'Seed Content': '#3498db'          # Blue
    }
    
    # Add raw count distribution (keep as bars)
    for group in ['With Generation', 'Without Generation', 'Seed Content']:
        group_data = df[df['group'] == group]
        
        if not group_data.empty:
            fig.add_trace(
                go.Bar(
                    x=group_data['like_type'],
                    y=group_data['count'],
                    name=group,
                    marker_color=colors[group],
                    showlegend=True
                ),
                row=1, col=1
            )
    
    # Define offsets for each group to prevent overlap
    group_offsets = {
        'With Generation': -0.2,
        'Without Generation': 0,
        'Seed Content': 0.2
    }
    
    # Add proportion distribution with confidence intervals as points with error bars
    for group in ['With Generation', 'Without Generation', 'Seed Content']:
        group_data = df[df['group'] == group]
        
        if not group_data.empty:
            # Add scatter plot with error bars
            fig.add_trace(
                go.Scatter(
                    x=[x + group_offsets[group] for x in range(len(group_data['like_type']))],
                    y=group_data['proportion'],
                    name=group,
                    mode='markers',
                    marker=dict(
                        color=colors[group],
                        size=10
                    ),
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array=[
                            bootstrap_results[group]['ci_upper'].get(like_type, 0) - prop
                            for like_type, prop in zip(group_data['like_type'], group_data['proportion'])
                        ],
                        arrayminus=[
                            prop - bootstrap_results[group]['ci_lower'].get(like_type, 0)
                            for like_type, prop in zip(group_data['like_type'], group_data['proportion'])
                        ],
                        width=8,
                        thickness=1.5
                    ),
                    showlegend=False
                ),
                row=1, col=2
            )
    
    # Update layout
    fig.update_layout(
        title='Distribution of Like Types: Comments With vs Without Generation vs Seed Content',
        height=500,
        barmode='group',
        showlegend=True,
        legend_title_text='Comment Type'
    )
    
    # Update axes
    fig.update_xaxes(title_text="Like Type", row=1, col=1)
    fig.update_xaxes(
        title_text="Like Type",
        ticktext=list(group_data['like_type']),
        tickvals=list(range(len(group_data['like_type']))),
        row=1, col=2
    )
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Proportion", row=1, col=2)
    
    # Add sample size annotation
    total_with_gen = sum(plot_data['With Generation'].values())
    total_without_gen = sum(plot_data['Without Generation'].values())
    total_seed = sum(plot_data['Seed Content'].values())
    total_actions = total_with_gen + total_without_gen + total_seed
    
    fig.add_annotation(
        text=f"Total likes:<br>With Generation: {total_with_gen}<br>Without Generation: {total_without_gen}<br>Seed Content: {total_seed}<br>Total: {total_actions}",
        xref="paper", yref="paper",
        x=1.02, y=1,
        showarrow=False,
        font=dict(size=10),
        align="left"
    )
    
    return fig

def create_generation_type_comparison_plot(df_generations_matched):
    treatment_data = df_generations_matched.get('suggestions_5', {})
    
    # Initialize data structures for each round
    round_data = {
        'positive': [[] for _ in range(3)],
        'neutral': [[] for _ in range(3)],
        'negative': [[] for _ in range(3)],
        'without_generation': [[] for _ in range(3)]
    }
    
    # Process each game
    for game_data in treatment_data.values():
        for round_idx, round_comments in enumerate(game_data['comments']):
            round_counts = {
                'positive': 0, 'neutral': 0, 'negative': 0, 'without_generation': 0
            }
            
            for comment in round_comments:
                if 'generation_info' in comment:
                    gen_type = comment['generation_info']['selection']['type']
                    round_counts[gen_type] += 1
                else:
                    round_counts['without_generation'] += 1
            
            for key in round_counts:
                round_data[key][round_idx].append(round_counts[key])
    
    # Calculate statistics
    stats = {
        'averages': {key: [] for key in round_data},
        'stds': {key: [] for key in round_data},
        'totals': {key: [] for key in round_data}
    }
    
    for key in round_data:
        for round_counts in round_data[key]:
            stats['averages'][key].append(np.mean(round_counts))
            stats['stds'][key].append(np.std(round_counts))
            stats['totals'][key].append(sum(round_counts))
    
    # Calculate combined generation stats
    combined_averages = []
    combined_stds = []
    combined_totals = []
    
    for round_idx in range(3):
        round_sums = []
        for game_idx in range(len(round_data['positive'][round_idx])):
            round_sum = (round_data['positive'][round_idx][game_idx] +
                        round_data['neutral'][round_idx][game_idx] +
                        round_data['negative'][round_idx][game_idx])
            round_sums.append(round_sum)
        combined_averages.append(np.mean(round_sums))
        combined_stds.append(np.std(round_sums))
        combined_totals.append(sum(round_sums))

    # Create figure
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Average Comments per Round', 'Total Comments per Round')
    )
    
    colors = {
        'positive': '#2ecc71',    # Green
        'neutral': '#f1c40f',     # Yellow
        'negative': '#e74c3c',    # Red
        'without_generation': '#95a5a6',  # Gray
        'all_generations': '#3498db'      # Blue
    }
    
    x_vals = [0, 1, 2]
    
    # Function to add traces for a specific view
    def add_view_traces(view_type='combined'):
        if view_type == 'combined':
            # Add combined generation trace
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=combined_averages,
                    name='With Generation',
                    mode='lines+markers',
                    line=dict(color=colors['all_generations']),
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # Add combined error band
            fig.add_trace(
                go.Scatter(
                    x=x_vals + x_vals[::-1],
                    y=(np.array(combined_averages) + np.array(combined_stds)).tolist() + 
                      (np.array(combined_averages) - np.array(combined_stds)).tolist()[::-1],
                    fill='toself',
                    fillcolor=colors['all_generations'],
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=False,
                    opacity=0.2
                ),
                row=1, col=1
            )
            
            # Add combined total
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=combined_totals,
                    name='With Generation',
                    mode='lines+markers',
                    line=dict(color=colors['all_generations']),
                    showlegend=False
                ),
                row=1, col=2
            )
        else:  # separate view
            for gen_type in ['positive', 'neutral', 'negative']:
                # Add average line
                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=stats['averages'][gen_type],
                        name=f'{gen_type.capitalize()} Generation',
                        mode='lines+markers',
                        line=dict(color=colors[gen_type]),
                        showlegend=True
                    ),
                    row=1, col=1
                )
                
                # Add error band
                fig.add_trace(
                    go.Scatter(
                        x=x_vals + x_vals[::-1],
                        y=(np.array(stats['averages'][gen_type]) + np.array(stats['stds'][gen_type])).tolist() + 
                          (np.array(stats['averages'][gen_type]) - np.array(stats['stds'][gen_type])).tolist()[::-1],
                        fill='toself',
                        fillcolor=colors[gen_type],
                        line=dict(color='rgba(255,255,255,0)'),
                        showlegend=False,
                        opacity=0.2
                    ),
                    row=1, col=1
                )
                
                # Add total line
                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=stats['totals'][gen_type],
                        name=f'{gen_type.capitalize()} Generation',
                        mode='lines+markers',
                        line=dict(color=colors[gen_type]),
                        showlegend=False
                    ),
                    row=1, col=2
                )
    
    # Add initial view (combined)
    add_view_traces('combined')
    
    # Always add without generation traces
    # Average
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=stats['averages']['without_generation'],
            name='Without Generation',
            mode='lines+markers',
            line=dict(color=colors['without_generation']),
            showlegend=True
        ),
        row=1, col=1
    )
    
    # Error band
    fig.add_trace(
        go.Scatter(
            x=x_vals + x_vals[::-1],
            y=(np.array(stats['averages']['without_generation']) + np.array(stats['stds']['without_generation']) ).tolist() + 
              (np.array(stats['averages']['without_generation']) - np.array(stats['stds']['without_generation']) ).tolist()[::-1],
            fill='toself',
            fillcolor=colors['without_generation'],
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            opacity=0.2
        ),
        row=1, col=1
    )
    
    # Total
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=stats['totals']['without_generation'],
            name='Without Generation',
            mode='lines+markers',
            line=dict(color=colors['without_generation']),
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Add separate view traces (initially hidden)
    for gen_type in ['positive', 'neutral', 'negative']:
        # Add average line
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=stats['averages'][gen_type],
                name=f'{gen_type.capitalize()} Selection',
                mode='lines+markers',
                line=dict(color=colors[gen_type]),
                showlegend=True,
                visible=False
            ),
            row=1, col=1
        )
        
        # Add error band
        fig.add_trace(
            go.Scatter(
                x=x_vals + x_vals[::-1],
                y=(np.array(stats['averages'][gen_type]) + np.array(stats['stds'][gen_type])).tolist() + 
                  (np.array(stats['averages'][gen_type]) - np.array(stats['stds'][gen_type])).tolist()[::-1],
                fill='toself',
                fillcolor=colors[gen_type],
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                opacity=0.2,
                visible=False
            ),
            row=1, col=1
        )
        
        # Add total line
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=stats['totals'][gen_type],
                name=f'{gen_type.capitalize()} Selection',
                mode='lines+markers',
                line=dict(color=colors[gen_type]),
                showlegend=False,
                visible=False
            ),
            row=1, col=2
        )
    
    # Update layout with buttons
    fig.update_layout(
        title='Comments per Round by Generation Type',
        height=500,
        showlegend=True,
        legend_title_text='Comment Type',
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=0.7,
            y=1.2,
            showactive=True,
            buttons=[
                dict(
                    label="Combined",
                    method="update",
                    args=[{"visible": [True, True, True] +  # Combined view traces
                                    [True, True, True] +    # Without generation traces
                                    [False] * 9}]          # Hide separate view traces
                ),
                dict(
                    label="Separate",
                    method="update",
                    args=[{"visible": [False, False, False] +  # Hide combined view traces
                                    [True, True, True] +       # Without generation traces
                                    [True] * 9}]              # Show separate view traces
                )
            ]
        )]
    )
    
    # Update axes
    for i in range(1, 3):
        fig.update_xaxes(
            title_text="Round",
            ticktext=['Round 0', 'Round 1', 'Round 2'],
            tickvals=[0, 1, 2],
            row=1, col=i
        )
    
    fig.update_yaxes(title_text="Average Number of Comments", row=1, col=1)
    fig.update_yaxes(title_text="Total Number of Comments", row=1, col=2)
    
    return fig

def create_generation_sentiment_plot(df_generations_matched):
    """
    Create plots comparing sentiment between comments with and without generations,
    with options to view combined or separated by generation type.
    """
    # Initialize sentiment pipeline
    sentiment_task = load_sentiment_pipeline()
    
    treatment_data = df_generations_matched.get('suggestions_5', {})
    
    # Initialize data structures for sentiment scores
    sentiment_data = {
        'with_generation': {'negative': [], 'neutral': [], 'positive': []},
        'without_generation': {'negative': [], 'neutral': [], 'positive': []},
        'generation_types': {
            'positive': {'negative': [], 'neutral': [], 'positive': []},
            'neutral': {'negative': [], 'neutral': [], 'positive': []},
            'negative': {'negative': [], 'neutral': [], 'positive': []}
        }
    }
    
    # Collect all comments first and count total
    all_comments = []
    for game_data in treatment_data.values():
        for round_comments in game_data['comments']:
            all_comments.extend(round_comments)
    total_comments = len(all_comments)
    
    # Process comments in batches
    with st.spinner('Analyzing sentiment of comments...'):
        progress_bar = st.progress(0)
        status_placeholder = st.empty()
        comments_processed = 0
        
        # Process in batches
        batch_size = 32
        for i in range(0, len(all_comments), batch_size):
            batch = all_comments[i:i + batch_size]
            texts = [comment['content'] for comment in batch]
            
            # Use caching for sentiment analysis
            results = analyze_sentiment_batch(texts)
            
            # Process results
            for comment, result in zip(batch, results):
                scores = {score['label'].lower(): score['score'] for score in result}
                
                if 'generation_info' in comment:
                    # Store in combined view
                    for sentiment in ['negative', 'neutral', 'positive']:
                        sentiment_data['with_generation'][sentiment].append(scores[sentiment])
                    
                    # Store by generation type
                    gen_type = comment['generation_info']['selection']['type']
                    for sentiment in ['negative', 'neutral', 'positive']:
                        sentiment_data['generation_types'][gen_type][sentiment].append(scores[sentiment])
                else:
                    for sentiment in ['negative', 'neutral', 'positive']:
                        sentiment_data['without_generation'][sentiment].append(scores[sentiment])
            
            comments_processed += len(batch)
            progress = min(1.0, comments_processed / total_comments)
            progress_bar.progress(progress)
            status_placeholder.write(f"Processed {comments_processed}/{total_comments} comments...")
    
    status_placeholder.empty()
    st.success('Sentiment analysis completed!')
    
    # Calculate bootstrap CIs for each group and sentiment
    ci_data = []
    sentiments = ['negative', 'neutral', 'positive']
    
    # Process combined view
    for group in ['with_generation', 'without_generation']:
        for sentiment in sentiments:
            scores = pd.Series(sentiment_data[group][sentiment])
            if not scores.empty:
                mean, ci_lower, ci_upper = calculate_bootstrap_ci(scores)
                ci_data.append({
                    'group': group,
                    'sentiment': sentiment,
                    'mean': mean,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'view': 'combined'
                })
    
    # Process separate view
    for gen_type in ['positive', 'neutral', 'negative']:
        for sentiment in sentiments:
            scores = pd.Series(sentiment_data['generation_types'][gen_type][sentiment])
            if not scores.empty:
                mean, ci_lower, ci_upper = calculate_bootstrap_ci(scores)
                ci_data.append({
                    'group': f'{gen_type}_selection',
                    'sentiment': sentiment,
                    'mean': mean,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'view': 'separate'
                })
    
    # Create figure
    fig = go.Figure()
    
    # Colors for consistency
    colors = {
        'with_generation': '#3498db',     # Blue
        'without_generation': '#95a5a6',   # Gray
        'positive_selection': '#2ecc71',   # Green
        'neutral_selection': '#f1c40f',    # Yellow
        'negative_selection': '#e74c3c'    # Red
    }
    
    # Define offsets for each group
    group_offsets = {
        'with_generation': -0.1,
        'without_generation': 0.1,
        'positive_selection': -0.2,
        'neutral_selection': 0,
        'negative_selection': 0.2
    }
    
    # Add combined view traces (initially visible)
    for group in ['with_generation', 'without_generation']:
        group_data = pd.DataFrame([d for d in ci_data if d['group'] == group and d['view'] == 'combined'])
        
        fig.add_trace(
            go.Scatter(
                x=[sentiments.index(s) + group_offsets[group] for s in group_data['sentiment']],
                y=group_data['mean'],
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=group_data['ci_upper'] - group_data['mean'],
                    arrayminus=group_data['mean'] - group_data['ci_lower'],
                    width=10,
                ),
                mode='markers',
                marker=dict(size=10, color=colors[group]),
                name=group.replace('_', ' ').title(),
                showlegend=True
            )
        )
    
    # Add separate view traces (initially hidden)
    for group in ['positive_selection', 'neutral_selection', 'negative_selection', 'without_generation']:
        group_data = pd.DataFrame([d for d in ci_data if d['group'] == group and d['view'] == 'separate'])
        
        if not group_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=[sentiments.index(s) + group_offsets[group] for s in group_data['sentiment']],
                    y=group_data['mean'],
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array=group_data['ci_upper'] - group_data['mean'],
                        arrayminus=group_data['mean'] - group_data['ci_lower'],
                        width=10,
                    ),
                    mode='markers',
                    marker=dict(size=10, color=colors[group]),
                    name=group.replace('_', ' ').title(),
                    visible=False
                )
            )
    
    # Update layout with buttons
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=0.7,
            y=1.2,
            showactive=True,
            buttons=[
                dict(
                    label="Combined",
                    method="update",
                    args=[{"visible": [True, True] + [False] * 4}]
                ),
                dict(
                    label="Separate",
                    method="update",
                    args=[{"visible": [False, False] + [True] * 4}]
                )
            ]
        )]
    )
    
    # Update layout
    fig.update_layout(
        title='Sentiment Analysis by Generation Type (with 95% Bootstrap CI)',
        height=500,
        showlegend=True,
        legend_title_text='Comment Type',
        margin=dict(t=100),
        xaxis=dict(
            title_text="Sentiment Class",
            ticktext=sentiments,
            tickvals=list(range(len(sentiments)))
        ),
        yaxis=dict(
            title_text="Sentiment Score",
            range=[0, 1]
        )
    )
    
    return fig, sentiment_data

def analyze_reply_patterns(df_generations_matched):
    """
    Analyze replies to comments with and without generations using bootstrap confidence intervals.
    """
    # Initialize sentiment pipeline
    sentiment_task = load_sentiment_pipeline()
    
    # Initialize data structures
    replies_to_generated = {
        'lengths': [],
        'sentiment': {'negative': [], 'neutral': [], 'positive': []}
    }
    replies_to_nongenerated = {
        'lengths': [],
        'sentiment': {'negative': [], 'neutral': [], 'positive': []}
    }
    
    treatment_data = df_generations_matched.get('suggestions_5', {})
    
    # Process each game
    with st.spinner('Analyzing replies...'):
        for game_data in treatment_data.values():
            for round_comments in game_data['comments']:
                comment_map = {comment['content_id']: comment for comment in round_comments}
                
                for comment in round_comments:
                    if 'parent_content_id' in comment:  # This is a reply
                        parent_id = comment['parent_content_id']
                        if parent_id in comment_map:  # Parent is in the same round
                            parent_comment = comment_map[parent_id]
                            
                            # Skip if reply has generation_info
                            if 'generation_info' in comment:
                                continue
                                
                            word_count = len(comment['content'].split())
                            
                            try:
                                result = sentiment_task(comment['content'], return_all_scores=True)[0]
                                
                                
                                if 'generation_info' in parent_comment:
                                    replies_to_generated['lengths'].append(word_count)
                                    for score in result:
                                        sentiment = score['label'].lower()
                                        replies_to_generated['sentiment'][sentiment].append(score['score'])
                                else:
                                    replies_to_nongenerated['lengths'].append(word_count)
                                    for score in result:
                                        sentiment = score['label'].lower()
                                        replies_to_nongenerated['sentiment'][sentiment].append(score['score'])
                                        
                            except Exception as e:
                                print(f"Error processing text: {comment['content'][:50]}... Error: {str(e)}")

    # Create figure with two subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Reply Length (with 95% Bootstrap CI)', 
                       'Reply Sentiment (with 95% Bootstrap CI)')
    )
    
    # Colors for consistency
    colors = {
        'to_generated': '#3498db',     # Blue
        'to_nongenerated': '#95a5a6'   # Gray
    }
    
    # Define offsets for each group
    group_offsets = {
        'to_generated': -0.1,
        'to_nongenerated': 0.1
    }
    
    # Calculate bootstrap CI for lengths
    for group, data in [('to_generated', replies_to_generated['lengths']), 
                       ('to_nongenerated', replies_to_nongenerated['lengths'])]:
        if data:
            mean, ci_lower, ci_upper = calculate_bootstrap_ci(pd.Series(data))
            
            fig.add_trace(
                go.Scatter(
                    x=[0 + group_offsets[group]],
                    y=[mean],
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array=[ci_upper - mean],
                        arrayminus=[mean - ci_lower],
                        width=10,
                    ),
                    mode='markers',
                    marker=dict(size=10, color=colors[group]),
                    name='Replies to Generated' if group == 'to_generated' else 'Replies to Non-Generated',
                    showlegend=True
                ),
                row=1, col=1
            )
    
    # Calculate bootstrap CI for sentiments
    sentiments = ['negative', 'neutral', 'positive']
    for group, data in [('to_generated', replies_to_generated['sentiment']), 
                       ('to_nongenerated', replies_to_nongenerated['sentiment'])]:
        for i, sentiment in enumerate(sentiments):
            if data[sentiment]:
                mean, ci_lower, ci_upper = calculate_bootstrap_ci(pd.Series(data[sentiment]))
                
                fig.add_trace(
                    go.Scatter(
                        x=[i + group_offsets[group]],
                        y=[mean],
                        error_y=dict(
                            type='data',
                            symmetric=False,
                            array=[ci_upper - mean],
                            arrayminus=[mean - ci_lower],
                            width=10,
                        ),
                        mode='markers',
                        marker=dict(size=10, color=colors[group]),
                        name='Replies to Generated' if group == 'to_generated' else 'Replies to Non-Generated',
                        showlegend=(i == 0)  # Only show legend for first sentiment
                    ),
                    row=1, col=2
                )
    
    # Update layout
    fig.update_layout(
        title='Reply Analysis: Generated vs Non-Generated Comments',
        height=500,
        template='plotly_white',
        showlegend=True,
        legend_title_text='Reply Type'
    )
    
    # Update axes
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(
        title_text="Sentiment Category",
        ticktext=sentiments,
        tickvals=list(range(len(sentiments))),
        row=1, col=2
    )
    fig.update_yaxes(title_text="Number of Words", row=1, col=1)
    fig.update_yaxes(title_text="Sentiment Score", range=[0, 1], row=1, col=2)
    
    # Add sample size annotation
    gen_count = len(replies_to_generated['lengths'])
    nongen_count = len(replies_to_nongenerated['lengths'])
    
    fig.add_annotation(
        text=f"Sample sizes:<br>Replies to Generated: {gen_count}<br>Replies to Non-Generated: {nongen_count}<br>Total: {gen_count + nongen_count}",
        xref="paper", yref="paper",
        x=1.02, y=1,
        showarrow=False,
        font=dict(size=10),
        align="left"
    )
    
    return fig

def create_generation_reaction_distribution_plot(df_generations_matched):
    """
    Create plot comparing reaction distributions across different generation types
    using bootstrap confidence intervals for proportions.
    
    Args:
        df_generations_matched (dict): Dictionary containing matched treatment/control data
    
    Returns:
        tuple: (plotly figure, dict of statistics)
    """
    # Initialize data structures
    reaction_data = {
        'positive': defaultdict(list),    # Will store lists of reactions for bootstrapping
        'neutral': defaultdict(list),
        'negative': defaultdict(list)
    }
    
    # Track totals for each generation type
    totals = {
        'positive': 0,
        'neutral': 0,
        'negative': 0
    }
    
    # Process treatment data
    treatment_data = df_generations_matched.get('suggestions_5', {})
    
    # Create mapping of content_ids to generation types
    content_id_to_type = {}
    for game_data in treatment_data.values():
        for round_comments in game_data['comments']:
            for comment in round_comments:
                if 'generation_info' in comment:
                    content_id_to_type[comment['content_id']] = \
                        comment['generation_info']['selection']['type']
    
    # Process actions and store individual reactions for bootstrapping
    for game_data in treatment_data.values():
        for round_actions in game_data['actions']:
            for action in round_actions:
                content_id = action.get('content_id')
                if content_id in content_id_to_type:
                    gen_type = content_id_to_type[content_id]
                    like_type = action.get('like_type')
                    if like_type:
                        reaction_data[gen_type][like_type].append(1)
                        # Add zeros for other reaction types to maintain proportions
                        for other_like_type in set().union(*[d.keys() for d in reaction_data.values()]):
                            if other_like_type != like_type:
                                reaction_data[gen_type][other_like_type].append(0)
                        totals[gen_type] += 1
    
    # Create figure
    fig = go.Figure()
    
    # Colors for generation types
    colors = {
        'positive': '#2ecc71',    # Green
        'neutral': '#f1c40f',     # Yellow
        'negative': '#e74c3c'     # Red
    }
    
    # Define offsets for each generation type
    group_offsets = {
        'positive': -0.2,
        'neutral': 0,
        'negative': 0.2
    }
    
    # Get all unique reaction types
    all_reactions = sorted(set().union(*[d.keys() for d in reaction_data.values()]))
    
    # Calculate bootstrap CIs and add traces
    for gen_type in ['positive', 'neutral', 'negative']:
        for i, reaction in enumerate(all_reactions):
            if reaction in reaction_data[gen_type]:
                # Calculate proportion CI
                proportions = pd.Series(reaction_data[gen_type][reaction])
                prop_mean, prop_ci_lower, prop_ci_upper = calculate_bootstrap_ci(proportions)
                
                # Add proportion trace
                fig.add_trace(
                    go.Scatter(
                        x=[i + group_offsets[gen_type]],
                        y=[prop_mean],
                        error_y=dict(
                            type='data',
                            symmetric=False,
                            array=[prop_ci_upper - prop_mean],
                            arrayminus=[prop_mean - prop_ci_lower],
                            width=10,
                        ),
                        mode='markers',
                        marker=dict(size=10, color=colors[gen_type]),
                        name=f'{gen_type.capitalize()} Selection',
                        showlegend=(i == 0)  # Show legend only for first reaction type
                    )
                )
    
    # Perform chi-square test on raw counts
    contingency_table = pd.DataFrame({
        reaction: [sum(reaction_data[gen_type][reaction]) for gen_type in ['positive', 'neutral', 'negative']]
        for reaction in all_reactions
    }, index=['positive', 'neutral', 'negative'])
    
    chi2, p_value = chi2_contingency(contingency_table)[:2]
    
    # Update layout
    fig.update_layout(
        title=f'Reaction Distribution by Generation Type (with 95% Bootstrap CI)<br>'
              f'<sup>Chi-square test: p-value = {p_value:.4f}</sup>',
        height=500,
        template='plotly_white',
        showlegend=True,
        legend_title_text='Selection Type'
    )
    
    # Update axes
    fig.update_xaxes(
        title_text="Reaction Type",
        ticktext=all_reactions,
        tickvals=list(range(len(all_reactions)))
    )
    fig.update_yaxes(
        title_text="Proportion",
        range=[0, 1]
    )
    
    # Add sample size annotation
    fig.add_annotation(
        text=f"Sample sizes:<br>" + 
             "<br>".join([f"{k.capitalize()}: {v}" for k, v in totals.items()]) + \
             f"<br>Total: {sum(totals.values())}",
        xref="paper", yref="paper",
        x=1.02, y=1,
        showarrow=False,
        font=dict(size=10),
        align="left"
    )
    
    return fig, {
        'raw_data': reaction_data,
        'totals': totals,
        'chi_square': chi2,
        'p_value': p_value
    }

def create_suggestion_usage_time_series(df_generations_matched):
    """
    Create time series plot showing when suggestions were used during rounds.
    
    Metrics tracked:
    - total_suggestions_used: Number of times a suggestion was used in a comment
    - total_suggestion_sets: Number of times suggestions were generated (3 at a time)
    """
    # Initialize data structures
    bucket_counts = {i: [] for i in range(5)}  # Store counts per round
    total_suggestions_used = 0  # Comments that used a suggestion
    total_suggestion_sets = 0   # Sets of suggestions generated (3 per set)
    out_of_window = {'early': 0, 'late': 0}
    
    # Process treatment group data
    treatment_data = df_generations_matched.get('suggestions_5', {})
    
    # Process each game
    for game_data in treatment_data.values():
        round_starts = [pd.Timestamp(start) for start in game_data['round_starts']]
        
        # Process each round
        for round_idx, start_time in enumerate(round_starts):
            # Initialize counts for this round
            round_bucket_counts = [0] * 5
            
            # Count suggestion sets generated
            round_suggestions = game_data.get('suggestions', {}).get(str(round_idx), {})
            for user_data in round_suggestions.values():
                if 'suggestions' in user_data:
                    total_suggestion_sets += len(user_data['suggestions'])
            
            # Count suggestions used in comments
            round_comments = game_data['comments'][round_idx]
            for comment in round_comments:
                if 'generation_info' in comment:
                    total_suggestions_used += 1
                    timestamp = pd.Timestamp(comment['timestamp'])
                    minutes_from_start = (timestamp - start_time).total_seconds() / 60
                    
                    if minutes_from_start < 0:
                        round_bucket_counts[0] += 1
                        out_of_window['early'] += 1
                    elif minutes_from_start > 10:
                        round_bucket_counts[4] += 1
                        out_of_window['late'] += 1
                    else:
                        bucket_idx = min(4, int(minutes_from_start / 2))
                        round_bucket_counts[bucket_idx] += 1
            
            # Store counts for this round
            for i in range(5):
                bucket_counts[i].append(round_bucket_counts[i])
    
    # Calculate statistics for each bucket
    bucket_stats = []
    for bucket in range(5):
        counts = bucket_counts[bucket]
        if counts:
            avg = np.mean(counts)
            std = np.std(counts)
        else:
            avg = 0
            std = 0
        bucket_stats.append({'avg': avg, 'std': std})
    
    # Create figure
    fig = go.Figure()
    
    # X-axis values (center of each 2-minute bucket)
    x_values = [1, 3, 5, 7, 9]  # Centers of 0-2, 2-4, 4-6, 6-8, 8-10 buckets
    
    # Add main line
    y = [stats['avg'] for stats in bucket_stats]
    error_y = [stats['std'] for stats in bucket_stats]
    
    fig.add_trace(go.Scatter(
        x=x_values,
        y=y,
        name='Average Suggestions Used',
        mode='lines+markers',
        line=dict(color='#2ecc71')  # Green color
    ))
    
    # Add error bands
    fig.add_trace(go.Scatter(
        x=x_values + x_values[::-1],
        y=[y[i] + error_y[i] for i in range(len(y))] + 
          [y[i] - error_y[i] for i in range(len(y))][::-1],
        fill='toself',
        fillcolor='#2ecc71',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        opacity=0.2,
        name='Error Band'
    ))
    
    # Update layout
    fig.update_layout(
        title='Average Suggestion Usage Over Time',
        xaxis=dict(
            title='Minutes',
            tickmode='array',
            ticktext=['0-2', '2-4', '4-6', '6-8', '8-10'],
            tickvals=x_values
        ),
        yaxis_title='Average Number of Suggestions Used',
        showlegend=True,
        height=500,
        hovermode='x unified'
    )
    
    # Add total suggestions annotation with both metrics
    fig.add_annotation(
        text=(f"Total suggestions used: {total_suggestions_used}<br>"
              f"Out-of-window suggestions:<br>"
              f"Early: {out_of_window['early']}<br>"
              f"Late: {out_of_window['late']}"),
        xref="paper", yref="paper",
        x=1.02, y=1,
        showarrow=False,
        font=dict(size=10),
        align="left"
    )
    
    return fig