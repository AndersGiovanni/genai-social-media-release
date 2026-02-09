import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
from transformers import pipeline
import streamlit as st
from collections import defaultdict
import json
import os
from scipy import stats
from players_helpers.ate import calculate_cohens_d


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


def create_engagement_plots(processed_data):
    """
    Create confidence interval plots comparing control and treatment groups.
    This function adapts to a scenario where non-control treatment groups
    can be combined under the key "treatment".
    """
    # Define color palette for multiple groups (including combined "treatment").
    colors = {
        'control': '#1f77b4',       # blue
        'treatment': '#2ca02c',      # green (for a combined treatment group)
        'chat_5': '#2ca02c',         # green
        'conversation_5': '#ff7f0e', # orange
        'feedback_5': '#d62728',     # red
        'suggestions_5': '#9467bd'   # purple
    }
    
    # Determine groups; if 'control' exists, ensure it is placed first.
    groups = list(processed_data.keys())
    if 'control' in groups:
        groups.remove('control')
        groups = ['control'] + groups

    # Prepare data structure for confidence intervals.
    metrics = ['Actions', 'Comments', 'Total Engagement']
    ci_data = []
    
    # Calculate CIs for overall metrics (summed across rounds).
    for group_type in groups:
        group_data = processed_data[group_type]
        
        # Collect per-game totals for each metric.
        game_totals = {
            'Actions': [],
            'Comments': [],
            'Total Engagement': []
        }
        
        for game_id, game in group_data.items():
            total_actions = sum(len(actions) for actions in game['actions'])
            total_comments = sum(len(comments) for comments in game['comments'])
            total_engagement = total_actions + total_comments
            
            game_totals['Actions'].append(total_actions)
            game_totals['Comments'].append(total_comments)
            game_totals['Total Engagement'].append(total_engagement)
        
        # Calculate CI for each metric.
        for metric in metrics:
            mean, ci_lower, ci_upper = calculate_bootstrap_ci(pd.Series(game_totals[metric]))
            ci_data.append({
                'group': group_type,
                'metric': metric,
                'mean': mean,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            })
    
    # Create overall engagement plot.
    df_ci = pd.DataFrame(ci_data)
    overall_fig = go.Figure()
    
    # Define offsets for each group. Note the added offset for 'treatment'.
    group_offsets = {
        'control': -0.2,
        'treatment': -0.1,
        'chat_5': -0.1,
        'conversation_5': 0,
        'feedback_5': 0.1,
        'suggestions_5': 0.2
    }
    
    # Add traces for each group.
    for group in groups:
        group_data = df_ci[df_ci['group'] == group]
        
        # Add mean points and CI error bars.
        overall_fig.add_trace(go.Scatter(
            x=[x + group_offsets.get(group, 0) for x in range(len(metrics))],
            y=group_data['mean'],
            error_y=dict(
                type='data',
                symmetric=False,
                array=group_data['ci_upper'] - group_data['mean'],
                arrayminus=group_data['mean'] - group_data['ci_lower'],
                width=10,
            ),
            mode='markers',
            marker=dict(size=10, color=colors.get(group, '#000000')),
            name=group,
            showlegend=True
        ))
    
    overall_fig.update_layout(
        title='Overall Game Engagement (with 95% Bootstrap CI)',
        xaxis=dict(
            ticktext=metrics,
            tickvals=list(range(len(metrics))),
            title="Metric"
        ),
        yaxis_title="Count per Game",
        height=500,
        template='plotly_white',
        showlegend=True,
        legend_title_text='Group'
    )
    
    # Create round-wise plots.
    round_figs = []
    for metric in metrics:
        round_ci_data = []
        
        for group_type in groups:
            group_data = processed_data[group_type]
            
            # Collect round-wise values.
            for round_num in range(3):
                round_values = []
                for game in group_data.values():
                    if metric == 'Actions':
                        value = len(game['actions'][round_num])
                    elif metric == 'Comments':
                        value = len(game['comments'][round_num])
                    else:  # Total Engagement
                        value = len(game['actions'][round_num]) + len(game['comments'][round_num])
                    round_values.append(value)
                
                # Calculate CI for this round.
                mean, ci_lower, ci_upper = calculate_bootstrap_ci(pd.Series(round_values))
                round_ci_data.append({
                    'group': group_type,
                    'round': f'Round {round_num + 1}',
                    'mean': mean,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper
                })
        
        # Create round-wise plot for the current metric.
        df_round = pd.DataFrame(round_ci_data)
        round_fig = go.Figure()
        
        for group in groups:
            group_data = df_round[df_round['group'] == group]
            
            round_fig.add_trace(go.Scatter(
                x=[float(x.split()[-1]) - 1 + group_offsets.get(group, 0) for x in group_data['round']],
                y=group_data['mean'],
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=group_data['ci_upper'] - group_data['mean'],
                    arrayminus=group_data['mean'] - group_data['ci_lower'],
                    width=10,
                ),
                mode='markers',
                marker=dict(size=10, color=colors.get(group, '#000000')),
                name=group,
                showlegend=True
            ))
        
        round_fig.update_layout(
            title=f'Round-wise {metric} (with 95% Bootstrap CI)',
            xaxis=dict(
                ticktext=['Round 1', 'Round 2', 'Round 3'],
                tickvals=[0, 1, 2],
                title="Round"
            ),
            yaxis_title=f"{metric} per Round",
            height=500,
            template='plotly_white',
            showlegend=True,
            legend_title_text='Group'
        )
        
        round_figs.append(round_fig)
    
    return overall_fig, round_figs

def create_metric_comparison_plots(total_metrics, avg_metrics):
    """
    Create two plots showing averages and totals for control and treatment(s) using bootstrap CIs.
    
    This function now supports both cases:
       - When a control group exists (with key 'control') 
         and treatment groups are provided individually.
       - When all treatment groups are combined under a single key (e.g. 'treatment')
         and no separate control grouping exists.
         
    Args:
        total_metrics (dict): A dictionary containing total game counts and related metrics,
                              keyed by group.
        avg_metrics (dict): A dictionary containing average metrics per group and round.
                            For each group, metric, and round, this dictionary must provide a dict with:
                                'avg': the average value,
                                'ci_lower': the lower bootstrap confidence interval value, and
                                'ci_upper': the upper bootstrap confidence interval value.
    
    Returns:
        tuple: (avg_figs, total_figs) where each is a list of Plotly figures.
    """
    # Define color palette for multiple groups (using distinct colors).
    # An entry for "treatment" is added in case treatments are combined.
    colors = {
        'control': '#1f77b4',  # blue
        'treatment': '#2ca02c', # green (for a combined treatment)
        'chat_5': '#2ca02c',    # green
        'conversation_5': '#ff7f0e',  # orange
        'feedback_5': '#d62728',  # red
        'suggestions_5': '#9467bd'  # purple
    }
    
    line_styles = {'actions': 'solid', 'comments': 'dash'}
    
    # Get groups and ensure that if 'control' exists it is placed first. 
    groups = list(sorted(total_metrics.keys()))
    if 'control' in groups:
        groups.remove('control')
        groups = ['control'] + groups

    # Create average comparison plot with error bands using bootstrap CIs.
    avg_data = []
    for round_num in range(3):
        for metric in ['actions', 'comments']:
            for group in groups:
                avg_data.append({
                    'round': round_num,
                    'metric': metric,
                    'group': group,
                    'value': avg_metrics[group][metric][round_num]['avg'],
                    'ci_lower': avg_metrics[group][metric][round_num]['ci_lower'],
                    'ci_upper': avg_metrics[group][metric][round_num]['ci_upper']
                })
    
    df_avg = pd.DataFrame(avg_data)
    
    avg_figs = []
    total_figs = []
    
    for metric in ['actions', 'comments']:
        # Average plot
        avg_fig = go.Figure()
        
        for group in groups:
            mask = (df_avg['group'] == group) & (df_avg['metric'] == metric)
            df_subset = df_avg[mask]
            
            # Add the main line for averages.
            avg_fig.add_trace(go.Scatter(
                x=df_subset['round'],
                y=df_subset['value'],
                name=group,
                mode='lines+markers',
                line=dict(
                    color=colors.get(group, '#000000'),
                    dash=line_styles[metric]
                ),
                showlegend=True
            ))
            
            # Now add error bands using the bootstrap CI values.
            avg_fig.add_trace(go.Scatter(
                x=df_subset['round'].tolist() + df_subset['round'].tolist()[::-1],
                y=df_subset['ci_upper'].tolist() + df_subset['ci_lower'].tolist()[::-1],
                fill='toself',
                fillcolor=colors.get(group, '#000000'),
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                opacity=0.2,
                name=f"{group} CI Band"
            ))
        
        avg_fig.update_layout(
            title=f'Average {metric.capitalize()} per Round (with 95% Bootstrap CI)',
            xaxis_title='Round',
            yaxis_title=f'Average {metric.capitalize()} Count',
            xaxis=dict(
                tickmode='array',
                ticktext=['Round 0', 'Round 1', 'Round 2'],
                tickvals=[0, 1, 2]
            ),
            height=500,
            legend_title='Groups',
            showlegend=True,
            hovermode='x unified'
        )
        avg_figs.append(avg_fig)
        
        # Total plot
        total_fig = go.Figure()
        
        for group in groups:
            # Calculate total per round based on the number of games.
            num_games = total_metrics[group]['games']
            y_values = [
                avg_metrics[group][metric][round_num]['avg'] * num_games
                for round_num in range(3)
            ]
            
            total_fig.add_trace(go.Scatter(
                x=[0, 1, 2],
                y=y_values,
                name=group,
                mode='lines+markers',
                line=dict(color=colors.get(group, '#000000')),
                marker=dict(size=10)
            ))
        
        total_fig.update_layout(
            title=f'Total {metric.capitalize()} per Round',
            xaxis_title='Round',
            yaxis_title=f'Total {metric.capitalize()} Count',
            xaxis=dict(
                tickmode='array',
                ticktext=['Round 0', 'Round 1', 'Round 2'],
                tickvals=[0, 1, 2]
            ),
            height=500,
            legend_title='Groups',
            showlegend=True,
            hovermode='x unified'
        )
        total_figs.append(total_fig)
    
    return avg_figs, total_figs

def create_comment_length_plot(processed_data):
    """
    Create confidence interval plots comparing comment lengths between groups.
    """
    # Define color palette for multiple groups (including a 'treatment' option).
    colors = {
        'control': '#1f77b4',   # blue
        'treatment': '#2ca02c',  # green (for a combined treatment)
        'chat_5': '#2ca02c',     # green
        'conversation_5': '#ff7f0e',  # orange
        'feedback_5': '#d62728', # red
        'suggestions_5': '#9467bd'  # purple
    }
    
    # Determine group ordering: if 'control' exists, place it first.
    groups = sorted(processed_data.keys())
    if 'control' in groups:
        groups.remove('control')
        groups = ['control'] + groups
    
    # Define offsets for each group marker.
    group_offsets = {
        'control': -0.2,
        'treatment': -0.1,
        'chat_5': -0.1,
        'conversation_5': 0,
        'feedback_5': 0.1,
        'suggestions_5': 0.2
    }
    
    # Process all groups and calculate bootstrap confidence intervals.
    ci_data = []
    raw_data = {'words': [], 'characters': []}  # For summary statistics
    
    # Store raw values for significance testing
    raw_values = {group: {'words': [], 'characters': []} for group in groups}
    
    for group_type in groups:
        group_data = processed_data[group_type]
        words = []
        chars = []
        
        # Collect all comments from all games and rounds.
        for game in group_data.values():
            for round_comments in game['comments']:
                for comment in round_comments:
                    word_count = len(comment['content'].split())
                    char_count = len(comment['content'])
                    words.append(word_count)
                    chars.append(char_count)
                    raw_data['words'].append({'group': group_type, 'value': word_count})
                    raw_data['characters'].append({'group': group_type, 'value': char_count})
                    raw_values[group_type]['words'].append(word_count)
                    raw_values[group_type]['characters'].append(char_count)
        
        # Calculate bootstrap confidence intervals.
        word_mean, word_ci_lower, word_ci_upper = calculate_bootstrap_ci(pd.Series(words))
        char_mean, char_ci_lower, char_ci_upper = calculate_bootstrap_ci(pd.Series(chars))
        
        ci_data.append({
            'group': group_type,
            'metric': 'words',
            'mean': word_mean,
            'ci_lower': word_ci_lower,
            'ci_upper': word_ci_upper
        })
        ci_data.append({
            'group': group_type,
            'metric': 'characters',
            'mean': char_mean,
            'ci_lower': char_ci_lower,
            'ci_upper': char_ci_upper
        })
    
    # Calculate significance and effect size compared to control
    significance_data = {}
    if 'control' in groups:
        for group in groups:
            if group != 'control':
                significance_data[group] = {}
                for metric in ['words', 'characters']:
                    control_vals = raw_values['control'][metric]
                    treatment_vals = raw_values[group][metric]
                    if len(control_vals) > 0 and len(treatment_vals) > 0:
                        p_value = calculate_p_value(pd.Series(treatment_vals), pd.Series(control_vals))
                        effect = calculate_cohens_d(treatment_vals, control_vals)
                        significance_data[group][metric] = {
                            'p_value': p_value,
                            'stars': get_stars(p_value),
                            'cohens_d': effect['d'],
                            'd_ci_lower': effect['ci_lower'],
                            'd_ci_upper': effect['ci_upper'],
                            'effect_interpretation': effect['interpretation']
                        }
    
    # Create output data structure
    output_data = {}

    for metric in ['words', 'characters']:
        metric_data = {
            "x_vals": list(range(len(groups))),
            "y_vals": [],
            "err": [],
            "significance": [],
            "p_values": [],
            "cohens_d": [],
            "d_ci_lower": [],
            "d_ci_upper": [],
            "effect_interpretation": [],
            "n_values": [],
            "treatment_labels": groups,
            "question_label": f"Comment {metric} analysis"
        }

        for group in groups:
            group_data = [d for d in ci_data if d['group'] == group and d['metric'] == metric][0]

            # Add mean value
            metric_data["y_vals"].append(group_data['mean'])

            # Add error (using upper CI - mean as the error value)
            metric_data["err"].append(group_data['ci_upper'] - group_data['mean'])

            # Add significance and effect size (or reference for control group)
            if group == 'control':
                metric_data["significance"].append("ns")
                metric_data["p_values"].append(None)
                metric_data["cohens_d"].append(0.0)
                metric_data["d_ci_lower"].append(0.0)
                metric_data["d_ci_upper"].append(0.0)
                metric_data["effect_interpretation"].append("reference")
            else:
                sig_data = significance_data.get(group, {}).get(metric, {})
                metric_data["significance"].append(sig_data.get('stars', 'ns'))
                metric_data["p_values"].append(sig_data.get('p_value'))
                metric_data["cohens_d"].append(sig_data.get('cohens_d'))
                metric_data["d_ci_lower"].append(sig_data.get('d_ci_lower'))
                metric_data["d_ci_upper"].append(sig_data.get('d_ci_upper'))
                metric_data["effect_interpretation"].append(sig_data.get('effect_interpretation', ''))

            # Add n_values (number of comments)
            n = len(raw_values[group][metric])
            metric_data["n_values"].append(n)

        output_data[f"comment_{metric}"] = metric_data
    
    # Save to JSON file
    # Create directory if it doesn't exist
    os.makedirs('data/plotting', exist_ok=True)
    
    # Save the data
    with open('data/comment_length_data.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Create figure with two subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Word Count (with 95% Bootstrap CI)', 
                       'Character Count (with 95% Bootstrap CI)')
    )
    
    # Convert confidence interval data to a DataFrame.
    df_ci = pd.DataFrame(ci_data)
    
    # Find maximum y values for each subplot to position stars
    max_y_words = max(df_ci[df_ci['metric'] == 'words']['ci_upper'])
    max_y_chars = max(df_ci[df_ci['metric'] == 'characters']['ci_upper'])
    
    # Add traces and significance stars for both subplots
    for col, (metric, max_y) in enumerate([('words', max_y_words), ('characters', max_y_chars)], 1):
        for group in groups:
            group_data = df_ci[(df_ci['group'] == group) & (df_ci['metric'] == metric)]
            
            # Add main trace
            fig.add_trace(
                go.Scatter(
                    x=[0 + group_offsets.get(group, 0)],
                    y=[group_data['mean'].iloc[0]],
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array=[group_data['ci_upper'].iloc[0] - group_data['mean'].iloc[0]],
                        arrayminus=[group_data['mean'].iloc[0] - group_data['ci_lower'].iloc[0]],
                        width=10,
                    ),
                    mode='markers',
                    marker=dict(size=10, color=colors.get(group, '#000000')),
                    name=group,
                    showlegend=(col == 1)  # Show legend only for first subplot
                ),
                row=1, col=col
            )
            
            # Get sample size and mean/CI for this group
            n_samples = len(raw_values[group][metric])
            mean_val = group_data['mean'].iloc[0]
            ci_lower = group_data['ci_lower'].iloc[0]
            ci_upper = group_data['ci_upper'].iloc[0]
            y_pos = ci_upper

            # Add significance stars, p-values and Cohen's d with CI for treatment groups
            if group != 'control' and group in significance_data and metric in significance_data[group]:
                sig_data = significance_data[group][metric]
                p_val = sig_data.get('p_value')
                stars = sig_data.get('stars', 'ns')
                d_val = sig_data.get('cohens_d')
                d_ci_lower = sig_data.get('d_ci_lower')
                d_ci_upper = sig_data.get('d_ci_upper')
                effect_interp = sig_data.get('effect_interpretation', '')

                # Build annotation text with mean, CI, p-value, and Cohen's d
                annotation_text = f"n={n_samples}<br>μ={mean_val:.1f} [{ci_lower:.1f}, {ci_upper:.1f}]<br>p={p_val:.3f} {stars}"
                if d_val is not None:
                    if d_ci_lower is not None and d_ci_upper is not None:
                        annotation_text += f"<br>d={d_val:.2f} [{d_ci_lower:.2f}, {d_ci_upper:.2f}] ({effect_interp})"
                    else:
                        annotation_text += f"<br>d={d_val:.2f} ({effect_interp})"

                fig.add_annotation(
                    x=group_offsets.get(group, 0),
                    y=y_pos + (max_y * 0.18),
                    text=annotation_text,
                    showarrow=False,
                    font=dict(size=8),
                    align='center',
                    row=1, col=col
                )
            else:
                # Control group - show N and mean with CI
                fig.add_annotation(
                    x=group_offsets.get(group, 0),
                    y=y_pos + (max_y * 0.10),
                    text=f"n={n_samples}<br>μ={mean_val:.1f} [{ci_lower:.1f}, {ci_upper:.1f}]",
                    showarrow=False,
                    font=dict(size=8),
                    align='center',
                    row=1, col=col
                )
    
    # Update layout with adjusted y-ranges for stars
    fig.update_layout(
        title='Comment Length Analysis (with 95% Bootstrap CI)',
        height=500,
        template='plotly_white',
        showlegend=True,
        legend_title_text='Group',
        margin=dict(l=60, r=60, t=80, b=120)
    )
    
    # Update axes
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=2)
    fig.update_yaxes(title_text="Number of Words", range=[15, max_y_words * 1.45], row=1, col=1)
    fig.update_yaxes(title_text="Number of Characters", range=[50, max_y_chars * 1.45], row=1, col=2)
    
    # Add summary statistics as annotations
    for i, metric in enumerate(['words', 'characters']):
        df_raw = pd.DataFrame(raw_data[metric])
        annotation_text = []
        for group in groups:
            group_stats = df_raw[df_raw['group'] == group]['value']
            mean_val = group_stats.mean()
            median_val = group_stats.median()
            annotation_text.append(f"{group}: Mean={mean_val:.1f}, Median={median_val:.1f}")
        fig.add_annotation(
            text="<br>".join(annotation_text),
            xref=f"x{i+1}", yref="paper",
            x=0.5, y=-0.35,
            showarrow=False,
            font=dict(size=10),
            align="left"
        )
    
    # Add significance legend
    fig.add_annotation(
        xref='paper',
        yref='paper',
        x=1.15,
        y=1.0,
        text='†p<0.1, *p<0.05, **p<0.01, ***p<0.001<br>ns: not significant',
        showarrow=False,
        font=dict(size=10),
        xanchor='right',
        yanchor='top'
    )

    return fig

def create_likes_distribution_plot(processed_data):
    """Create plot comparing the distribution of like types across all groups using bootstrap CIs."""
    
    # Colors for consistency
    colors = {
        'control': '#1f77b4',   # blue
        'treatment': '#2ca02c',  # green (for a combined treatment)
        'chat_5': '#2ca02c',     # green
        'conversation_5': '#ff7f0e',  # orange
        'feedback_5': '#d62728', # red
        'suggestions_5': '#9467bd'  # purple
    }
    
    # Determine group ordering: if 'control' exists, place it first.
    groups = sorted(processed_data.keys())
    if 'control' in groups:
        groups.remove('control')
        groups = ['control'] + groups
    
    # Define offsets for each group marker (increased spacing to prevent text overlap)
    group_offsets = {
        'control': -0.35,
        'treatment': -0.15,  # offset for a combined treatment
        'chat_5': -0.15,
        'conversation_5': 0.05,
        'feedback_5': 0.25,
        'suggestions_5': 0.45
    }
    
    # Process data and calculate proportions
    like_data = {group: defaultdict(list) for group in groups}
    totals = {group: 0 for group in groups}
    
    for group_type in groups:
        group_data = processed_data[group_type]
        for game in group_data.values():
            for round_actions in game['actions']:
                for action in round_actions:
                    if 'like_type' in action:
                        like_type = action['like_type']
                        like_data[group_type][like_type].append(1)
                        for other_like_type in set().union(*[d.keys() for d in like_data.values()]):
                            if other_like_type != like_type:
                                like_data[group_type][other_like_type].append(0)
                        totals[group_type] += 1
    
    # Create output data structure
    output_data = {}
    all_like_types = sorted(set().union(*[d.keys() for d in like_data.values()]))

    # Pre-calculate significance and effect sizes for all groups and like types
    significance_data = {}
    for group in groups:
        if group != 'control':
            significance_data[group] = {}
            for like_type in all_like_types:
                if like_type in like_data[group] and like_type in like_data['control']:
                    control_props = like_data['control'][like_type]
                    treatment_props = like_data[group][like_type]
                    if len(control_props) > 0 and len(treatment_props) > 0:
                        p_value = calculate_p_value(pd.Series(treatment_props), pd.Series(control_props))
                        effect = calculate_cohens_d(treatment_props, control_props)
                        significance_data[group][like_type] = {
                            'p_value': p_value,
                            'stars': get_stars(p_value),
                            'cohens_d': effect['d'],
                            'd_ci_lower': effect['ci_lower'],
                            'd_ci_upper': effect['ci_upper'],
                            'effect_interpretation': effect['interpretation']
                        }

    for like_type in all_like_types:
        like_type_data = {
            "x_vals": list(range(len(groups))),
            "y_vals": [],
            "err": [],
            "significance": [],
            "p_values": [],
            "cohens_d": [],
            "d_ci_lower": [],
            "d_ci_upper": [],
            "effect_interpretation": [],
            "n_values": [],
            "treatment_labels": groups,
            "question_label": f"Distribution of {like_type} reactions"
        }

        for group in groups:
            if like_type in like_data[group]:
                # Calculate proportion CI
                proportions = pd.Series(like_data[group][like_type])
                prop_mean, prop_ci_lower, prop_ci_upper = calculate_bootstrap_ci(proportions)

                # Add mean value
                like_type_data["y_vals"].append(prop_mean)

                # Add error (using upper CI - mean as the error value)
                like_type_data["err"].append(prop_ci_upper - prop_mean)

                # Add significance and effect size (or reference for control group)
                if group == 'control':
                    like_type_data["significance"].append("ns")
                    like_type_data["p_values"].append(None)
                    like_type_data["cohens_d"].append(0.0)
                    like_type_data["d_ci_lower"].append(0.0)
                    like_type_data["d_ci_upper"].append(0.0)
                    like_type_data["effect_interpretation"].append("reference")
                else:
                    sig_data = significance_data.get(group, {}).get(like_type, {})
                    like_type_data["significance"].append(sig_data.get('stars', 'ns'))
                    like_type_data["p_values"].append(sig_data.get('p_value'))
                    like_type_data["cohens_d"].append(sig_data.get('cohens_d'))
                    like_type_data["d_ci_lower"].append(sig_data.get('d_ci_lower'))
                    like_type_data["d_ci_upper"].append(sig_data.get('d_ci_upper'))
                    like_type_data["effect_interpretation"].append(sig_data.get('effect_interpretation', ''))

                # Add n_values (count of this specific reaction type for this group)
                like_type_data["n_values"].append(sum(like_data[group][like_type]))

        output_data[f"likes_{like_type}"] = like_type_data
    
    # Save to JSON file
    # Create directory if it doesn't exist
    os.makedirs('data/plotting', exist_ok=True)
    
    # Save the data
    with open('data/likes_distribution_data.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Create figure
    fig = go.Figure()

    # Like types to display visually (subset of all_like_types)
    display_like_types = ['Like', 'Love', 'Funny', 'Dislike']
    # Filter to only include types that exist in the data
    display_like_types = [lt for lt in display_like_types if lt in all_like_types]

    # Find max y value for positioning stars
    max_y = 0

    # Calculate bootstrap CIs and add traces (only for display_like_types)
    for group in groups:
        for i, like_type in enumerate(display_like_types):
            if like_type in like_data[group]:
                # Calculate proportion CI
                proportions = pd.Series(like_data[group][like_type])
                prop_mean, prop_ci_lower, prop_ci_upper = calculate_bootstrap_ci(proportions)
                max_y = max(max_y, prop_ci_upper)

                # Add trace
                fig.add_trace(
                    go.Scatter(
                        x=[i + group_offsets[group]],
                        y=[prop_mean],
                        error_y=dict(
                            type='data',
                            symmetric=False,
                            array=[prop_ci_upper - prop_mean],
                            arrayminus=[prop_mean - prop_ci_lower],
                            width=10,
                        ),
                        mode='markers',
                        marker=dict(size=10, color=colors[group]),
                        name=group,
                        showlegend=(i == 0)  # Show legend only for first like type
                    )
                )

                # Get sample size for this group and like type (sum of 1s, not length)
                n_samples = sum(like_data[group][like_type])

                # Add significance stars, p-values and Cohen's d with CI for treatment groups
                if group != 'control':
                    sig_data = significance_data.get(group, {}).get(like_type, {})
                    p_val = sig_data.get('p_value')
                    stars = sig_data.get('stars', 'ns')
                    d_val = sig_data.get('cohens_d')
                    d_ci_lower = sig_data.get('d_ci_lower')
                    d_ci_upper = sig_data.get('d_ci_upper')
                    effect_interp = sig_data.get('effect_interpretation', '')

                    # Build annotation text with mean, CI, p-value, and Cohen's d
                    annotation_text = f"n={n_samples}<br>μ={prop_mean:.2f} [{prop_ci_lower:.2f}, {prop_ci_upper:.2f}]<br>p={p_val:.3f} {stars}"
                    if d_val is not None:
                        if d_ci_lower is not None and d_ci_upper is not None:
                            annotation_text += f"<br>d={d_val:.2f} [{d_ci_lower:.2f}, {d_ci_upper:.2f}])"
                        else:
                            annotation_text += f"<br>d={d_val:.2f}"

                    fig.add_annotation(
                        x=i + group_offsets[group],
                        y=prop_mean + (prop_ci_upper - prop_mean) + 0.18,
                        text=annotation_text,
                        showarrow=False,
                        font=dict(size=8),
                        align='center'
                    )
                else:
                    # Control group - show N and mean with CI
                    fig.add_annotation(
                        x=i + group_offsets[group],
                        y=prop_mean + (prop_ci_upper - prop_mean) + 0.10,
                        text=f"n={n_samples}<br>μ={prop_mean:.2f} [{prop_ci_lower:.2f}, {prop_ci_upper:.2f}]",
                        showarrow=False,
                        font=dict(size=8),
                        align='center'
                    )
    
    # Update layout
    fig.update_layout(
        title='Distribution of Like Types (with 95% Bootstrap CI)',
        height=500,
        template='plotly_white',
        showlegend=True,
        legend_title_text='Group',
        yaxis=dict(
            title_text="Proportion",
            range=[0, max_y + 0.55]  # Add space for annotations with N, mean, CI, p-value, Cohen's d
        ),
        xaxis=dict(
            title_text="Like Type",
            ticktext=display_like_types,
            tickvals=list(range(len(display_like_types)))
        )
    )

    # Add significance legend
    fig.add_annotation(
        xref='paper',
        yref='paper',
        x=1.15,
        y=1.0,
        text='†p<0.1, *p<0.05, **p<0.01, ***p<0.001<br>ns: not significant',
        showarrow=False,
        font=dict(size=10),
        xanchor='right',
        yanchor='top'
    )
    
    # Add sample size annotation
    fig.add_annotation(
        text=f"Sample sizes:<br>" + 
             "<br>".join([f"{k}: {v}" for k, v in totals.items()]) + \
             f"<br>Total: {sum(totals.values())}",
        xref="paper", yref="paper",
        x=1.02, y=1,
        showarrow=False,
        font=dict(size=10),
        align="left"
    )
    
    return fig

def process_time_series_data(processed_data):
    """
    Process engagement data into 2-minute time buckets for each round.
    
    Args:
        processed_data (dict): Dictionary containing processed data for all groups
        
    Returns:
        tuple: (
            dict: Processed time series data for each group and metric,
            dict: Flags indicating out-of-window engagements
        )
    """
    # Initialize data structures for multiple groups
    time_series_data = {group: {'actions': [], 'comments': [], 'total': []} for group in processed_data.keys()}
    out_of_window = {group: {'early': 0, 'late': 0} for group in processed_data.keys()}
    
    # Create 5 two-minute buckets
    buckets = list(range(5))
    
    for group_type, group_data in processed_data.items():
        # Initialize arrays for each game's counts
        round_counts = {
            'actions': [[] for _ in buckets],
            'comments': [[] for _ in buckets],
            'total': [[] for _ in buckets]
        }
        
        # Process each game
        for game in group_data.values():
            round_starts = [pd.Timestamp(start) for start in game['round_starts']]
            
            # Process each round
            for round_idx, start_time in enumerate(round_starts):
                # Get actions and comments for this round
                round_actions = game['actions'][round_idx]
                round_comments = game['comments'][round_idx]
                
                # Process actions
                valid_actions, invalid_actions = process_items_for_bucket(
                    round_actions, 
                    start_time,
                    'action'
                )
                
                # Process comments
                valid_comments, invalid_comments = process_items_for_bucket(
                    round_comments,
                    start_time,
                    'comment'
                )
                
                # Count invalid items
                out_of_window[group_type]['early'] += sum(
                    1 for item in invalid_actions + invalid_comments
                    if item['timestamp'] < start_time - pd.Timedelta(seconds=30)
                )
                out_of_window[group_type]['late'] += sum(
                    1 for item in invalid_actions + invalid_comments
                    if item['timestamp'] > start_time + pd.Timedelta(minutes=10, seconds=30)
                )
                
                # Initialize bucket counts for this round
                bucket_counts = {
                    'actions': [0] * len(buckets),
                    'comments': [0] * len(buckets),
                    'total': [0] * len(buckets)
                }
                
                # Process valid actions
                for action in valid_actions:
                    timestamp = pd.to_datetime(action['timestamp'])
                    minutes_from_start = (timestamp - start_time).total_seconds() / 60
                    bucket_idx = min(4, max(0, int(minutes_from_start / 2)))
                    bucket_counts['actions'][bucket_idx] += 1
                    bucket_counts['total'][bucket_idx] += 1
                
                # Process valid comments
                for comment in valid_comments:
                    timestamp = pd.to_datetime(comment['timestamp'])
                    minutes_from_start = (timestamp - start_time).total_seconds() / 60
                    bucket_idx = min(4, max(0, int(minutes_from_start / 2)))
                    bucket_counts['comments'][bucket_idx] += 1
                    bucket_counts['total'][bucket_idx] += 1
                
                # Add this round's counts to the overall arrays
                for metric in ['actions', 'comments', 'total']:
                    for bucket_idx in range(len(buckets)):
                        round_counts[metric][bucket_idx].append(bucket_counts[metric][bucket_idx])
        
        # Calculate averages and std for each bucket
        for metric in ['actions', 'comments', 'total']:
            bucket_stats = []
            for bucket_counts in round_counts[metric]:
                avg = np.mean(bucket_counts)
                std = np.std(bucket_counts)
                bucket_stats.append({'avg': avg, 'std': std})
            
            time_series_data[group_type][metric] = bucket_stats
    
    return time_series_data, out_of_window

def create_time_series_plots(time_series_data, out_of_window):
    """
    Create interactive time series plots with error bands for each metric.
    
    Args:
        time_series_data (dict): Processed time series data
        out_of_window (dict): Flags for out-of-window engagements
        
    Returns:
        list: Plotly figures for each metric
    """
    # Define colors including an entry for combined treatments.
    colors = {
        'control': '#1f77b4',   # blue
        'treatment': '#2ca02c',  # green (for a combined treatment)
        'chat_5': '#2ca02c',
        'conversation_5': '#ff7f0e',
        'feedback_5': '#d62728',
        'suggestions_5': '#9467bd'
    }
    
    # X-axis values (center of each 2-minute bucket)
    x_values = [1, 3, 5, 7, 9]  # Centers of 0-2, 2-4, 4-6, 6-8, 8-10 buckets
    
    # Determine group ordering. If 'control' exists, place it first.
    groups = list(time_series_data.keys())
    if 'control' in groups:
        groups.remove('control')
        groups = ['control'] + groups
    
    # Create a plot for each metric
    metric_figs = []
    for metric in ['actions', 'comments', 'total']:
        fig = go.Figure()
        
        for group in groups:
            data = time_series_data[group][metric]
            y = [d['avg'] for d in data]
            
            fig.add_trace(go.Scatter(
                x=x_values,
                y=y,
                name=f'{group.capitalize()}',
                mode='lines+markers',
                line=dict(color=colors.get(group, '#000000'))
            ))
        
        fig.update_layout(
            title=f'{metric.capitalize()} Over Time (2-minute Buckets)',
            xaxis=dict(
                title='Minutes',
                tickmode='array',
                ticktext=['0-2', '2-4', '4-6', '6-8', '8-10'],
                tickvals=x_values
            ),
            yaxis_title='Average Count',
            showlegend=True,
            height=500,
            hovermode='x unified'
        )
        
        metric_figs.append(fig)
    
    # Display out-of-window engagements separately.
    out_of_window_text = []
    for group, counts in out_of_window.items():
        early = counts['early']
        late = counts['late']
        if early + late > 0:
            out_of_window_text.append(f"{group.capitalize()}: {early} early, {late} late")
    
    if out_of_window_text:
        st.markdown("**Out-of-window Engagements:**")
        st.write(", ".join(out_of_window_text))
    
    return metric_figs


@st.cache_resource
def load_sentiment_pipeline():
    """Load and cache the sentiment analysis model"""
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    sentiment_task = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)
    return sentiment_task

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

def create_sentiment_plot(processed_data):
    """
    Create confidence interval plot comparing sentiment scores across all groups.
    """
    # Colors for consistency
    colors = {
        'control': '#1f77b4',  # blue
        'treatment': '#2ca02c',  # green (for a combined treatment)
        'chat_5': '#2ca02c',   # green
        'conversation_5': '#ff7f0e',  # orange
        'feedback_5': '#d62728',  # red
        'suggestions_5': '#9467bd'  # purple
    }
    
    # Determine group ordering. If 'control' exists, ensure it is first.
    groups = sorted(processed_data.keys())
    if 'control' in groups:
        groups.remove('control')
        groups = ['control'] + groups
    
    # Define offsets for each group.
    group_offsets = {
        'control': -0.2,
        'treatment': -0.1,  # offset for a combined treatment group
        'chat_5': -0.1,
        'conversation_5': 0,
        'feedback_5': 0.1,
        'suggestions_5': 0.2
    }
    
    # Initialize structure for sentiment scores.
    sentiment_data = {group: {'negative': [], 'neutral': [], 'positive': []} for group in groups}
    
    # Count the total number of comments.
    total_comments = 0
    for group_data in processed_data.values():
        for game in group_data.values():
            for round_comments in game['comments']:
                total_comments += len(round_comments)
    
    # Process comments and collect sentiment scores.
    with st.spinner('Analyzing sentiment of comments...'):
        progress_bar = st.progress(0)
        status_placeholder = st.empty()
        comments_processed = 0
        
        for group_type in groups:
            group_data = processed_data[group_type]
            all_texts = [comment['content'] for game in group_data.values() 
                         for round_comments in game['comments'] 
                         for comment in round_comments]
            
            batch_size = len(all_texts)
            if batch_size > 0:
                percentage = min(100, round((comments_processed / total_comments) * 100))
                status_placeholder.write(
                    f"Processing {batch_size} comments for {group_type}...\n"
                    f"Progress: {percentage}% ({comments_processed}/{total_comments} comments processed)"
                )
                
                results = analyze_sentiment_batch(all_texts)
                if results:
                    for text_results in results:
                        for score_dict in text_results:
                            label = score_dict['label'].lower()
                            score = score_dict['score']
                            sentiment_data[group_type][label].append(score)
                
                comments_processed += batch_size
                progress_bar.progress(min(1.0, comments_processed / total_comments))
    
    status_placeholder.empty()
    st.success('Sentiment analysis completed!')
    
    # Calculate confidence intervals (CIs) for each sentiment.
    ci_data = []
    sentiments = ['negative', 'neutral', 'positive']
    
    for group in groups:
        for sentiment in sentiments:
            scores = pd.Series(sentiment_data[group][sentiment])
            if not scores.empty:
                mean, ci_lower, ci_upper = calculate_bootstrap_ci(scores)
                ci_data.append({
                    'group': group,
                    'sentiment': sentiment,
                    'mean': mean,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper
                })
    
    # After collecting sentiment scores and before creating the plot, add significance testing
    # Calculate p-values comparing each treatment group to control for each sentiment
    significance_data = {}
    if 'control' in groups:
        for group in groups:
            if group != 'control':
                significance_data[group] = {}
                for sentiment in ['negative', 'neutral', 'positive']:
                    control_scores = pd.Series(sentiment_data['control'][sentiment])
                    treatment_scores = pd.Series(sentiment_data[group][sentiment])
                    if not control_scores.empty and not treatment_scores.empty:
                        p_value = calculate_p_value(treatment_scores, control_scores)
                        significance_data[group][sentiment] = get_stars(p_value)
    
    # Modify the plot creation to add significance stars
    fig = go.Figure()
    
    # Find the maximum y-value to position stars
    max_y = 0
    for d in ci_data:
        max_y = max(max_y, d['ci_upper'])
    
    # Add traces for each group
    for group in groups:
        group_data = pd.DataFrame([d for d in ci_data if d['group'] == group])
        
        fig.add_trace(go.Scatter(
            x=[sentiments.index(s) + group_offsets.get(group, 0) for s in group_data['sentiment']],
            y=group_data['mean'],
            error_y=dict(
                type='data',
                symmetric=False,
                array=group_data['ci_upper'] - group_data['mean'],
                arrayminus=group_data['mean'] - group_data['ci_lower'],
                width=10,
            ),
            mode='markers',
            marker=dict(size=10, color=colors.get(group, '#000000')),
            name=group,
            showlegend=True
        ))
        
        # Add significance stars for treatment groups
        if group != 'control' and group in significance_data:
            for sentiment in sentiments:
                if sentiment in significance_data[group]:
                    stars = significance_data[group][sentiment]
                    x_pos = sentiments.index(sentiment) + group_offsets[group]
                    y_pos = group_data[group_data['sentiment'] == sentiment]['ci_upper'].iloc[0]
                    
                    fig.add_annotation(
                        x=x_pos,
                        y=y_pos + (max_y * 0.05),  # Position stars slightly above the error bar
                        text=stars,
                        showarrow=False,
                        font=dict(size=12)
                    )
    
    # Update layout to add space for stars and add significance legend
    fig.update_layout(
        title='Sentiment Analysis by Group (with 95% Bootstrap CI)',
        xaxis=dict(
            ticktext=sentiments,
            tickvals=list(range(len(sentiments))),
            title="Sentiment Type"
        ),
        yaxis=dict(
            title="Sentiment Score",
            range=[0, max_y * 1.15]  # Add 15% space for significance stars
        ),
        height=500,
        template='plotly_white',
        showlegend=True,
        legend_title_text='Group'
    )
    
    # Add significance legend
    fig.add_annotation(
        xref='paper',
        yref='paper',
        x=1.15,
        y=1.0,
        text='*p<0.1, **p<0.05, ***p<0.01<br>ns: not significant',
        showarrow=False,
        font=dict(size=10),
        xanchor='right',
        yanchor='top'
    )
    
    return fig


@st.cache_resource
def load_toxicity_pipeline():
    model_name = "tomh/toxigen_roberta"
    toxicity_task = pipeline("text-classification", model=model_name, tokenizer=model_name)
    return toxicity_task

@st.cache_data
def analyze_toxicity_batch(texts):
    """Analyze toxicity for a batch of texts"""
    toxicity_task = load_toxicity_pipeline()
    try:
        results = toxicity_task(texts, return_all_scores=True)
        return results
    except Exception as e:
        st.error(f"Error in batch toxicity analysis: {str(e)}")
        return None

def create_toxicity_plot(processed_data):
    """
    Create confidence interval plot comparing toxicity scores across all groups.
    """
    # Colors for consistency
    colors = {
        'control': '#1f77b4',  # blue
        'chat_5': '#2ca02c',   # green
        'control': '#1f77b4',   # blue
        'treatment': '#2ca02c',  # green (for a combined treatment)
        'chat_5': '#2ca02c',
        'conversation_5': '#ff7f0e',  # orange
        'feedback_5': '#d62728',  # red
        'suggestions_5': '#9467bd'  # purple
    }
    
    # Determine group ordering. If 'control' exists, place it first.
    groups = sorted(processed_data.keys())
    if 'control' in groups:
        groups.remove('control')
        groups = ['control'] + groups
    
    # Define offsets for each group.
    group_offsets = {
        'control': -0.2,
        'treatment': -0.1,  # offset for a combined treatment group
        'chat_5': -0.1,
        'conversation_5': 0,
        'feedback_5': 0.1,
        'suggestions_5': 0.2
    }
    
    # Initialize data structure for toxicity scores.
    toxicity_data = {group: {'not_toxic': [], 'toxic': []} for group in groups}
    
    # Count total comments.
    total_comments = 0
    for group_data in processed_data.values():
        for game in group_data.values():
            for round_comments in game['comments']:
                total_comments += len(round_comments)
    
    # Process comments for each group.
    progress_bar = st.progress(0)
    status_placeholder = st.empty()
    comments_processed = 0
    
    for group_type in groups:
        group_data = processed_data[group_type]
        all_texts = [comment['content'] for game in group_data.values() 
                     for round_comments in game['comments'] 
                     for comment in round_comments]
        
        batch_size = len(all_texts)
        if batch_size > 0:
            percentage = min(100, round((comments_processed / total_comments) * 100))
            status_placeholder.write(
                f"Processing {batch_size} comments for {group_type}...\n"
                f"Progress: {percentage}% ({comments_processed}/{total_comments} comments processed)"
            )
            
            results = analyze_toxicity_batch(all_texts)
            if results:
                for text_results in results:
                    for score_dict in text_results:
                        label = 'not_toxic' if score_dict['label'] == 'LABEL_0' else 'toxic'
                        toxicity_data[group_type][label].append(score_dict['score'])
            
            comments_processed += batch_size
            progress_bar.progress(min(1.0, comments_processed / total_comments))
    
    status_placeholder.empty()
    st.success('Toxicity analysis completed!')
    
    # Calculate CIs for each toxicity type.
    ci_data = []
    toxicity_types = ['not_toxic', 'toxic']
    
    for group in groups:
        for tox_type in toxicity_types:
            scores = pd.Series(toxicity_data[group][tox_type])
            if not scores.empty:
                mean, ci_lower, ci_upper = calculate_bootstrap_ci(scores)
                ci_data.append({
                    'group': group,
                    'toxicity': tox_type,
                    'mean': mean,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper
                })
    
    # Create the plot.
    fig = go.Figure()
    
    # Add traces for each group.
    for group in groups:
        group_data = pd.DataFrame([d for d in ci_data if d['group'] == group])
        
        fig.add_trace(go.Scatter(
            x=[toxicity_types.index(t) + group_offsets.get(group, 0) for t in group_data['toxicity']],
            y=group_data['mean'],
            error_y=dict(
                type='data',
                symmetric=False,
                array=group_data['ci_upper'] - group_data['mean'],
                arrayminus=group_data['mean'] - group_data['ci_lower'],
                width=10,
            ),
            mode='markers',
            marker=dict(size=10, color=colors.get(group, '#000000')),
            name=group,
            showlegend=True
        ))
    
    fig.update_layout(
        title='Toxicity Analysis by Group (with 95% Bootstrap CI)',
        xaxis=dict(
            ticktext=['Not Toxic', 'Toxic'],
            tickvals=[0, 1],
            title="Toxicity Type"
        ),
        yaxis=dict(
            title="Toxicity Score",
            range=[0, 1]  # Set fixed range from 0 to 1
        ),
        height=500,
        template='plotly_white',
        showlegend=True,
        legend_title_text='Group'
    )
    
    return fig

def calculate_bootstrap_ci(data: pd.Series, confidence: float = 0.95, n_bootstrap: int = 1000) -> tuple:
    """Calculate mean and confidence interval bounds using bootstrapping"""
    mean = data.mean()
    
    # Bootstrap sampling
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = data.sample(n=len(data), replace=True)
        bootstrap_means.append(sample.mean())
    
    # Calculate confidence interval
    alpha = (1 - confidence) / 2
    ci_lower, ci_upper = np.percentile(bootstrap_means, [alpha * 100, (1 - alpha) * 100])
    
    return mean, ci_lower, ci_upper

def calculate_p_value(treatment_data, control_data, n_bootstrap=10000):
    """Calculate p-value using t-test on original data"""
    from scipy import stats
    
    # Perform t-test on the original data
    _, p_value = stats.ttest_ind(treatment_data, control_data)
    
    return p_value

def get_stars(p_value):
    """Convert p-value to significance stars"""
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    elif p_value < 0.1:
        return "†"
    else:
        return "ns"