import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import plotly.figure_factory as ff
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from engagement_helpers.visuals import calculate_bootstrap_ci, calculate_p_value, get_stars
from players import bootstrap_mean_ci
from players_helpers.ate import calculate_cohens_d
import json
import os

def display_comparative_metrics(processed_data):
    """
    Display comparative metrics for all treatment groups and control.
    
    Args:
        processed_data (dict): Dictionary containing processed data for all groups
        
    Returns:
        tuple: (total_metrics, avg_metrics) Dictionaries containing calculated metrics
    """
    # Initialize metrics dictionaries for all groups
    total_metrics = {group: {} for group in processed_data.keys()}
    avg_metrics = {group: {} for group in processed_data.keys()}
    
    # Calculate metrics for each group
    for group, group_data in processed_data.items():
        # Total aggregate numbers
        total_games = len(group_data)
        total_actions = sum(
            len(actions)
            for game in group_data.values()
            for actions in game['actions']
        )
        total_comments = sum(
            len(comments)
            for game in group_data.values()
            for comments in game['comments']
        )
        
        # Calculate action/comment ratio
        action_comment_ratio = total_actions / total_comments if total_comments > 0 else 0
        
        # Calculate overall reactions per comment and round-specific metrics
        all_comment_reactions = []
        round_comment_reactions = {i: [] for i in range(3)}
        round_actions = {i: [] for i in range(3)}
        round_comments = {i: [] for i in range(3)}
        
        for game in group_data.values():
            for round_num in range(3):
                comments = game['comments'][round_num]
                actions = game['actions'][round_num]
                
                # Store per-round counts
                round_actions[round_num].append(len(actions))
                round_comments[round_num].append(len(comments))
                
                # Count reactions for each comment in this round
                reactions_count = {comment['content_id']: 0 for comment in comments}
                for action in actions:
                    if action['content_id'] in reactions_count:
                        reactions_count[action['content_id']] += 1
                round_reactions = list(reactions_count.values())
                round_comment_reactions[round_num].extend(round_reactions)
                all_comment_reactions.extend(round_reactions)
        
        avg_reactions_per_comment = np.mean(all_comment_reactions) if all_comment_reactions else 0
        
        # Accumulate unique users from all rounds for a per-user metric.
        unique_users = set()
        for game in group_data.values():
            for round_num in range(3):
                unique_users.update(comment['user_id'] for comment in game['comments'][round_num])
        comments_per_user_round = (total_comments / (len(unique_users) * 3)) if unique_users else 0
        
        total_metrics[group] = {
            'games': total_games,
            'actions': total_actions,
            'avg_actions_per_game': total_actions / total_games if total_games > 0 else 0,
            'comments': total_comments,
            'avg_comments_per_game': total_comments / total_games if total_games > 0 else 0,
            'ratio': action_comment_ratio,
            'avg_reactions_per_comment': avg_reactions_per_comment,
            'comments_per_user_round': comments_per_user_round
        }
        
        # Compute round averages along with bootstrap confidence intervals instead of std.
        avg_metrics[group] = {
            'actions': {
                i: {
                    'avg': np.mean(round_actions[i]),
                    'ci_lower': calculate_bootstrap_ci(pd.Series(round_actions[i]))[1] if round_actions[i] else 0,
                    'ci_upper': calculate_bootstrap_ci(pd.Series(round_actions[i]))[2] if round_actions[i] else 0,
                } for i in range(3)
            },
            'comments': {
                i: {
                    'avg': np.mean(round_comments[i]),
                    'ci_lower': calculate_bootstrap_ci(pd.Series(round_comments[i]))[1] if round_comments[i] else 0,
                    'ci_upper': calculate_bootstrap_ci(pd.Series(round_comments[i]))[2] if round_comments[i] else 0,
                } for i in range(3)
            },
            'reactions_per_comment': {
                i: {
                    'avg': np.mean(round_comment_reactions[i]) if round_comment_reactions[i] else 0,
                    'ci_lower': calculate_bootstrap_ci(pd.Series(round_comment_reactions[i]))[1] if round_comment_reactions[i] else 0,
                    'ci_upper': calculate_bootstrap_ci(pd.Series(round_comment_reactions[i]))[2] if round_comment_reactions[i] else 0,
                } for i in range(3)
            }
        }
    
    def calc_percentage_diff(treatment_val, control_val):
        if control_val == 0:
            return float('inf') if treatment_val > 0 else 0
        return ((treatment_val - control_val) / control_val) * 100

    # Display aggregate metrics
    st.subheader("Aggregate Numbers")
    metrics = ["Games", "Actions (avg)", "Comments (avg)", "Action/Comment Ratio", "Avg Reactions per Comment", "Comments per User/Round"]
    cols = st.columns(len(metrics))
    
    groups = sorted(processed_data.keys())
    groups.remove('control')
    groups = ['control'] + groups
    
    for col, metric in zip(cols, metrics):
        with col:
            st.markdown(f"**{metric}**")
            control_value = None
            if metric == "Games":
                control_value = total_metrics['control']['games']
                st.metric("control", control_value)
            elif metric == "Actions (avg)":
                control_value = total_metrics['control']['avg_actions_per_game']
                st.metric("control", f"{control_value:.1f}")
            elif metric == "Comments (avg)":
                control_value = total_metrics['control']['avg_comments_per_game']
                st.metric("control", f"{control_value:.1f}")
            elif metric == "Action/Comment Ratio":
                control_value = total_metrics['control']['ratio']
                st.metric("control", f"{control_value:.2f}")
            elif metric == "Avg Reactions per Comment":
                control_value = total_metrics['control']['avg_reactions_per_comment']
                st.metric("control", f"{control_value:.2f}")
            elif metric == "Comments per User/Round":
                control_value = total_metrics['control']['comments_per_user_round']
                st.metric("control", f"{control_value:.2f}")
            
            for group in groups:
                if group != 'control':
                    if metric == "Games":
                        value = total_metrics[group]['games']
                    elif metric == "Actions (avg)":
                        value = total_metrics[group]['avg_actions_per_game']
                    elif metric == "Comments (avg)":
                        value = total_metrics[group]['avg_comments_per_game']
                    elif metric == "Action/Comment Ratio":
                        value = total_metrics[group]['ratio']
                    elif metric == "Avg Reactions per Comment":
                        value = total_metrics[group]['avg_reactions_per_comment']
                    elif metric == "Comments per User/Round":
                        value = total_metrics[group]['comments_per_user_round']
                    
                    pct_diff = calc_percentage_diff(value, control_value)
                    st.metric(group, f"{value:.2f}" if isinstance(value, float) else value, delta=f"{pct_diff:+.1f}%")
    
    st.subheader("Round-specific Metrics")
    round_data = []
    for round_num in range(3):
        round_dict = {'Round': f'Round {round_num}'}
        round_dict.update({
            'control Actions': f"{avg_metrics['control']['actions'][round_num]['avg']:.2f} ± [{avg_metrics['control']['actions'][round_num]['ci_lower']:.2f}, {avg_metrics['control']['actions'][round_num]['ci_upper']:.2f}]",
            'control Comments': f"{avg_metrics['control']['comments'][round_num]['avg']:.2f} ± [{avg_metrics['control']['comments'][round_num]['ci_lower']:.2f}, {avg_metrics['control']['comments'][round_num]['ci_upper']:.2f}]",
            'control Reactions per Comment': f"{avg_metrics['control']['reactions_per_comment'][round_num]['avg']:.2f} ± [{avg_metrics['control']['reactions_per_comment'][round_num]['ci_lower']:.2f}, {avg_metrics['control']['reactions_per_comment'][round_num]['ci_upper']:.2f}]"
        })
        for group in groups:
            if group != 'control':
                round_dict.update({
                    f'{group} Actions': f"{avg_metrics[group]['actions'][round_num]['avg']:.2f} ± [{avg_metrics[group]['actions'][round_num]['ci_lower']:.2f}, {avg_metrics[group]['actions'][round_num]['ci_upper']:.2f}]",
                    f'{group} Comments': f"{avg_metrics[group]['comments'][round_num]['avg']:.2f} ± [{avg_metrics[group]['comments'][round_num]['ci_lower']:.2f}, {avg_metrics[group]['comments'][round_num]['ci_upper']:.2f}]",
                    f'{group} Reactions per Comment': f"{avg_metrics[group]['reactions_per_comment'][round_num]['avg']:.2f} ± [{avg_metrics[group]['reactions_per_comment'][round_num]['ci_lower']:.2f}, {avg_metrics[group]['reactions_per_comment'][round_num]['ci_upper']:.2f}]"
                })
        round_data.append(round_dict)
    
    df_rounds = pd.DataFrame(round_data)
    st.dataframe(
        df_rounds.style.set_properties(**{'text-align': 'center'}),
        hide_index=True
    )
    
    return total_metrics, avg_metrics

def shannon_entropy_comparison(processed_data):
    """
    Calculate and display Shannon entropy comparison between control and all treatment groups.
    
    Args:
        processed_data (dict): Dictionary containing processed data for all groups
        
    Returns:
        dict: Dictionary containing entropy metrics and statistical tests
    """
    # Store entropies for each group
    entropies = {group: [] for group in processed_data.keys()}
    
    # Calculate entropies for each group
    for group, group_data in processed_data.items():
        for game_id, game_data in group_data.items():
            # Calculate entropy for each round
            for round_num in range(3):
                # Get unique users for this specific round
                round_users = set()
                round_users.update(comment['user_id'] for comment in game_data['comments'][round_num])
                round_users.update(action['user_id'] for action in game_data['actions'][round_num])
                n_participants = len(round_users)

                if n_participants < 3:
                    continue
                
                # Calculate user comment counts for this round
                user_comments = {user_id: 0 for user_id in round_users}
                for comment in game_data['comments'][round_num]:
                    user_comments[comment['user_id']] += 1
                
                comment_counts = list(user_comments.values())
                round_entropy = calculate_normalized_entropy(comment_counts, n_participants)
                entropies[group].append(round_entropy)
    
    
    # Display results
    st.header("Shannon Entropy Analysis")
    st.markdown("""Comparing the distribution of comment participation using normalized Shannon entropy. 
                In basic terms, the entropy for a given round will indicate how evenly the comments are distributed across the participants - a high score indicate that more participant contribute with comments, while a low score indicates that contributions are dominated by one or a few participants. 
                Ideally, we would like this number to be higher for the treatments compared to the control.""")
    
    # Create box plot using plotly
    fig = go.Figure()
    
    # Ensure control is first, then sort other groups
    groups = sorted(processed_data.keys())
    groups.remove('control')
    groups = ['control'] + groups
    
    # Add traces in order (control first, then other groups)
    for group in groups:
        fig.add_trace(go.Box(
            y=entropies[group],
            name=group,
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8
        ))
    
    fig.update_layout(
        title='Distribution of Normalized Shannon Entropy',
        yaxis_title='Normalized Shannon Entropy',
        showlegend=True,
        boxmode='group'
    )
    
    st.plotly_chart(fig)
    
    # Display summary statistics and test results in columns
    cols = st.columns(len(groups))
    
    # Display control first, then other groups
    for col, group in zip(cols, groups):
        with col:
            st.markdown(f"**{group}**")
            st.write(f"Mean: {np.mean(entropies[group]):.3f}")
            st.write(f"Median: {np.median(entropies[group]):.3f}")
            st.write(f"Std: {np.std(entropies[group]):.3f}")

    st.divider()
    
    return entropies

def calculate_normalized_entropy(comment_counts, n_participants):
    """Helper function to calculate normalized Shannon entropy using log base 2"""
    total_comments = sum(comment_counts)
    if total_comments == 0:
        return 0
            
    probabilities = np.array(comment_counts) / total_comments
    # Calculate entropy using log2 (terms with p=0 automatically contribute 0)
    entropy = -np.sum(probabilities[probabilities > 0] * np.log2(probabilities[probabilities > 0]))
    # Normalize by log2(n)
    normalized_entropy = entropy / np.log2(n_participants) if n_participants > 1 else 0
    return normalized_entropy


def shannon_entropy_comparison_pointplot(processed_data):
    """
    Calculate and display Shannon entropy comparison between control and all treatment groups
    using a plotly point plot.
    """
    # Store entropies for each group
    entropies = {group: [] for group in processed_data.keys()}
    
    # Calculate entropies for each group
    for group, group_data in processed_data.items():
        for game_id, game_data in group_data.items():
            # Calculate entropy for each round
            for round_num in range(3):
                # Get unique users for this specific round
                round_users = set()
                round_users.update(comment['user_id'] for comment in game_data['comments'][round_num])
                round_users.update(action['user_id'] for action in game_data['actions'][round_num])
                n_participants = len(round_users)

                if n_participants < 3:
                    continue
                
                # Calculate user comment counts for this round
                user_comments = {user_id: 0 for user_id in round_users}
                for comment in game_data['comments'][round_num]:
                    user_comments[comment['user_id']] += 1
                
                comment_counts = list(user_comments.values())
                round_entropy = calculate_normalized_entropy(comment_counts, n_participants)
                entropies[group].append(round_entropy)
    
    # Ensure control is first, then sort other groups
    groups = sorted(processed_data.keys())
    groups.remove('control')
    groups = ['control'] + groups
    
    # Calculate means and confidence intervals for each group
    means = []
    errors = []
    conf_intervals = []
    
    np.random.seed(42)

    for group in groups:
        # Use bootstrap_mean_ci function for consistent calculations
        data = np.array(entropies[group])
        mean, ci_lower, ci_upper = bootstrap_mean_ci(pd.Series(data))
        
        means.append(mean)
        conf_intervals.append([ci_lower, ci_upper])
        # Use the larger of the two differences from the mean for the error bar
        error = max(abs(mean - ci_lower), abs(ci_upper - mean))
        errors.append(error)
    
    # Calculate significance vs control using t-test and Cohen's d
    control_data = np.array(entropies['control'])
    significance = ["ns"]  # First element is control group
    p_values = [None]  # First element is control group
    cohens_d_values = [0.0]  # First element is control group (reference)
    effect_interpretations = ["reference"]  # First element is control group

    for group in groups[1:]:  # Skip control group
        treatment_data = np.array(entropies[group])
        p_value = calculate_p_value(treatment_data, control_data)
        stars = get_stars(p_value)
        significance.append(stars)
        p_values.append(p_value)

        # Calculate Cohen's d
        effect = calculate_cohens_d(list(treatment_data), list(control_data))
        cohens_d_values.append(effect['d'])
        effect_interpretations.append(effect['interpretation'])

    # Create output data structure
    output_data = {
        "shannon_entropy": {
            "x_vals": list(range(len(groups))),
            "y_vals": means,
            "err": errors,
            "significance": significance,
            "p_values": p_values,
            "cohens_d": cohens_d_values,
            "effect_interpretation": effect_interpretations,
            "n_values": [len(entropies[group]) for group in groups],
            "treatment_labels": groups,
            "question_label": "Distribution of Normalized Shannon Entropy"
        }
    }
    
    # Create directory if it doesn't exist
    os.makedirs('data/plotting', exist_ok=True)
    
    # Save the data
    with open('data/shannon_entropy_data.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Create plotly figure
    fig = go.Figure()
    
    # Add error bars
    fig.add_trace(go.Scatter(
        x=groups,
        y=means,
        error_y=dict(
            type='data',
            array=errors,
            visible=True,
            thickness=2,
            width=15,
            color='rgba(31, 119, 180, 0.8)'
        ),
        mode='markers',
        marker=dict(
            size=12,
            color='rgb(31, 119, 180)',
            line=dict(
                width=2,
                color='rgb(8, 48, 107)'
            )
        ),
        name='Group Means'
    ))
    
    # Add significance markers with p-values and Cohen's d
    max_y = max([m + e for m, e in zip(means, errors)])

    for i, group in enumerate(groups[1:], 1):  # Skip control group
        # Get p-value using t-test
        treatment_data = np.array(entropies[group])
        n_treatment = len(treatment_data)
        mean_val = means[i]
        ci_lower, ci_upper = conf_intervals[i]
        p_value = calculate_p_value(treatment_data, control_data)
        stars = get_stars(p_value)

        # Calculate Cohen's d with CI
        effect = calculate_cohens_d(list(treatment_data), list(control_data))
        d_val = effect['d']
        d_ci_lower = effect['ci_lower']
        d_ci_upper = effect['ci_upper']
        effect_interp = effect['interpretation']

        # Build annotation text with mean, CI, p-value, and Cohen's d
        annotation_text = f"n={n_treatment}<br>μ={mean_val:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]<br>p={p_value:.3f} {stars}"
        if d_val is not None and d_ci_lower is not None and d_ci_upper is not None:
            annotation_text += f"<br>d={d_val:.2f} [{d_ci_lower:.2f}, {d_ci_upper:.2f}] ({effect_interp})"
        elif d_val is not None:
            annotation_text += f"<br>d={d_val:.2f} ({effect_interp})"

        # Add significance marker
        fig.add_annotation(
            x=group,
            y=max_y,
            text=annotation_text,
            showarrow=False,
            yshift=45,
            font=dict(size=9),
            align='center'
        )

    # Add annotation for control group with mean and CI
    n_control = len(control_data)
    control_mean = means[0]
    control_ci_lower, control_ci_upper = conf_intervals[0]
    fig.add_annotation(
        x=groups[0],
        y=max_y,
        text=f"n={n_control}<br>μ={control_mean:.2f} [{control_ci_lower:.2f}, {control_ci_upper:.2f}]",
        showarrow=False,
        yshift=35,
        font=dict(size=9),
        align='center'
    )

    # Add legend for significance levels
    fig.add_annotation(
        xref='paper',
        yref='paper',
        x=1.15,
        y=1.0,
        text='†p<0.1, *p<0.05, **p<0.01, ***p<0.001, ns: not significant',
        showarrow=False,
        font=dict(size=10),
        xanchor='right',
        yanchor='top'
    )
    
    # Update layout
    fig.update_layout(
        title='Distribution of Normalized Shannon Entropy (95% CI)',
        yaxis=dict(
            title='Normalized Shannon Entropy',
            range=[0.5, max_y + 0.20],  # Increased to accommodate annotations with N, p-value, and Cohen's d
            gridcolor='rgba(128, 128, 128, 0.2)',
            gridwidth=0.5,
            griddash='dash'
        ),
        xaxis=dict(
            title='',
            showgrid=False
        ),
        showlegend=False,
        margin=dict(
            t=50,
            l=50,
            r=150,
            b=50
        )
    )
    
    # Display the plot in Streamlit
    st.plotly_chart(fig)
    
    # Perform statistical tests against control
    stats_results = {}
    control_entropies = entropies['control']
    for group in processed_data.keys():
        if group != 'control':
            # T-test
            t_stat, t_pval = stats.ttest_ind(control_entropies, entropies[group])
            # Mann-Whitney U test
            mw_stat, mw_pval = stats.mannwhitneyu(control_entropies, entropies[group], alternative='two-sided')
            
            stats_results[group] = {
                't_test': {'statistic': t_stat, 'p_value': t_pval},
                'mann_whitney': {'statistic': mw_stat, 'p_value': mw_pval}
            }

    st.divider()
    
    return stats_results