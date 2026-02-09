from difflib import SequenceMatcher
from scipy import stats
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import numpy as np
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from scipy.stats import chi2_contingency

def display_topic_metrics(data, display: bool = True):
    """Display topic metrics across all games.
    
    Args:
        data (dict): df_generations_matched contains all the information needed
        display (bool): Whether to display the metrics
        
    Returns:
        dict: Dictionary containing calculated metrics
    """
    # Get all treatment groups (excluding control)
    treatment_groups = [group for group in data.keys() if group != 'control']

    topics_content = {
        'Cats': {
            'actions': {'control': []},
            'comments': {'control': []}
        },
        'Oats': {
            'actions': {'control': []},
            'comments': {'control': []}
        },
        'Politics': {
            'actions': {'control': []},
            'comments': {'control': []}
        }
    }
    
    # Initialize treatment groups in topics_content
    for topic in topics_content:
        for group in treatment_groups:
            topics_content[topic]['actions'][group] = []
            topics_content[topic]['comments'][group] = []

    total_games = {'control': 0}
    for group in treatment_groups:
        total_games[group] = 0

    # Process data for control and all treatment groups
    for group in ['control'] + treatment_groups:
        group_data = data[group]
        if not isinstance(group_data, dict):
            continue

        for game_id, game_data in group_data.items():
            total_games[group] += 1
            topic_order = game_data['topic_order']

            # Handle topic_order whether it's a list or dict
            if isinstance(topic_order, dict):
                topic_order_items = topic_order.items()
            else:
                # If it's a list, create enumerate items
                topic_order_items = enumerate(topic_order)

            for order, topic in topic_order_items:
                # Normalize topic name to match our structure
                topic = topic.capitalize()
                if topic in topics_content:
                    topics_content[topic]['actions'][group].extend(game_data['actions'][order])
                    topics_content[topic]['comments'][group].extend(game_data['comments'][order])

    if display:
        topic_cols = st.columns(3)

        for idx, topic in enumerate(topics_content):
            with topic_cols[idx]:
                st.subheader(topic)
                
                # Create two columns within each topic column
                metric_cols = st.columns(2)
                
                # Left column for Comments
                with metric_cols[0]:
                    st.markdown("**💬 Comments**")
                    
                    # Display control metrics first
                    control_comments = topics_content[topic]['comments']['control']
                    total_control_comments = len(control_comments)
                    avg_control_comments = total_control_comments / total_games['control'] if total_games['control'] > 0 else 0
                    
                    st.markdown("*Control*")
                    st.metric(
                        "Total",
                        total_control_comments,
                        delta=f"Avg: {avg_control_comments:.1f}",
                        delta_color="off"
                    )
                    
                    # Display metrics for each treatment group
                    total_all_comments = total_control_comments
                    for group in treatment_groups:
                        treatment_comments = topics_content[topic]['comments'][group]
                        total_treatment_comments = len(treatment_comments)
                        total_all_comments += total_treatment_comments
                        
                        avg_treatment_comments = total_treatment_comments / total_games[group] if total_games[group] > 0 else 0
                        avg_comments_diff_pct = ((avg_treatment_comments - avg_control_comments) / avg_control_comments * 100) if avg_control_comments > 0 else 0
                        
                        delta_color = "normal" if avg_comments_diff_pct >= 0 else "inverse"

                        st.markdown(f"*{group}*")
                        st.metric(
                            "Total",
                            total_treatment_comments,
                            delta=f"Avg: {avg_treatment_comments:.1f} ({avg_comments_diff_pct:+.1f}% vs C)",
                            delta_color=delta_color
                        )
                    
                    st.markdown("*Total All Groups*")
                    st.metric("Total", total_all_comments)
                
                # Right column for Actions/Reactions
                with metric_cols[1]:
                    st.markdown("**👍 Actions**")
                    
                    # Display control metrics first
                    control_actions = topics_content[topic]['actions']['control']
                    total_control_actions = len(control_actions)
                    avg_control_actions = total_control_actions / total_games['control'] if total_games['control'] > 0 else 0
                    
                    st.markdown("*Control*")
                    st.metric(
                        "Total",
                        total_control_actions,
                        delta=f"Avg: {avg_control_actions:.1f}",
                        delta_color="off"
                    )
                    
                    # Display metrics for each treatment group
                    total_all_actions = total_control_actions
                    for group in treatment_groups:
                        treatment_actions = topics_content[topic]['actions'][group]
                        total_treatment_actions = len(treatment_actions)
                        total_all_actions += total_treatment_actions
                        
                        avg_treatment_actions = total_treatment_actions / total_games[group] if total_games[group] > 0 else 0
                        avg_actions_diff_pct = ((avg_treatment_actions - avg_control_actions) / avg_control_actions * 100) if avg_control_actions > 0 else 0
                        
                        # Determine color based on the percentage difference
                        delta_color = "normal" if avg_actions_diff_pct >= 0 else "inverse"
                        
                        st.markdown(f"*{group}*")
                        st.metric(
                            "Total",
                            total_treatment_actions,
                            delta=f"Avg: {avg_treatment_actions:.1f} ({avg_actions_diff_pct:+.1f}% vs C)",
                            delta_color=delta_color
                        )
                    
                    st.markdown("*Total All Groups*")
                    st.metric("Total", total_all_actions)
                
                st.divider()

                # Calculate and display average actions per comment for all groups
                st.subheader(topic)
                st.markdown(f"**📊 Average Actions per Comment**")
                
                # Control group
                avg_control_actions_per_comment = avg_control_actions / avg_control_comments if avg_control_comments > 0 else 0
                st.metric(
                    "Control",
                    f"{avg_control_actions_per_comment:.2f}"
                )
                
                # Treatment groups
                for group in treatment_groups:
                    avg_treatment_actions = len(topics_content[topic]['actions'][group]) / total_games[group] if total_games[group] > 0 else 0
                    avg_treatment_comments = len(topics_content[topic]['comments'][group]) / total_games[group] if total_games[group] > 0 else 0
                    
                    avg_treatment_actions_per_comment = avg_treatment_actions / avg_treatment_comments if avg_treatment_comments > 0 else 0
                    actions_per_comment_diff_pct = ((avg_treatment_actions_per_comment - avg_control_actions_per_comment) / avg_control_actions_per_comment * 100) if avg_control_actions_per_comment > 0 else 0
                    
                    st.metric(
                        group,
                        f"{avg_treatment_actions_per_comment:.2f}",
                        delta=f"{actions_per_comment_diff_pct:+.1f}% vs C",
                        delta_color="normal"
                    )

        # After showing all metrics, add bootstrap analysis and plots
        st.divider()
        
        # Helper function for bootstrap
        def bootstrap_mean(data, n_bootstrap=1000):
            """Calculate mean and confidence intervals using bootstrap"""
            if not data:  # Guard against empty data
                return 0, 0, 0
            
            means = []
            for _ in range(n_bootstrap):
                sample = np.random.choice(data, size=len(data), replace=True)
                means.append(np.mean(sample))
            
            mean = np.mean(data)  # Use actual mean, not bootstrap mean
            ci_lower, ci_upper = np.percentile(means, [2.5, 97.5])
            return mean, ci_lower, ci_upper

        def get_significance_stars(p_value):
            """Convert p-value to significance stars"""
            if p_value < 0.001:
                return "***"
            elif p_value < 0.01:
                return "**"
            elif p_value < 0.05:
                return "*"
            elif p_value < 0.1:
                return "†"
            return "ns"  # Changed from "" to "ns" for consistency
        
        def calculate_p_value(control_data, treatment_data):
            """Calculate p-value using t-test on original data"""
            if not control_data or not treatment_data:
                return 1.0
            
            # Perform t-test on the original data
            _, p_value = stats.ttest_ind(control_data, treatment_data)
            
            return p_value

        # Prepare data structure for per-user metrics
        topic_user_data = {topic: {'control': {'comments': [], 'actions': []}} for topic in topics_content}
        for topic in topic_user_data:
            for group in treatment_groups:
                topic_user_data[topic][group] = {'comments': [], 'actions': []}

        # Collect per-user data for each topic and group
        for group in ['control'] + treatment_groups:
            group_data = data[group]
            if not isinstance(group_data, dict):
                continue
                
            for game_id, game_data in group_data.items():
                topic_order = game_data['topic_order']
                
                # Handle topic_order whether it's a list or dict
                if isinstance(topic_order, dict):
                    topic_mapping = {idx: topic.capitalize() for topic, idx in topic_order.items()}
                else:
                    topic_mapping = {idx: topic.capitalize() for idx, topic in enumerate(topic_order)}
                
                # Collect per-user data for each topic
                for idx, topic in topic_mapping.items():
                    if topic not in topic_user_data:
                        continue
                        
                    # Get comments per user
                    user_comments = {}
                    for comment in game_data['comments'][idx]:
                        user_id = comment['user_id']
                        user_comments[user_id] = user_comments.get(user_id, 0) + 1
                    
                    # Get actions per user
                    user_actions = {}
                    for action in game_data['actions'][idx]:
                        user_id = action['user_id']
                        user_actions[user_id] = user_actions.get(user_id, 0) + 1
                    
                    # Add to our data structure
                    topic_user_data[topic][group]['comments'].extend(user_comments.values())
                    topic_user_data[topic][group]['actions'].extend(user_actions.values())

        # Prepare plot data for per-user metrics
        plot_data_per_user = {
            'comments': {'group': [], 'topic': [], 'mean': [], 'ci_lower': [], 'ci_upper': [], 'stars': []},
            'actions': {'group': [], 'topic': [], 'mean': [], 'ci_lower': [], 'ci_upper': [], 'stars': []}
        }

        # Calculate statistics and significance for per-user metrics
        for topic in topic_user_data:
            # Process control group first
            control_comments = topic_user_data[topic]['control']['comments']
            control_actions = topic_user_data[topic]['control']['actions']
            
            # Add control group data
            mean_comments, ci_lower, ci_upper = bootstrap_mean(control_comments)
            plot_data_per_user['comments']['group'].append('control')
            plot_data_per_user['comments']['topic'].append(topic)
            plot_data_per_user['comments']['mean'].append(mean_comments)
            plot_data_per_user['comments']['ci_lower'].append(ci_lower)
            plot_data_per_user['comments']['ci_upper'].append(ci_upper)
            plot_data_per_user['comments']['stars'].append('')
            
            mean_actions, ci_lower, ci_upper = bootstrap_mean(control_actions)
            plot_data_per_user['actions']['group'].append('control')
            plot_data_per_user['actions']['topic'].append(topic)
            plot_data_per_user['actions']['mean'].append(mean_actions)
            plot_data_per_user['actions']['ci_lower'].append(ci_lower)
            plot_data_per_user['actions']['ci_upper'].append(ci_upper)
            plot_data_per_user['actions']['stars'].append('')
            
            # Process treatment groups
            for group in treatment_groups:
                treatment_comments = topic_user_data[topic][group]['comments']
                treatment_actions = topic_user_data[topic][group]['actions']
                
                # Comments
                mean_comments, ci_lower, ci_upper = bootstrap_mean(treatment_comments)
                p_value = calculate_p_value(control_comments, treatment_comments)
                stars = get_significance_stars(p_value)
                
                plot_data_per_user['comments']['group'].append(group)
                plot_data_per_user['comments']['topic'].append(topic)
                plot_data_per_user['comments']['mean'].append(mean_comments)
                plot_data_per_user['comments']['ci_lower'].append(ci_lower)
                plot_data_per_user['comments']['ci_upper'].append(ci_upper)
                plot_data_per_user['comments']['stars'].append(stars)
                
                # Actions
                mean_actions, ci_lower, ci_upper = bootstrap_mean(treatment_actions)
                p_value = calculate_p_value(control_actions, treatment_actions)
                stars = get_significance_stars(p_value)
                
                plot_data_per_user['actions']['group'].append(group)
                plot_data_per_user['actions']['topic'].append(topic)
                plot_data_per_user['actions']['mean'].append(mean_actions)
                plot_data_per_user['actions']['ci_lower'].append(ci_lower)
                plot_data_per_user['actions']['ci_upper'].append(ci_upper)
                plot_data_per_user['actions']['stars'].append(stars)

        # Add new section for per-user metrics
        st.divider()
        st.subheader("Per-User Analysis")
        
        plot_cols_per_user = st.columns(2)
        
        with plot_cols_per_user[0]:
            st.markdown("**💬 Mean Comments per User with 95% CI**")
            # Add hover data with p-values
            hover_data = []
            for g, t, m, s in zip(plot_data_per_user['comments']['group'], 
                                plot_data_per_user['comments']['topic'],
                                plot_data_per_user['comments']['mean'],
                                plot_data_per_user['comments']['stars']):
                if g == 'control':
                    hover_data.append('Control group')
                else:
                    hover_data.append(f"Mean: {m:.2f}<br>{'Significant vs control: ' + s if s else 'No significant difference vs control'}")

            fig_comments_per_user = px.scatter(
                plot_data_per_user['comments'],
                x='group',
                y='mean',
                color='topic',
                error_y=[ci_u - m for ci_u, m in zip(plot_data_per_user['comments']['ci_upper'], plot_data_per_user['comments']['mean'])],
                error_y_minus=[m - ci_l for m, ci_l in zip(plot_data_per_user['comments']['mean'], plot_data_per_user['comments']['ci_lower'])],
                labels={'mean': 'Mean Comments per User', 'group': 'Group', 'topic': 'Topic'},
                custom_data=[hover_data]
            )
            
            # Update hover template
            fig_comments_per_user.update_traces(
                hovertemplate="<b>%{x}</b><br>" +
                             "<b>Topic:</b> %{fullData.name}<br>" +
                             "<b>%{customdata[0]}</b><br>" +
                             "<extra></extra>"
            )
            
            # Add offset and annotations
            for i, trace in enumerate(fig_comments_per_user.data):
                # Create mapping of group names to numeric positions
                unique_groups = sorted(set(trace.x))
                x_positions = {group: idx for idx, group in enumerate(unique_groups)}
                
                # Get stars for this topic
                topic_name = trace.name
                topic_stars = [s for g, t, s in zip(plot_data_per_user['comments']['group'], 
                                                  plot_data_per_user['comments']['topic'], 
                                                  plot_data_per_user['comments']['stars']) 
                             if t == topic_name]
                
                # Apply offset and add annotations
                new_x = []
                for j, (x, y) in enumerate(zip(trace.x, trace.y)):
                    new_x_pos = x_positions[x] + (i-1)*0.2
                    new_x.append(new_x_pos)
                    if topic_stars[j]:  # Add annotation if there are stars
                        fig_comments_per_user.add_annotation(
                            x=new_x_pos,
                            y=y,
                            text=topic_stars[j],
                            showarrow=False,
                            yshift=30,
                            font=dict(size=25)
                        )
                trace.x = new_x
                
                # Update axis to show original group names
                fig_comments_per_user.update_layout(
                    xaxis=dict(
                        ticktext=unique_groups,
                        tickvals=[x_positions[x] for x in unique_groups],
                    )
                )
            
            fig_comments_per_user.update_traces(marker=dict(size=10))
            fig_comments_per_user.update_layout(height=400)
            st.plotly_chart(fig_comments_per_user, use_container_width=True)

        with plot_cols_per_user[1]:
            st.markdown("**👍 Mean Actions per User with 95% CI**")
            # Add hover data with p-values
            hover_data = []
            for g, t, m, s in zip(plot_data_per_user['actions']['group'], 
                                plot_data_per_user['actions']['topic'],
                                plot_data_per_user['actions']['mean'],
                                plot_data_per_user['actions']['stars']):
                if g == 'control':
                    hover_data.append('Control group')
                else:
                    hover_data.append(f"Mean: {m:.2f}<br>{'Significant vs control: ' + s if s else 'No significant difference vs control'}")

            fig_actions_per_user = px.scatter(
                plot_data_per_user['actions'],
                x='group',
                y='mean',
                color='topic',
                error_y=[ci_u - m for ci_u, m in zip(plot_data_per_user['actions']['ci_upper'], plot_data_per_user['actions']['mean'])],
                error_y_minus=[m - ci_l for m, ci_l in zip(plot_data_per_user['actions']['mean'], plot_data_per_user['actions']['ci_lower'])],
                labels={'mean': 'Mean Actions per User', 'group': 'Group', 'topic': 'Topic'},
                custom_data=[hover_data]
            )
            
            # Update hover template
            fig_actions_per_user.update_traces(
                hovertemplate="<b>%{x}</b><br>" +
                             "<b>Topic:</b> %{fullData.name}<br>" +
                             "<b>%{customdata[0]}</b><br>" +
                             "<extra></extra>"
            )
            
            # Add offset and annotations
            for i, trace in enumerate(fig_actions_per_user.data):
                unique_groups = sorted(set(trace.x))
                x_positions = {group: idx for idx, group in enumerate(unique_groups)}
                
                topic_name = trace.name
                topic_stars = [s for g, t, s in zip(plot_data_per_user['actions']['group'], 
                                                  plot_data_per_user['actions']['topic'], 
                                                  plot_data_per_user['actions']['stars']) 
                             if t == topic_name]
                
                new_x = []
                for j, (x, y) in enumerate(zip(trace.x, trace.y)):
                    new_x_pos = x_positions[x] + (i-1)*0.2
                    new_x.append(new_x_pos)
                    if topic_stars[j]:
                        fig_actions_per_user.add_annotation(
                            x=new_x_pos,
                            y=y,
                            text=topic_stars[j],
                            showarrow=False,
                            yshift=15,
                            font=dict(size=12)
                        )
                trace.x = new_x
                
                fig_actions_per_user.update_layout(
                    xaxis=dict(
                        ticktext=unique_groups,
                        tickvals=[x_positions[x] for x in unique_groups],
                    )
                )
            
            fig_actions_per_user.update_traces(marker=dict(size=10))
            fig_actions_per_user.update_layout(height=400)
            st.plotly_chart(fig_actions_per_user, use_container_width=True)

    return topics_content, total_games


def display_suggestions_metrics_by_topic(data, display: bool = True):
    # Extract treatment data
    treatment_data = data.get('suggestions_5', {})
    
    # Initialize per-topic metrics
    topics_metrics = {
        'Cats': {'total_actions': 0, 'total_comments': 0, 'total_generations': 0, 
                'total_selections': 0, 'unique_users': set(), 'users_with_suggestions': set(),
                'selection_types': {'positive': 0, 'neutral': 0, 'negative': 0}},
        'Oats': {'total_actions': 0, 'total_comments': 0, 'total_generations': 0, 
                'total_selections': 0, 'unique_users': set(), 'users_with_suggestions': set(),
                'selection_types': {'positive': 0, 'neutral': 0, 'negative': 0}},
        'Politics': {'total_actions': 0, 'total_comments': 0, 'total_generations': 0, 
                'total_selections': 0, 'unique_users': set(), 'users_with_suggestions': set(),
                'selection_types': {'positive': 0, 'neutral': 0, 'negative': 0}}
    }
    
    # Process data by topic
    for game in treatment_data.values():
        topic_order = game.get('topic_order', {})
        
        # Handle topic_order whether it's a list or dict
        if isinstance(topic_order, dict):
            topic_to_idx = {topic.capitalize(): idx for topic, idx in topic_order.items()}
        else:
            # If it's a list, create a mapping using enumerate
            topic_to_idx = {topic.capitalize(): idx for idx, topic in enumerate(topic_order)}
        
        # Process actions and comments by topic
        for topic, idx in topic_to_idx.items():
            if topic not in topics_metrics:
                continue
                
            # Count actions
            for action in game['actions'][idx]:
                if isinstance(action, dict) and 'user_id' in action:
                    topics_metrics[topic]['unique_users'].add(action['user_id'])
            topics_metrics[topic]['total_actions'] += len(game['actions'][idx])
            
            # Count comments
            for comment in game['comments'][idx]:
                if isinstance(comment, dict) and 'user_id' in comment:
                    topics_metrics[topic]['unique_users'].add(comment['user_id'])
            topics_metrics[topic]['total_comments'] += len(game['comments'][idx])
        
        # Process suggestions by topic
        for round_num, round_data in game.get('suggestions', {}).items():
            round_idx = int(round_num)
            current_topic = next((topic for topic, idx in topic_to_idx.items() if idx == round_idx), None)
            
            if current_topic:
                for user_id, user_data in round_data.items():
                    topics_metrics[current_topic]['unique_users'].add(user_id)
                    
                    if 'suggestions' in user_data:
                        topics_metrics[current_topic]['total_generations'] += len(user_data['suggestions'])
                    
                    if 'selected' in user_data:
                        selections = user_data['selected']
                        topics_metrics[current_topic]['total_selections'] += len(selections)
                        topics_metrics[current_topic]['users_with_suggestions'].add(user_id)
                        
                        # Match selections with suggestions
                        for selection in selections:
                            selected_text = selection[0]
                            
                            for suggestion_set in user_data.get('suggestions', []):
                                if isinstance(suggestion_set[1], list):
                                    for idx, reply in enumerate(suggestion_set[1]):
                                        if reply.get('reply') == selected_text:
                                            if idx == 0:
                                                topics_metrics[current_topic]['selection_types']['positive'] += 1
                                            elif idx == 1:
                                                topics_metrics[current_topic]['selection_types']['neutral'] += 1
                                            elif idx == 2:
                                                topics_metrics[current_topic]['selection_types']['negative'] += 1

    if display:
        
        st.divider()
        st.subheader("Suggestions Metrics by Topic")
        st.markdown("In the metrics below, we look at how the AI suggestions are used in the different topics of the game. "
                   "We look at the total number of __generations__ (when a user clicks on the button to have suggestions generated by the AI), the total number of __selections__ (when a user selects a suggestion), the engagement rate, "
                   "the number of unique users, the number of users who have used suggestions, and the user engagement rate. "
                   "We also show how much the different types are selected.")
        # Display metrics for each topic
        for topic in topics_metrics:
            st.subheader(f"{topic}")
            metrics = topics_metrics[topic]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Generations", metrics['total_generations'])
                st.metric("Total Selections", metrics['total_selections'])
                engagement_rate = metrics['total_selections'] / metrics['total_generations'] if metrics['total_generations'] > 0 else 0
                st.metric("Selection Rate", f"{engagement_rate:.2%}")
            with col2:
                st.metric("Total Users", len(metrics['unique_users']))
                st.metric("Users Using Suggestions", len(metrics['users_with_suggestions']))
                user_rate = len(metrics['users_with_suggestions']) / len(metrics['unique_users']) if metrics['unique_users'] else 0
                st.metric("User Engagement Rate", f"{user_rate:.2%}")
            with col3:

                st.metric("🟢 Positive Selections", 
                          metrics['selection_types']['positive'],
                          delta=f"{metrics['selection_types']['positive']/metrics['total_selections']:.1%} of total selections",
                          delta_color="off")
                st.metric("🟡 Neutral Selections", 
                          metrics['selection_types']['neutral'],
                          delta=f"{metrics['selection_types']['neutral']/metrics['total_selections']:.1%} of total selections",
                          delta_color="off")
                st.metric("🔴 Negative Selections", 
                          metrics['selection_types']['negative'],
                          delta=f"{metrics['selection_types']['negative']/metrics['total_selections']:.1%} of total selections",
                          delta_color="off")
            
            st.markdown("---")  # Add separator between topics
    
        # Add statistical analysis section
        st.divider()
        st.subheader("Statistical Analysis of Selection Types")
        
        # 1. Chi-square test for overall distribution comparison
        st.markdown("#### Distribution Comparison (Chi-square)")
        
        # Create contingency tables for pairwise comparisons
        def create_contingency_table(topic1, topic2):
            """Create 2x3 contingency table for two topics"""
            return np.array([
                [topics_metrics[topic1]['selection_types']['positive'],
                 topics_metrics[topic1]['selection_types']['neutral'],
                 topics_metrics[topic1]['selection_types']['negative']],
                [topics_metrics[topic2]['selection_types']['positive'],
                 topics_metrics[topic2]['selection_types']['neutral'],
                 topics_metrics[topic2]['selection_types']['negative']]
            ])
        
        # Perform pairwise chi-square tests
        comparisons = {
            'Cats vs Oats': ('Cats', 'Oats'),
            'Cats vs Politics': ('Cats', 'Politics')
        }
        
        for label, (topic1, topic2) in comparisons.items():
            contingency = create_contingency_table(topic1, topic2)
            chi2, p_val, dof, expected = chi2_contingency(contingency)
            
            st.markdown(f"**{label}**")
            
            # Create a DataFrame to show the distributions
            dist_data = pd.DataFrame({
                'Selection Type': ['Positive', 'Neutral', 'Negative'],
                topic1: contingency[0],
                f'{topic1} %': [f"{x/sum(contingency[0])*100:.1f}%" for x in contingency[0]],
                topic2: contingency[1],
                f'{topic2} %': [f"{x/sum(contingency[1])*100:.1f}%" for x in contingency[1]]
            })
            st.dataframe(dist_data)
            
            st.markdown(f"χ² = {chi2:.2f}, df = {dof}, p = {p_val:.4f}")
            if p_val < 0.05:
                # Analyze which categories contribute most to the chi-square
                contributions = (contingency - expected)**2 / expected
                largest_contrib = np.unravel_index(contributions.argmax(), contributions.shape)
                topics = [topic1, topic2]
                types = ['positive', 'neutral', 'negative']
                st.markdown(f"Largest difference: {topics[largest_contrib[0]]} - {types[largest_contrib[1]]}")
            
            st.markdown("---")
        
        # 2. Z-test for individual proportions (existing code)
        st.markdown("#### Individual Proportion Tests (Z-test)")
        
        def run_proportion_test(count1, n1, count2, n2):
            """
            Run z-test for comparing two proportions
            
            Args:
                count1: number of successes in first sample
                n1: size of first sample
                count2: number of successes in second sample
                n2: size of second sample
            """
            if n1 == 0 or n2 == 0:
                return None, None, None
                
            p1 = count1 / n1
            p2 = count2 / n2
            
            # Pooled proportion
            p_pooled = (count1 + count2) / (n1 + n2)
            
            # Standard error
            se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
            
            # Z-statistic
            if se == 0:  # Avoid division by zero
                return None, None, None
                
            z_stat = (p2 - p1) / se
            
            # Two-tailed p-value
            p_val = 2 * (1 - norm.cdf(abs(z_stat)))
            
            # Difference in proportions
            diff = p2 - p1
            
            return z_stat, p_val, diff

        # Store all p-values for multiple testing correction
        all_tests = []
        test_labels = []
        
        # Run tests for each selection type
        comparisons = {
            'Cats vs Oats': ('Cats', 'Oats'),
            'Cats vs Politics': ('Cats', 'Politics')
        }
        
        for selection_type in ['positive', 'neutral', 'negative']:
            st.markdown(f"#### {selection_type.title()} Selections")
            
            # Store results for this selection type
            results = []
            
            for label, (baseline, comparison) in comparisons.items():
                baseline_count = topics_metrics[baseline]['selection_types'][selection_type]
                baseline_total = topics_metrics[baseline]['total_selections']
                comparison_count = topics_metrics[comparison]['selection_types'][selection_type]
                comparison_total = topics_metrics[comparison]['total_selections']
                
                test_result = run_proportion_test(
                    baseline_count, baseline_total,
                    comparison_count, comparison_total
                )
                
                if test_result[0] is not None:
                    stat, p_val, diff = test_result
                    all_tests.append(p_val)
                    test_labels.append(f"{selection_type} - {label}")
                    results.append((label, stat, p_val, diff))
            
            # Display results table for this selection type
            if results:
                data = {
                    'Comparison': [r[0] for r in results],
                    'Z-statistic': [f"{r[1]:.2f}" for r in results],
                    'p-value': [f"{r[2]:.4f}" for r in results],
                    'Difference': [f"{r[3]:+.1%}" for r in results]
                }
                st.dataframe(pd.DataFrame(data))

        # Apply multiple testing correction
        if all_tests:
            st.markdown("#### Multiple Testing Correction")
            st.markdown("Benjamini-Hochberg correction for multiple comparisons:")
            
            _, corrected_p, _, _ = multipletests(all_tests, method='bonferroni')
            
            correction_data = {
                'Test': test_labels,
                'Original p-value': [f"{p:.4f}" for p in all_tests],
                'Corrected p-value': [f"{p:.4f}" for p in corrected_p],
                'Significant': ['*' if p < 0.05 else '' for p in corrected_p]
            }
            st.dataframe(pd.DataFrame(correction_data))

    return topics_metrics

def display_chat_metrics_by_topic(data, display: bool = True):
    # Extract treatment data
    treatment_data = data.get('chat_5', {})
    st.write(f"Number of games in treatment: {len(treatment_data)}")  # Debug
    
    # Initialize per-topic metrics
    topics_metrics = {
        'Cats': {'total_actions': 0, 'total_comments': 0, 'total_prompts': 0,
                'unique_users': set(), 'users_with_chat': set(),
                'prompts_per_user': [], 'chat_inspired_comments': [],
                'comments_data': []},
        'Oats': {'total_actions': 0, 'total_comments': 0, 'total_prompts': 0,
                'unique_users': set(), 'users_with_chat': set(),
                'prompts_per_user': [], 'chat_inspired_comments': [],
                'comments_data': []},
        'Politics': {'total_actions': 0, 'total_comments': 0, 'total_prompts': 0,
                'unique_users': set(), 'users_with_chat': set(),
                'prompts_per_user': [], 'chat_inspired_comments': [],
                'comments_data': []}
    }
    
    # Process each game
    for game_id, game in treatment_data.items():
        topic_order = game.get('topic_order', [])
        
        # Handle topic_order whether it's a list or dict
        if isinstance(topic_order, dict):
            topic_to_idx = {topic.capitalize(): idx for topic, idx in topic_order.items()}
        else:
            topic_to_idx = {topic.capitalize(): idx for idx, topic in enumerate(topic_order)}
        
        # Process actions and comments by topic
        for topic, idx in topic_to_idx.items():
            if topic not in topics_metrics:
                continue
                
            # Count actions
            for action in game['actions'][idx]:
                if isinstance(action, dict) and 'user_id' in action:
                    topics_metrics[topic]['unique_users'].add(action['user_id'])
            topics_metrics[topic]['total_actions'] += len(game['actions'][idx])
            
            # Count comments and check for chat inspiration
            comments = game['comments'][idx]
            round_chat = game.get('chat', {}).get(idx, {})
            
            # Track which comments we've already matched to avoid duplicates
            matched_comments = set()
            
            for comment in comments:
                # Skip if we've already matched this comment
                comment_id = f"{comment['user_id']}_{comment['content']}"
                if comment_id in matched_comments:
                    continue
                    
                comment_time = datetime.fromisoformat(str(comment['timestamp']))
                user_id = comment['user_id']
                
                # Get this user's chat messages
                user_chat = round_chat.get(user_id, [])
                assistant_messages = [
                    (msg[0].replace('assistant: ', ''), datetime.fromisoformat(msg[1].replace('Z', '+00:00')))
                    for msg in user_chat
                    if msg[0].startswith('assistant:') and 
                    datetime.fromisoformat(msg[1].replace('Z', '+00:00')) < comment_time
                ]
                
                # Find most similar chat message
                best_match = None
                best_similarity = 0
                best_delay = 0
                
                for chat_msg, chat_time in assistant_messages:
                    similarity = calculate_text_similarity(chat_msg, comment['content'])
                    if similarity > 0.5 and similarity > best_similarity:
                        best_similarity = similarity
                        best_match = chat_msg
                        best_delay = (comment_time - chat_time).total_seconds()
                
                # If we found a match, add it and mark comment as matched
                if best_match:
                    topics_metrics[topic]['chat_inspired_comments'].append({
                        'comment': comment['content'],
                        'chat_message': best_match,
                        'similarity': best_similarity,
                        'delay': best_delay
                    })
                    matched_comments.add(comment_id)
            
            for comment in comments:
                if isinstance(comment, dict) and 'user_id' in comment:
                    topics_metrics[topic]['unique_users'].add(comment['user_id'])
                    topics_metrics[topic]['comments_data'].append({
                        'user_id': comment['user_id'],
                        'content': comment['content'],
                        'timestamp': comment['timestamp']
                    })
            topics_metrics[topic]['total_comments'] += len(comments)
        
        # Process chat by topic
        for round_num, round_data in game.get('chat', {}).items():
            round_idx = int(round_num)
            current_topic = next((topic for topic, idx in topic_to_idx.items() if idx == round_idx), None)
            
            if current_topic:
                # Track prompts per user for this round
                user_prompts = {}
                
                for user_id, messages in round_data.items():
                    topics_metrics[current_topic]['unique_users'].add(user_id)
                    user_prompts[user_id] = 0
                    
                    # Count user messages
                    for message in messages:
                        if message[0].startswith('user:'):
                            user_prompts[user_id] += 1
                            topics_metrics[current_topic]['total_prompts'] += 1
                    
                    if user_prompts[user_id] > 0:
                        topics_metrics[current_topic]['users_with_chat'].add(user_id)
                        topics_metrics[current_topic]['prompts_per_user'].append(user_prompts[user_id])

    if display:
        for topic in topics_metrics:
            st.subheader(f"{topic}")
            metrics = topics_metrics[topic]
            
            # Calculate averages and rates
            avg_prompts = np.mean(metrics['prompts_per_user']) if metrics['prompts_per_user'] else 0
            std_prompts = np.std(metrics['prompts_per_user']) if metrics['prompts_per_user'] else 0
            user_rate = len(metrics['users_with_chat']) / len(metrics['unique_users']) if metrics['unique_users'] else 0
            chat_inspired_count = len(metrics['chat_inspired_comments'])
            chat_inspired_rate = chat_inspired_count / metrics['total_comments'] if metrics['total_comments'] > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Actions", metrics['total_actions'])
                st.metric("Total Comments", metrics['total_comments'])
                st.metric("Chat-Inspired Comments", chat_inspired_count)
                st.metric("Chat Inspiration Rate", f"{chat_inspired_rate:.2%}")
            with col2:
                st.metric("Total Users", len(metrics['unique_users']))
                st.metric("Users Using Chat", len(metrics['users_with_chat']))
                st.metric("Chat Usage Rate", f"{user_rate:.2%}")
            with col3:
                st.metric("Total Prompts", metrics['total_prompts'])
                st.metric("Avg Prompts/User", f"{avg_prompts:.2f}")
                st.metric("Std Prompts/User", f"{std_prompts:.2f}")
                
            
            st.markdown("---")
    
    return topics_metrics


def calculate_text_similarity(text1, text2):
    """Calculate best similarity ratio between texts, handling different lengths."""
    # Convert to lowercase
    text1 = text1.lower()
    text2 = text2.lower()
    
    # If one text is significantly longer, check for substring matches
    if len(text1.split()) > 2 * len(text2.split()) or len(text2.split()) > 2 * len(text1.split()):
        # Make sure text1 is the shorter one
        if len(text1) > len(text2):
            text1, text2 = text2, text1
            
        # Split into sentences or chunks
        chunks = text2.split('.')
        best_ratio = 0
        for chunk in chunks:
            if chunk.strip():  # Skip empty chunks
                ratio = SequenceMatcher(None, text1, chunk.strip()).ratio()
                best_ratio = max(best_ratio, ratio)
        return best_ratio
    else:
        # For similar length texts, use regular comparison
        return SequenceMatcher(None, text1, text2).ratio()


def analyze_commenters_by_topic_distribution(data, chat_impact=None):
    """Analyze commenting patterns within each topic, grouping users by their relative activity levels."""
    # Initialize structure for results
    topic_analysis = {
        'Cats': {'user_stats': {}, 'distribution': {}},
        'Oats': {'user_stats': {}, 'distribution': {}},
        'Politics': {'user_stats': {}, 'distribution': {}}
    }
    
    # Get chat_5 treatment data
    treatment_data = data.get('chat_5', {})
    
    # First pass: Collect raw data per user per topic
    for game_id, game_data in treatment_data.items():
        topic_order = game_data.get('topic_order', [])
        
        # Process each round/topic
        for round_idx, topic in enumerate(topic_order):
            topic = topic.capitalize()
            if topic not in topic_analysis:
                continue
            
            # Process comments for this round
            round_comments = game_data['comments'][round_idx]
            round_actions = game_data['actions'][round_idx]
            
            # Process comments
            for comment in round_comments:
                user_id = comment.get('user_id')
                content_id = comment.get('content_id')
                
                if user_id:
                    # Initialize user stats if needed
                    if user_id not in topic_analysis[topic]['user_stats']:
                        topic_analysis[topic]['user_stats'][user_id] = {
                            'n_comments': 0,
                            'likes_received': 0,
                            'likes_per_comment': 0,
                            'chat_usage': 0,
                            'comments': []
                        }
                    topic_analysis[topic]['user_stats'][user_id]['n_comments'] += 1
                    topic_analysis[topic]['user_stats'][user_id]['comments'].append(comment['content'])
            
            # Process actions (likes)
            for action in round_actions:
                if isinstance(action, dict):
                    target_content_id = action.get('content_id')
                    # Find the comment with this content_id
                    for comment in round_comments:
                        if comment.get('content_id') == target_content_id:
                            user_id = comment.get('user_id')
                            if user_id in topic_analysis[topic]['user_stats']:
                                topic_analysis[topic]['user_stats'][user_id]['likes_received'] += 1
                            break
            
            # Process chat usage - Fixed structure based on treatment_info.py
            if 'chat' in game_data:
                round_data = game_data['chat'].get(round_idx, {})                
                for user_id, messages in round_data.items():
                    if user_id not in topic_analysis[topic]['user_stats']:
                        topic_analysis[topic]['user_stats'][user_id] = {
                            'n_comments': 0,
                            'likes_received': 0,
                            'likes_per_comment': 0,
                            'chat_usage': 0,
                            'comments': []
                        }
                    
                    # Count user messages
                    user_prompt_count = sum(1 for message in messages if message[0].startswith('user:'))
                    topic_analysis[topic]['user_stats'][user_id]['chat_usage'] += user_prompt_count
    
    # Calculate likes per comment for each user
    for topic in topic_analysis:
        for user_id, stats in topic_analysis[topic]['user_stats'].items():
            if stats['n_comments'] > 0:
                stats['likes_per_comment'] = stats['likes_received'] / stats['n_comments']
    
    # Second pass: Calculate percentiles and group users
    for topic in topic_analysis:
        users = topic_analysis[topic]['user_stats']
        if not users:
            continue
            
        # Get comment counts for all users in this topic
        comment_counts = [stats['n_comments'] for stats in users.values()]
        
        if not comment_counts:  # Skip if no comments for this topic
            continue
            
        # Calculate percentiles
        percentile_25 = np.percentile(comment_counts, 25)
        percentile_75 = np.percentile(comment_counts, 75)
        
        # Initialize distribution groups
        topic_analysis[topic]['distribution'] = {
            'top_commenters': [],
            'average_commenters': [],
            'light_commenters': []
        }
        
        # Categorize users and calculate metrics
        for user_id, stats in users.items():
            if stats['n_comments'] >= percentile_75:
                group = 'top_commenters'
            elif stats['n_comments'] <= percentile_25:
                group = 'light_commenters'
            else:
                group = 'average_commenters'
            
            topic_analysis[topic]['distribution'][group].append({
                'user_id': user_id,
                'stats': stats
            })
    
    # Display results
    st.header("Commenter Analysis by Topic")
    
    for topic in topic_analysis:
        st.subheader(topic)
        
        distribution = topic_analysis[topic]['distribution']
        if not distribution:  # Skip if no distribution data
            continue
            
        for group in ['top_commenters', 'average_commenters', 'light_commenters']:
            users = distribution.get(group, [])  # Use .get() to avoid KeyError
            if not users:
                continue
            
            avg_comments = np.mean([u['stats']['n_comments'] for u in users])
            avg_likes = np.mean([u['stats'].get('likes_received', 0) / u['stats']['n_comments'] if u['stats']['n_comments'] > 0 else 0 for u in users])
            
            # Make sure we show which group we're looking at
            st.markdown(f"**{group.replace('_', ' ').title()}** (n={len(users)})")
            
            # Your existing metrics with normalization
            avg_comments = np.mean([u['stats']['n_comments'] for u in users])
            avg_likes = np.mean([u['stats'].get('likes_received', 0) / u['stats']['n_comments'] if u['stats']['n_comments'] > 0 else 0 for u in users])
            
            # Try to calculate inspiration rate with debugging
            inspiration_rate = 0
            if chat_impact and topic in chat_impact:
                group_inspired_comments = 0
                group_total_comments = 0
                
                for user in users:
                    user_id = user['user_id']
                    user_comments = user['stats']['comments']  # Get all comments for this user
                    total_user_comments = len(user_comments)
                    
                    # Count how many of user's comments were chat-inspired
                    user_inspired_comments = sum(
                        1 for inspired in chat_impact[topic]['chat_inspired_comments']
                        if inspired['comment'] in user_comments
                    )
                    
                    group_inspired_comments += user_inspired_comments
                    group_total_comments += total_user_comments
                
                inspiration_rate = group_inspired_comments / group_total_comments if group_total_comments > 0 else 0
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg Comments", f"{avg_comments:.1f}")
            with col2:
                st.metric("Avg Likes per Comment", f"{avg_likes:.1f}")
            with col3:
                if chat_impact and topic in chat_impact:
                    st.metric(
                        "Chat Inspiration Rate", 
                        f"{inspiration_rate:.1%}",
                        delta=f"{group_inspired_comments} of {group_total_comments} comments",
                        delta_color="off"
                    )
                else:
                    st.metric("Chat Inspiration Rate", "N/A")
            with col4:
                avg_chat_usage = np.mean([u['stats']['chat_usage'] for u in users])
                st.metric("Avg Chat Usage", f"{avg_chat_usage:.1f}")
        
        st.divider()
    
    return topic_analysis

def display_conversation_metrics_by_topic(data, display: bool = True):
    # Extract treatment data
    treatment_data = data.get('conversation_5', {})
    st.write(f"Number of games in treatment: {len(treatment_data)}")
    
    # Initialize per-topic metrics
    topics_metrics = {
        'Cats': {'total_actions': 0, 'total_comments': 0, 'total_starters': 0,
                'unique_users': set(), 'users_with_starters': set(),
                'starters_per_user': [], 'inspired_comments': [],
                'comments_data': []},
        'Oats': {'total_actions': 0, 'total_comments': 0, 'total_starters': 0,
                'unique_users': set(), 'users_with_starters': set(),
                'starters_per_user': [], 'inspired_comments': [],
                'comments_data': []},
        'Politics': {'total_actions': 0, 'total_comments': 0, 'total_starters': 0,
                'unique_users': set(), 'users_with_starters': set(),
                'starters_per_user': [], 'inspired_comments': [],
                'comments_data': []}
    }
    
    # Process each game
    for game_id, game in treatment_data.items():
        topic_order = game.get('topic_order', [])
        
        # Handle topic_order whether it's a list or dict
        if isinstance(topic_order, dict):
            topic_to_idx = {topic.capitalize(): idx for topic, idx in topic_order.items()}
        else:
            topic_to_idx = {topic.capitalize(): idx for idx, topic in enumerate(topic_order)}
        
        # Process actions and comments by topic
        for topic, idx in topic_to_idx.items():
            if topic not in topics_metrics:
                continue
                
            # Count actions
            for action in game['actions'][idx]:
                if isinstance(action, dict) and 'user_id' in action:
                    topics_metrics[topic]['unique_users'].add(action['user_id'])
            topics_metrics[topic]['total_actions'] += len(game['actions'][idx])
            
            # Count comments and check for conversation starter inspiration
            comments = game['comments'][idx]
            round_starters = game.get('conversation_starter', {}).get(idx, {})
            
            # Track which comments we've already matched to avoid duplicates
            matched_comments = set()
            
            for comment in comments:
                # Skip if we've already matched this comment
                comment_id = f"{comment['user_id']}_{comment['content']}"
                if comment_id in matched_comments:
                    continue
                    
                comment_time = datetime.fromisoformat(str(comment['timestamp']))
                user_id = comment['user_id']
                
                # Get this user's conversation starters
                user_starters = round_starters.get(user_id, [])
                
                # Check each starter and its suggestions
                for starter in user_starters:
                    starter_time = datetime.fromisoformat(starter[2].replace('Z', '+00:00'))
                    if starter_time >= comment_time:
                        continue
                        
                    suggestions = starter[1]
                    # Check each suggestion type and its text
                    for suggestion_type, suggestion_text in suggestions.items():
                        similarity = calculate_text_similarity(suggestion_text, comment['content'])
                        if similarity > 0.5:  # You might want to adjust this threshold
                            topics_metrics[topic]['inspired_comments'].append({
                                'comment': comment['content'],
                                'starter_suggestion': suggestion_text,
                                'suggestion_type': suggestion_type,
                                'similarity': similarity,
                                'delay': (comment_time - starter_time).total_seconds()
                            })
                            matched_comments.add(comment_id)
                            break
            
            # Store comment data
            for comment in comments:
                if isinstance(comment, dict) and 'user_id' in comment:
                    topics_metrics[topic]['unique_users'].add(comment['user_id'])
                    topics_metrics[topic]['comments_data'].append({
                        'user_id': comment['user_id'],
                        'content': comment['content'],
                        'timestamp': comment['timestamp']
                    })
            topics_metrics[topic]['total_comments'] += len(comments)
        
        # Process conversation starters by topic
        for round_num, round_data in game.get('conversation_starter', {}).items():
            round_idx = int(round_num)
            current_topic = next((topic for topic, idx in topic_to_idx.items() if idx == round_idx), None)
            
            if current_topic:
                # Track starters per user for this round
                user_starters = {}
                
                for user_id, starters in round_data.items():
                    topics_metrics[current_topic]['unique_users'].add(user_id)
                    user_starters[user_id] = len(starters)
                    topics_metrics[current_topic]['total_starters'] += len(starters)
                    
                    if user_starters[user_id] > 0:
                        topics_metrics[current_topic]['users_with_starters'].add(user_id)
                        topics_metrics[current_topic]['starters_per_user'].append(user_starters[user_id])

    if display:
        for topic in topics_metrics:
            st.subheader(f"{topic}")
            metrics = topics_metrics[topic]
            
            # Calculate averages and rates
            avg_starters = np.mean(metrics['starters_per_user']) if metrics['starters_per_user'] else 0
            std_starters = np.std(metrics['starters_per_user']) if metrics['starters_per_user'] else 0
            user_rate = len(metrics['users_with_starters']) / len(metrics['unique_users']) if metrics['unique_users'] else 0
            inspired_count = len(metrics['inspired_comments'])
            inspired_rate = inspired_count / metrics['total_comments'] if metrics['total_comments'] > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Actions", metrics['total_actions'])
                st.metric("Total Comments", metrics['total_comments'])
                st.metric("Starter-Inspired Comments", inspired_count)
                st.metric("Starter Inspiration Rate", f"{inspired_rate:.2%}")
            with col2:
                st.metric("Total Users", len(metrics['unique_users']))
                st.metric("Users Using Starters", len(metrics['users_with_starters']))
                st.metric("Starter Usage Rate", f"{user_rate:.2%}")
            with col3:
                st.metric("Total Starters Used", metrics['total_starters'])
                st.metric("Avg Starters/User", f"{avg_starters:.2f}")
                st.metric("Std Starters/User", f"{std_starters:.2f}")
            
            st.markdown("---")
    
    return topics_metrics