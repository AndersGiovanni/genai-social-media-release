from typing import Dict, Any
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy import stats
import json
import os
import plotly.express as px
from scipy.stats import t

# Constants for our analysis
LIKERT_MAPPING = {
    'Strongly disagree': 1,
    'Disagree': 2,
    'Neutral': 3,
    'Agree': 4,
    'Strongly agree': 5,
    'Prefer not to answer': None,
    'EMPTY': None
}

# Mapping between pre and post survey questions
QUESTION_PAIRS = {
    'comfortableWithAI': 'aiComfortability',
    'aiSuggestions': 'aiParticipation',
    'lessToxic': 'aiLessToxic',
    'lessPolarizing': 'aiLessPolarizing',
    'reduceMisinformation': 'aiReducesMisinformation',
    'aiContentAccurate': 'aiAccuracy',
    'aiRegulation': 'aiRegulation'
}

# Question display names for better plot labels
QUESTION_LABELS = {
    'comfortableWithAI': 'I feel comfortable with AI being used on social media platforms.',
    'aiSuggestions': 'AI suggestions can make it more likely\nfor me to participate in online discussions.',
    'lessToxic': 'AI can make online discussions more positive and less toxic.',
    'lessPolarizing': 'AI can make discussions less polarizing.',
    'reduceMisinformation': 'AI can help reduce misinformation on social media.',
    'aiContentAccurate': 'AI-generated content is accurate and reliable.',
    'aiRegulation': 'AI should be regulated to prevent\nmisuse and ensure ethical use.'
}


def get_likert_value(response: str) -> int:
    """Convert Likert scale response to numeric value"""
    return LIKERT_MAPPING.get(response, None)


def process_surveys(df: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Process survey data and return nested dictionary structure with demographic information
    """
    result = {}
    
    # Group by treatment
    for treatment_name, treatment_group in df.groupby('treatmentName'):
        result[treatment_name] = {}
        
        # Process each player in the treatment
        for _, player in treatment_group.iterrows():
            # Extract demographic information
            demographic_info = {}
            if isinstance(player['initialSurvey'], dict) and 'demographicInfo' in player['initialSurvey']:
                for demo_key in ['age', 'gender', 'education', 'occupation', 'partyAffiliation']:
                    demo_value = player['initialSurvey']['demographicInfo'].get(demo_key, '')
                    
                    # # Special handling for party affiliation
                    if demo_key == 'partyAffiliation' and demo_value:
                        if 'Republican' in demo_value:
                            demo_value = demo_value
                        elif 'Democrat' in demo_value:
                            demo_value = demo_value
                    
                    # Only include non-empty values that aren't "Prefer not to answer"
                    if demo_value and demo_value != 'Prefer not to answer':
                        demographic_info[demo_key] = demo_value
            
            player_data = {
                'pre_survey': {},
                'post_survey': {},
                'differences': {},
                'demographics': demographic_info  # Add demographics to player data
            }
            
            # Get survey responses
            pre_survey = player['initialSurvey'].get('aiAssessmentInfo', {})
            post_survey = player['exitSurvey'].get('socialMediaAI', {}) if isinstance(player['exitSurvey'], dict) else {}
            
            # Process each question pair
            for pre_q, post_q in QUESTION_PAIRS.items():
                # Get pre-survey response
                pre_val = get_likert_value(pre_survey.get(pre_q, 'EMPTY'))
                player_data['pre_survey'][pre_q] = pre_val
                
                # Get post-survey response
                post_val = get_likert_value(post_survey.get(post_q, 'EMPTY'))
                player_data['post_survey'][post_q] = post_val
                
                # Calculate difference if both values exist
                if pre_val is not None and post_val is not None:
                    player_data['differences'][pre_q] = post_val - pre_val
                else:
                    player_data['differences'][pre_q] = None
            
            # Add player data to treatment dictionary
            result[treatment_name][player['id']] = player_data
    
    return result


def get_treatment_data(df: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Main function to process the dataframe and return the treatment data
    """
    # Ensure the necessary columns exist
    required_cols = ['id', 'treatmentName', 'initialSurvey', 'exitSurvey']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame missing required columns: {required_cols}")
    
    return process_surveys(df)

def treatment_dict_to_df(treatment_data: Dict[str, Dict[str, Dict[str, Any]]]) -> pd.DataFrame:
    """Convert the nested treatment dictionary to a flat DataFrame"""
    rows = []
    for treatment, players in treatment_data.items():
        for player_id, data in players.items():
            for question in QUESTION_PAIRS.keys():
                # Only include if we have both pre and post scores
                if data['differences'][question] is not None:
                    rows.append({
                        'treatment': treatment,
                        'player_id': player_id,
                        'question': question,
                        'pre_score': data['pre_survey'][question],
                        'post_score': data['post_survey'][QUESTION_PAIRS[question]],
                        'diff': data['differences'][question]
                    })
    return pd.DataFrame(rows)


def calculate_parametric_ci(data: pd.Series, confidence: float = 0.95) -> tuple:
    """Calculate mean and confidence interval using parametric method"""
    mean = data.mean()
    std_err = data.std() / np.sqrt(len(data))
    ci = std_err * 1.96  # 95% confidence interval
    return mean, ci


def calculate_bootstrap_ci(data: pd.Series, confidence: float = 0.95, n_bootstrap: int = 1000) -> tuple:
    """Calculate mean and confidence interval using bootstrapping"""
    mean = data.mean()
    
    # Bootstrap sampling
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = data.sample(n=len(data), replace=True)
        bootstrap_means.append(sample.mean())
    
    # Calculate confidence interval
    alpha = (1 - confidence) / 2
    ci = np.percentile(bootstrap_means, [alpha * 100, (1 - alpha) * 100])
    ci_width = (ci[1] - ci[0]) / 2
    
    return mean, ci_width


def calculate_ttest(treatment_data: pd.Series, control_data: pd.Series) -> tuple:
    """
    Perform two-sided t-test between treatment and control groups
    
    Parameters:
    -----------
    treatment_data : pd.Series
        Treatment group differences
    control_data : pd.Series
        Control group differences
        
    Returns:
    --------
    tuple : (t-statistic, p-value)
    """
    t_stat, p_val = stats.ttest_ind(treatment_data, control_data)
    return t_stat, p_val


def add_significance_markers(fig, x_pos: int, y_pos: float, p_value: float) -> None:
    """Add significance markers to the plot"""
    if p_value < 0.001:
        marker = "***"
    elif p_value < 0.01:
        marker = "**"
    elif p_value < 0.05:
        marker = "*"
    elif p_value < 0.1:
        marker = "†"
    else:
        marker = "ns"
        
    fig.add_annotation(
        x=x_pos,
        y=y_pos,
        text=marker,
        showarrow=False,
        yshift=10,
        font=dict(size=14)
    )


@st.cache_data
def calculate_treatment_stats(treatment_data, control_group, question, 
                             use_bootstrap=True, confidence=0.95, 
                             n_bootstrap=10000, demographic=None):
    """
    Calculate treatment statistics with optional demographic grouping
    """
    stats_list = []
    
    # If demographic is selected, we'll group by both treatment and demographic
    if demographic != 'None':
        # First, collect all data by treatment and demographic
        grouped_data = {}
        
        for treatment, players in treatment_data.items():
            for player_id, player_data in players.items():
                # Skip if no demographic data or the specific demographic is missing
                if 'demographics' not in player_data or demographic not in player_data['demographics']:
                    continue
                
                demo_value = player_data['demographics'][demographic]
                group_key = f"{treatment}_{demo_value}"
                
                if group_key not in grouped_data:
                    grouped_data[group_key] = {
                        'treatment': treatment,
                        'demographic': demo_value,
                        'differences': []
                    }
                
                # Add the difference value if it exists
                diff_value = player_data['differences'].get(question)
                if diff_value is not None:
                    grouped_data[group_key]['differences'].append(diff_value)
        
        # Calculate statistics for each group
        for group_key, group_data in grouped_data.items():
            differences = group_data['differences']
            
            if len(differences) > 0:
                mean_diff = np.mean(differences)
                n = len(differences)
                
                # Calculate confidence interval
                if use_bootstrap and n > 1:
                    # Bootstrap confidence interval
                    np.random.seed(42)  # For reproducibility
                    bootstrap_means = []
                    
                    for _ in range(n_bootstrap):
                        bootstrap_sample = np.random.choice(differences, size=n, replace=True)
                        bootstrap_means.append(np.mean(bootstrap_sample))
                    
                    ci_lower, ci_upper = np.percentile(bootstrap_means, 
                                                     [(1-confidence)*100/2, 100-(1-confidence)*100/2])
                    ci = max(mean_diff - ci_lower, ci_upper - mean_diff)
                else:
                    # Parametric confidence interval
                    if n > 1:
                        std_err = np.std(differences, ddof=1) / np.sqrt(n)
                        t_val = t.ppf((1 + confidence) / 2, n - 1)
                        ci = t_val * std_err
                    else:
                        ci = 0
                
                stats_list.append({
                    'treatment': group_data['treatment'],
                    'demographic': group_data['demographic'],
                    'mean': mean_diff,
                    'ci': ci,
                    'n': n,
                    'p_value': None,  # Will calculate later
                    'raw_differences': differences  # Store raw differences for p-value calculation
                })
        
        # Calculate p-values and effect sizes by comparing to control group within each demographic
        for demographic_value in set(item['demographic'] for item in stats_list):
            # Find control group for this demographic
            control_stats = next((s for s in stats_list
                                if s['demographic'] == demographic_value and s['treatment'] == control_group), None)

            if control_stats and 'raw_differences' in control_stats:
                control_differences = control_stats['raw_differences']

                # Compare each treatment to control within this demographic
                for i, stat in enumerate(stats_list):
                    if stat['demographic'] == demographic_value and stat['treatment'] != control_group:
                        treatment_differences = stat['raw_differences']

                        # Use the existing calculate_p_value function
                        p_value = calculate_p_value(treatment_differences, control_differences, n_bootstrap)
                        stats_list[i]['p_value'] = p_value

                        # Calculate Cohen's d effect size
                        effect_size = calculate_cohens_d(treatment_differences, control_differences, confidence)
                        stats_list[i]['cohens_d'] = effect_size['d']
                        stats_list[i]['d_ci_lower'] = effect_size['ci_lower']
                        stats_list[i]['d_ci_upper'] = effect_size['ci_upper']
                        stats_list[i]['effect_interpretation'] = effect_size['interpretation']
                    elif stat['demographic'] == demographic_value and stat['treatment'] == control_group:
                        # Control group compared to itself has d=0
                        stats_list[i]['cohens_d'] = 0.0
                        stats_list[i]['d_ci_lower'] = 0.0
                        stats_list[i]['d_ci_upper'] = 0.0
                        stats_list[i]['effect_interpretation'] = 'reference'
    
    else:
        # Original code for treatment-only statistics
        for treatment, players in treatment_data.items():
            differences = []
            
            for player_id, player_data in players.items():
                diff_value = player_data['differences'].get(question)
                if diff_value is not None:
                    differences.append(diff_value)
            
            if len(differences) > 0:
                mean_diff = np.mean(differences)
                n = len(differences)
                
                # Calculate confidence interval
                if use_bootstrap and n > 1:
                    # Bootstrap confidence interval
                    np.random.seed(42)  # For reproducibility
                    bootstrap_means = []
                    
                    for _ in range(n_bootstrap):
                        bootstrap_sample = np.random.choice(differences, size=n, replace=True)
                        bootstrap_means.append(np.mean(bootstrap_sample))
                    
                    ci_lower, ci_upper = np.percentile(bootstrap_means, 
                                                     [(1-confidence)*100/2, 100-(1-confidence)*100/2])
                    ci = max(mean_diff - ci_lower, ci_upper - mean_diff)
                else:
                    # Parametric confidence interval
                    if n > 1:
                        std_err = np.std(differences, ddof=1) / np.sqrt(n)
                        t_val = t.ppf((1 + confidence) / 2, n - 1)
                        ci = t_val * std_err
                    else:
                        ci = 0
                
                stats_list.append({
                    'treatment': treatment,
                    'mean': mean_diff,
                    'ci': ci,
                    'n': n,
                    'p_value': None,  # Will calculate later
                    'raw_differences': differences  # Store raw differences for p-value calculation
                })
        
        # Calculate p-values and effect sizes by comparing to control group
        control_stats = next((s for s in stats_list if s['treatment'] == control_group), None)
        if control_stats and 'raw_differences' in control_stats:
            control_differences = control_stats['raw_differences']

            for i, stat in enumerate(stats_list):
                if stat['treatment'] != control_group:
                    treatment_differences = stat['raw_differences']

                    # Use the existing calculate_p_value function
                    p_value = calculate_p_value(treatment_differences, control_differences, n_bootstrap)
                    stats_list[i]['p_value'] = p_value

                    # Calculate Cohen's d effect size
                    effect_size = calculate_cohens_d(treatment_differences, control_differences, confidence)
                    stats_list[i]['cohens_d'] = effect_size['d']
                    stats_list[i]['d_ci_lower'] = effect_size['ci_lower']
                    stats_list[i]['d_ci_upper'] = effect_size['ci_upper']
                    stats_list[i]['effect_interpretation'] = effect_size['interpretation']
                else:
                    # Control group compared to itself has d=0
                    stats_list[i]['cohens_d'] = 0.0
                    stats_list[i]['d_ci_lower'] = 0.0
                    stats_list[i]['d_ci_upper'] = 0.0
                    stats_list[i]['effect_interpretation'] = 'reference'
    
    # Convert to DataFrame and remove raw_differences column before returning
    stats_df = pd.DataFrame(stats_list)
    if 'raw_differences' in stats_df.columns:
        stats_df = stats_df.drop(columns=['raw_differences'])
    
    return stats_df

def plot_treatment_effects(treatment_data: Dict[str, Dict[str, Dict[str, Any]]], 
                          control_group: str = 'baseline_5',
                          use_bootstrap: bool = True,
                          confidence: float = 0.95,
                          n_bootstrap: int = 10000) -> None:
    """
    Create interactive dot plots with error bars for each question using Plotly
    """
    # Create a dictionary to store all plot data
    all_plot_data = {}
    
    # Display confidence interval settings at the top of the page
    st.write("### Confidence Interval Settings")
    use_bootstrap = st.checkbox("Use Bootstrap CI (otherwise parametric)", value=use_bootstrap)
    confidence = st.slider("Confidence Level", 0.8, 0.99, 0.95, 0.01)
    
    if use_bootstrap:
        n_bootstrap = st.number_input("Number of Bootstrap Samples", 
                                    min_value=100, 
                                    max_value=10000, 
                                    value=10000, 
                                    step=100)
    
    # Add demographic grouping option
    demographic_options = ['None', 'age', 'gender', 'education', 'occupation', 'partyAffiliation']
    selected_demographic = st.selectbox(
        "Group by demographic (optional):", 
        demographic_options,
        key="demographic_select_ate"
    )
    
    st.divider()

    # Process each question
    for question in QUESTION_PAIRS.keys():
        # Get cached statistics with demographic information if selected
        stats_df = calculate_treatment_stats(
            treatment_data, 
            control_group, 
            question,
            use_bootstrap,
            confidence,
            n_bootstrap,
            selected_demographic  # Pass the selected demographic to the stats function
        )
        
        # Create plot data from stats_df
        plot_data = {
            'x_vals': list(range(1, len(stats_df) + 1)),
            'y_vals': stats_df['mean'].tolist(),
            'err': stats_df['ci'].tolist(),
            'significance': [
                "***" if p < 0.001 else
                "**" if p < 0.01 else
                "*" if p < 0.05 else
                "†" if p < 0.1 else
                "ns" if p is not None else ""
                for p in stats_df['p_value']
            ],
            'n_values': stats_df['n'].tolist(),
            'treatment_labels': stats_df['treatment'].tolist(),
            'question_label': QUESTION_LABELS[question]
        }
        
        # Store the plot data
        all_plot_data[question] = plot_data
        
        # Create plot
        fig = go.Figure()
        
        # Reorder treatments so that control_group is always first
        treatment_order = list(stats_df['treatment'].unique())
        if control_group in treatment_order:
            treatment_order.remove(control_group)
            treatment_order = [control_group] + sorted(treatment_order)
        
        # If demographic grouping is selected, create separate traces for each demographic
        if selected_demographic != 'None' and 'demographic' in stats_df.columns:
            demographic_values = sorted(stats_df['demographic'].unique())
            demographic_colors = px.colors.qualitative.Plotly[:len(demographic_values)]
            demographic_color_map = dict(zip(demographic_values, demographic_colors))
            
            # Create offsets for each demographic to prevent overlap
            offset_amount = 0.15  # Adjust this value to control spacing
            offsets = np.linspace(-offset_amount * (len(demographic_values)-1)/2, 
                                 offset_amount * (len(demographic_values)-1)/2, 
                                 len(demographic_values))
            offset_map = dict(zip(demographic_values, offsets))
            
            # Group by demographic for the plot
            for demographic in demographic_values:
                demo_df = stats_df[stats_df['demographic'] == demographic]
                
                # Sort by treatment order
                demo_df = demo_df.sort_values(by='treatment', key=lambda x: x.map({t: i for i, t in enumerate(treatment_order)}))
                
                # Create a list of dictionaries with treatment and position
                positions = []
                for i, treatment in enumerate(demo_df['treatment']):
                    # Store the original treatment name and the position with offset
                    positions.append({
                        'original': treatment,
                        'position': i + offset_map[demographic]  # Use numeric position + offset
                    })
                
                fig.add_trace(go.Scatter(
                    x=[p['position'] for p in positions],  # Use numeric positions with offset
                    y=demo_df['mean'],
                    error_y=dict(
                        type='data',
                        array=demo_df['ci'],
                        visible=True,
                        color=demographic_color_map[demographic],
                    ),
                    mode='markers',
                    marker=dict(
                        size=12,
                        color=demographic_color_map[demographic],
                    ),
                    name=demographic,
                    text=demo_df['treatment'],  # Store original treatment names for hover
                    hovertemplate=(
                        "<b>Treatment:</b> %{text}<br>" +
                        "<b>Mean Difference:</b> %{y:.2f}<br>" +
                        "<b>Demographic:</b> " + demographic + "<br>" +
                        "<b>Sample Size:</b> " + demo_df['n'].astype(str) +
                        "<extra></extra>"
                    )
                ))
                
                # Add annotations for each point (mean with CI, and stats for treatment groups)
                max_y = max(stats_df['mean'] + stats_df['ci'])
                for i, (pos, row) in enumerate(zip(positions, demo_df.iterrows())):
                    mean_val = row[1]['mean']
                    ci_val = row[1]['ci']
                    ci_lower = mean_val - ci_val
                    ci_upper = mean_val + ci_val

                    # Build annotation text with mean and CI
                    annotation_text = f"n={row[1]['n']}<br>μ={mean_val:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]"

                    if row[1]['treatment'] != control_group and row[1]['p_value'] is not None:
                        # Add significance markers
                        if row[1]['p_value'] < 0.001:
                            symbol = '***'
                        elif row[1]['p_value'] < 0.01:
                            symbol = '**'
                        elif row[1]['p_value'] < 0.05:
                            symbol = '*'
                        elif row[1]['p_value'] < 0.1:
                            symbol = '†'
                        else:
                            symbol = 'ns'

                        # Add p-value and effect size
                        p_val = row[1]['p_value']
                        annotation_text += f"<br>p={p_val:.3f} {symbol}"

                        d_val = row[1].get('cohens_d')
                        d_ci_lower = row[1].get('d_ci_lower')
                        d_ci_upper = row[1].get('d_ci_upper')
                        effect_interp = row[1].get('effect_interpretation', '')

                        if d_val is not None and d_ci_lower is not None and d_ci_upper is not None:
                            annotation_text += f"<br>d={d_val:.2f} [{d_ci_lower:.2f}, {d_ci_upper:.2f}] ({effect_interp})"
                        elif d_val is not None:
                            annotation_text += f"<br>d={d_val:.2f} ({effect_interp})"

                    fig.add_annotation(
                        x=pos['position'],
                        y=row[1]['mean'] + row[1]['ci'] + 0.25,
                        text=annotation_text,
                        showarrow=False,
                        font=dict(size=9),
                        align='center'
                    )
            
            # Update x-axis to show treatment names at the right positions
            fig.update_layout(
                xaxis=dict(
                    tickmode='array',
                    tickvals=list(range(len(treatment_order))),
                    ticktext=treatment_order
                )
            )
            
            # Add control group reference lines for each demographic
            for demographic in demographic_values:
                demo_df = stats_df[(stats_df['demographic'] == demographic) & 
                                  (stats_df['treatment'] == control_group)]
                if not demo_df.empty:
                    control_mean = demo_df['mean'].iloc[0]
                    fig.add_shape(
                        type="line",
                        x0=-0.5,
                        y0=control_mean,
                        x1=len(treatment_order) - 0.5,
                        y1=control_mean,
                        line=dict(
                            color=demographic_color_map[demographic],
                            width=1.5,
                            dash="dash",
                        )
                    )
                    fig.add_annotation(
                        x=0,
                        y=control_mean,
                        text=f"{demographic} Control",
                        showarrow=False,
                        font=dict(size=10, color=demographic_color_map[demographic]),
                        xanchor="left",
                        yanchor="bottom",
                        xshift=5
                    )
        else:
            # Original code for treatment-only plot
            stats_df = stats_df.sort_values(by='treatment', key=lambda x: x.map({t: i for i, t in enumerate(treatment_order)}))
            
            # Add dots and error bars
            fig.add_trace(go.Scatter(
                x=stats_df['treatment'],
                y=stats_df['mean'],
                mode='markers',
                name='Mean',
                marker=dict(
                    size=12,
                    color='#1f77b4',
                ),
                error_y=dict(
                    type='data',
                    array=stats_df['ci'],
                    visible=True,
                    color='#1f77b4',
                ),
                hovertemplate=(
                    "<b>Treatment:</b> %{x}<br>" +
                    "<b>Mean Difference:</b> %{y:.2f}<br>" +
                    "<b>Sample Size:</b> " + stats_df['n'].astype(str) +
                    "<extra></extra>"
                )
            ))
            
            # Add control group reference line
            control_mean = stats_df[stats_df['treatment'] == control_group]['mean'].iloc[0]
            fig.add_hline(y=control_mean, 
                         line_dash="dash", 
                         line_color="red",
                         line_width=2,
                         annotation_text="Control Group",
                         annotation_position="top left",
                         annotation_font_size=12,
                         annotation_font_color="red"
                         )
            
            # Add annotations for each point (mean with CI, and stats for treatment groups)
            max_y = max(stats_df['mean'] + stats_df['ci'])
            for i, row in stats_df.iterrows():
                mean_val = row['mean']
                ci_val = row['ci']
                ci_lower = mean_val - ci_val
                ci_upper = mean_val + ci_val

                # Build annotation text with mean and CI
                annotation_text = f"n={row['n']}<br>μ={mean_val:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]"

                if row['treatment'] != control_group and row['p_value'] is not None:
                    # Add significance markers
                    if row['p_value'] < 0.001:
                        symbol = '***'
                    elif row['p_value'] < 0.01:
                        symbol = '**'
                    elif row['p_value'] < 0.05:
                        symbol = '*'
                    elif row['p_value'] < 0.1:
                        symbol = '†'
                    else:
                        symbol = 'ns'

                    # Add p-value and effect size
                    p_val = row['p_value']
                    annotation_text += f"<br>p={p_val:.3f} {symbol}"

                    d_val = row.get('cohens_d')
                    d_ci_lower = row.get('d_ci_lower')
                    d_ci_upper = row.get('d_ci_upper')
                    effect_interp = row.get('effect_interpretation', '')

                    if d_val is not None and d_ci_lower is not None and d_ci_upper is not None:
                        annotation_text += f"<br>d={d_val:.2f} [{d_ci_lower:.2f}, {d_ci_upper:.2f}] ({effect_interp})"
                    elif d_val is not None:
                        annotation_text += f"<br>d={d_val:.2f} ({effect_interp})"

                fig.add_annotation(
                    x=row['treatment'],
                    y=max_y + 0.25,
                    text=annotation_text,
                    showarrow=False,
                    font=dict(size=9),
                    align='center'
                )
        
        # Add legend for significance levels with more space
        fig.add_annotation(
            xref='paper',
            yref='paper',
            x=1.15,  # Moved further right
            y=1.0,
            text='†p<0.1, *p<0.05, **p<0.01, ***p<0.001, ns: not significant',
            showarrow=False,
            font=dict(size=10),
            xanchor='right',
            yanchor='top'
        )
        
        # Update title based on demographic selection
        title_text = f"Q: {QUESTION_LABELS[question]}"
        if selected_demographic != 'None':
            title_text = f"Q: {QUESTION_LABELS[question]} (by {selected_demographic})"
        
        # Update layout with adjusted margins and ranges
        fig.update_layout(
            title=dict(
                text=title_text,
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top',
                font=dict(size=20)
            ),
            xaxis_title="Treatment Group",
            yaxis_title="Mean Difference (Post - Pre)",
            xaxis=dict(
                tickangle=45,
                tickfont=dict(size=12)
            ),
            yaxis=dict(
                zeroline=True,
                zerolinewidth=2.5,
                zerolinecolor='black',
                gridcolor='lightgray',
                range=[
                    min(stats_df['mean'] - stats_df['ci']) - 0.4,  # More space at bottom
                    max_y + 0.3  # More space at top
                ]
            ),
            showlegend=(selected_demographic != 'None'),  # Only show legend when using demographics
            height=600,  # Increased height
            width=1000,  # Increased width
            template='simple_white',
            margin=dict(
                t=100,   # top margin
                b=120,   # bottom margin for n= labels
                r=150,   # right margin for significance legend
                l=80     # left margin
            )
        )
        
        # Display in Streamlit
        st.plotly_chart(fig, use_container_width=True)
        
        # Display detailed statistics
        if st.checkbox(f"Show detailed statistics for question: '_{QUESTION_LABELS[question]}_'", False):
            stats_display = stats_df.copy()
            stats_display['p_value'] = stats_display['p_value'].apply(
                lambda x: f"{x:.4f}" if x is not None else "N/A"
            )
            # Format Cohen's d
            stats_display['cohens_d'] = stats_display['cohens_d'].apply(
                lambda x: f"{x:.3f}" if x is not None else "N/A"
            )
            display_columns = ['treatment', 'mean', 'ci', 'n', 'p_value', 'cohens_d', 'effect_interpretation']
            if selected_demographic != 'None' and 'demographic' in stats_df.columns:
                display_columns.insert(1, 'demographic')

            # Rename columns for display
            stats_display = stats_display.rename(columns={
                'cohens_d': "Cohen's d",
                'effect_interpretation': 'Effect Size'
            })
            display_columns = [c if c not in ['cohens_d', 'effect_interpretation']
                             else ("Cohen's d" if c == 'cohens_d' else 'Effect Size')
                             for c in display_columns]

            st.dataframe(
                stats_display[display_columns]
                .round(3)
                .set_index('treatment')
            )

        st.divider()

    # Create a 'data' directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

def calculate_p_value(treatment_data, baseline_data, n_bootstrap=10000):
    """Calculate p-value using t-test on original data"""
    from scipy import stats

    # Perform t-test on the original data
    _, p_value = stats.ttest_ind(treatment_data, baseline_data)

    return p_value


def calculate_cohens_d(treatment_data, control_data, confidence=0.95):
    """
    Calculate Cohen's d effect size with confidence interval.

    Cohen's d = (M_treatment - M_control) / pooled_std

    Parameters:
    -----------
    treatment_data : list or array
        Treatment group values
    control_data : list or array
        Control group values
    confidence : float
        Confidence level for CI (default 0.95)

    Returns:
    --------
    dict : {d, ci_lower, ci_upper, interpretation}
    """
    treatment_data = np.array(treatment_data)
    control_data = np.array(control_data)

    n1, n2 = len(treatment_data), len(control_data)

    if n1 < 2 or n2 < 2:
        return {'d': None, 'ci_lower': None, 'ci_upper': None, 'interpretation': 'N/A'}

    mean1, mean2 = np.mean(treatment_data), np.mean(control_data)
    var1, var2 = np.var(treatment_data, ddof=1), np.var(control_data, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return {'d': None, 'ci_lower': None, 'ci_upper': None, 'interpretation': 'N/A'}

    # Cohen's d
    d = (mean1 - mean2) / pooled_std

    # Standard error of Cohen's d
    se_d = np.sqrt((n1 + n2) / (n1 * n2) + (d ** 2) / (2 * (n1 + n2)))

    # Confidence interval using t-distribution
    df = n1 + n2 - 2
    t_crit = t.ppf((1 + confidence) / 2, df)
    ci_lower = d - t_crit * se_d
    ci_upper = d + t_crit * se_d

    # Interpretation based on Cohen's conventions
    abs_d = abs(d)
    if abs_d < 0.2:
        interpretation = 'negligible'
    elif abs_d < 0.5:
        interpretation = 'small'
    elif abs_d < 0.8:
        interpretation = 'medium'
    else:
        interpretation = 'large'

    return {
        'd': d,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'interpretation': interpretation
    }


def calculate_delta_ci(treatment_data, control_data, confidence=0.95):
    """
    Calculate the difference (delta) between treatment and control means with CI.

    Parameters:
    -----------
    treatment_data : list or array
        Treatment group values
    control_data : list or array
        Control group values
    confidence : float
        Confidence level for CI (default 0.95)

    Returns:
    --------
    dict : {delta, ci_lower, ci_upper, treatment_mean, treatment_ci_lower, treatment_ci_upper,
            control_mean, control_ci_lower, control_ci_upper}
    """
    treatment_data = np.array(treatment_data)
    control_data = np.array(control_data)

    n1, n2 = len(treatment_data), len(control_data)

    if n1 < 2 or n2 < 2:
        return {
            'delta': None, 'delta_ci_lower': None, 'delta_ci_upper': None,
            'treatment_mean': None, 'treatment_ci_lower': None, 'treatment_ci_upper': None,
            'control_mean': None, 'control_ci_lower': None, 'control_ci_upper': None
        }

    mean1, mean2 = np.mean(treatment_data), np.mean(control_data)
    var1, var2 = np.var(treatment_data, ddof=1), np.var(control_data, ddof=1)

    # Delta (difference)
    delta = mean1 - mean2

    # Standard error of the difference
    se_delta = np.sqrt(var1/n1 + var2/n2)

    # Welch-Satterthwaite degrees of freedom
    df = ((var1/n1 + var2/n2)**2) / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))

    # CI for delta
    t_crit = t.ppf((1 + confidence) / 2, df)
    delta_ci_lower = delta - t_crit * se_delta
    delta_ci_upper = delta + t_crit * se_delta

    # CI for treatment mean
    se1 = np.sqrt(var1/n1)
    t_crit1 = t.ppf((1 + confidence) / 2, n1 - 1)
    treatment_ci_lower = mean1 - t_crit1 * se1
    treatment_ci_upper = mean1 + t_crit1 * se1

    # CI for control mean
    se2 = np.sqrt(var2/n2)
    t_crit2 = t.ppf((1 + confidence) / 2, n2 - 1)
    control_ci_lower = mean2 - t_crit2 * se2
    control_ci_upper = mean2 + t_crit2 * se2

    return {
        'delta': delta,
        'delta_ci_lower': delta_ci_lower,
        'delta_ci_upper': delta_ci_upper,
        'treatment_mean': mean1,
        'treatment_ci_lower': treatment_ci_lower,
        'treatment_ci_upper': treatment_ci_upper,
        'control_mean': mean2,
        'control_ci_lower': control_ci_lower,
        'control_ci_upper': control_ci_upper
    }