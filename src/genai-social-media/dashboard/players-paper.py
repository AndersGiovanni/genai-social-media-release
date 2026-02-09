import streamlit as st
import pandas as pd
import json
import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

st.set_page_config(layout="wide")

def flatten_survey_data(df):
    """
    Flatten the nested dictionary structures in initialSurvey and exitSurvey
    into separate columns for each survey question.
    
    Args:
        df: DataFrame with initialSurvey and exitSurvey columns containing nested dicts
        
    Returns:
        DataFrame with flattened survey data
    """
    # Create a copy of the DataFrame with just id and treatmentName
    result_df = df[['id', 'treatmentName']].copy()
    
    # Function to flatten a nested dictionary with keys joined by underscores
    def flatten_dict(d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                # Convert lists to string representation to avoid the iterable error
                if isinstance(v, list):
                    v = str(v)  # Convert list to string
                items.append((new_key, v))
        return dict(items)
    
    # Process initialSurvey and exitSurvey for each row
    for idx, row in df.iterrows():
        # Process initialSurvey
        if isinstance(row.get('initialSurvey'), dict):
            flattened = flatten_dict(row['initialSurvey'], 'initialSurvey')
            for key, value in flattened.items():
                result_df.loc[idx, key] = value
                
        # Process exitSurvey
        if isinstance(row.get('exitSurvey'), dict):
            # Create a copy of exitSurvey without userRating
            exit_survey_copy = row['exitSurvey'].copy()
            if 'userRating' in exit_survey_copy:
                del exit_survey_copy['userRating']
                
            flattened = flatten_dict(exit_survey_copy, 'exitSurvey')
            for key, value in flattened.items():
                result_df.loc[idx, key] = value
    
    return result_df

# Load and preprocess data
@st.cache_data
def load_data(file_path='data/combining/combined', combine_treatments=False, filter_users_using_treatments=False):
    # Handle case where file_path is a list of uploaded files
    if isinstance(file_path, list):
        file_path = 'data/combining/combined'  # Use default path if given a list
    
    # Load all necessary data
    df_player = pd.read_csv(os.path.join(file_path, 'player.csv'))
    df_game = pd.read_csv(os.path.join(file_path, 'game.csv'))
    df_batch = pd.read_csv(os.path.join(file_path, 'batch.csv'))
    
    # Filter for ended batches
    ended_batch_ids = df_batch[df_batch['status'] == 'ended']['id'].unique()
    
    # Filter games where both batch is ended and game is ended
    valid_game_ids = df_game[
        (df_game['batchID'].isin(ended_batch_ids)) & 
        (df_game['ended'] == True)
    ]['id'].unique()
    
    # Filter players based on valid games
    df_player = df_player[df_player['gameID'].isin(valid_game_ids)]
    
    # Additional filtering and processing
    df_player = df_player[df_player['ended'] == 'game ended']
    df_ate = df_player.copy()
    
    # Process survey data
    df_player['initialSurvey'] = df_player['initialSurvey'].apply(lambda x: json.loads(x) if pd.notna(x) else {})
    df_ate['initialSurvey'] = df_ate['initialSurvey'].apply(lambda x: json.loads(x) if pd.notna(x) else {})
    df_ate['exitSurvey'] = df_ate['exitSurvey'].apply(lambda x: json.loads(x) if pd.notna(x) else {})
    df_player['exitSurvey'] = df_player['exitSurvey'].apply(lambda x: json.loads(x) if isinstance(x, str) and x else {})
    
    # Convert timestamps
    df_player['gameIDLastChangedAt'] = pd.to_datetime(df_player['gameIDLastChangedAt'])
    df_player['introDoneLastChangedAt'] = pd.to_datetime(df_player['introDoneLastChangedAt'])
    
    # Keep only users using treatments
    if filter_users_using_treatments:
        with open('data/combining/combined/users_using_treatment.json', 'r') as f:
            users_using_treatments = json.load(f)
            
        # Create a mask that keeps baseline_5 rows and rows where id is in the corresponding treatment list
        mask_player = (df_player['treatmentName'] == 'baseline_5') | \
                     df_player.apply(lambda row: row['id'] in users_using_treatments.get(row['treatmentName'], []), axis=1)
        
        mask_at = (df_ate['treatmentName'] == 'baseline_5') | \
                 df_ate.apply(lambda row: row['id'] in users_using_treatments.get(row['treatmentName'], []), axis=1)
        
        # Apply the masks to filter the dataframes
        df_player = df_player[mask_player]
        df_ate = df_ate[mask_at]

    # Combine treatments if requested
    if combine_treatments:
        df_player['treatmentName'] = df_player['treatmentName'].apply(
            lambda x: x if x == 'baseline_5' else 'treatment'
        )
        df_ate['treatmentName'] = df_ate['treatmentName'].apply(
            lambda x: x if x == 'baseline_5' else 'treatment'
        )

    return df_player, df_ate

df, df_ate = load_data()



# New function to create and save publication-quality plots for Nature SI
def create_nature_si_plots(df, save_path='figures/initial-survey'):
    """
    Create publication-quality plots formatted for Nature SI and save them to disk.
    
    Args:
        df: DataFrame with survey data
        save_path: Path to save the generated figures
    """
    # Ensure the output directory exists
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # Set up the style for publication-quality plots - updated for modern Seaborn
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 8,
        'axes.titlesize': 10,
        'axes.labelsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 8,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })
    
    # Define questions for each category
    demographic_questions = {
        'age': 'What is your age group?',
        'gender': 'How do you identify your gender?',
        'education': 'What is the highest level of education you have completed?',
        'occupation': 'What is your current occupation or employment status?',
        'partyAffiliation': 'What is your political party affiliation?'
    }
    
    social_media_questions = {
        'socialMediaPlatforms': 'Which social media platforms do you use most frequently?',
        'healthcare': 'I seek information about health and wellbeing\npractices on online social media platforms.',
        'usPolitics': 'I stay informed about current\nevents and issues in US politics.',
        'onlineParticipation': 'I often participate in online\ndiscussions on social media.',
        'easyEngagement': 'It is easy for me to engage in online\nconversations.',
        'informativeDiscussion': 'I find discussions on social media\ninformative and high-quality.',
        'trustInformation': 'I trust the information shared by other\nusers on social media.',
        'changeOpinion': 'I recall changing my opinion based on\ninteractions on social media.',
        'agreeOpinions': 'I tend to agree with the\nopinions I see on social media.',
        'barriersPosting': 'Do you face any barriers when posting content on social media?'
    }
    
    ai_questions = {
        'comfortableWithAI': 'I feel comfortable with AI being\nused on social media platforms.',
        'aiSuggestions': 'AI suggestions can make it more\nlikely for me to participate in online discussions.',
        'lessToxic': 'AI can make online discussions\nmore positive and less toxic.',
        'lessPolarizing': 'AI can make discussions less polarizing.',
        'reduceMisinformation': 'AI can help reduce misinformation on social media.',
        'aiContentAccurate': 'AI-generated content is accurate and reliable.',
        'aiRegulation': 'AI should be regulated to prevent\nmisuse and ensure ethical use.'
    }
    
    # Process and create plots for each category
    process_survey_category(df, 'demographicInfo', demographic_questions, "Demographics", save_path)
    process_survey_category(df, 'socialMediaInfo', social_media_questions, "Social Media Usage", save_path)
    process_survey_category(df, 'aiAssessmentInfo', ai_questions, "AI Assessment", save_path)
    

def process_survey_category(df, category, questions_dict, title, save_path):
    """
    Process each survey category and create publication-quality plots for each question.
    
    Args:
        df: DataFrame with survey data
        category: Category of questions to process
        questions_dict: Dictionary of question keys and text
        title: Title for the category
        save_path: Path to save the generated figures
    """
    # Process each question in the category
    for question_key, question_text in questions_dict.items():
        # Create a clean filename from the question key
        filename = f"{category}_{question_key}"
        
        # Get responses for this question
        responses_data = []
        
        for _, row in df.iterrows():
            treatment = row['treatmentName']
            if isinstance(row['initialSurvey'], dict) and category in row['initialSurvey']:
                response = row['initialSurvey'][category].get(question_key, 'EMPTY')
                
                # Handle list responses (e.g., for multiple choice questions)
                if isinstance(response, list):
                    if not response:  # Empty list
                        responses_data.append({'Treatment': treatment, 'Response': 'EMPTY'})
                    else:
                        for item in response:
                            responses_data.append({'Treatment': treatment, 'Response': item if item else 'EMPTY'})
                else:
                    # Handle single responses
                    responses_data.append({
                        'Treatment': treatment, 
                        'Response': response if response and response != '' else 'EMPTY'
                    })
        
        response_df = pd.DataFrame(responses_data)
        
        # Check if this is a Likert-scale question
        is_likert = check_if_likert(response_df)
        
        if is_likert:
            # Create a publication-quality plot for Likert-scale questions
            create_likert_plot(response_df, question_text, filename, save_path)
        else:
            # Create a publication-quality plot for categorical questions
            create_categorical_plot(response_df, question_text, filename, save_path)

def check_if_likert(response_df):
    """
    Check if a question uses Likert scale responses.
    
    Args:
        response_df: DataFrame with responses
        
    Returns:
        bool: True if it's a Likert-scale question, False otherwise
    """
    likert_options = {
        'Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree',
        'Very Poor', 'Poor', 'Fair', 'Good', 'Excellent'
    }
    
    unique_responses = set(response_df['Response'].unique())
    return len(unique_responses.intersection(likert_options)) >= 3

def create_likert_plot(response_df, question_text, filename, save_path, show_significance=True):
    """
    Create a publication-quality plot for Likert-scale questions.
    
    Args:
        response_df: DataFrame with responses
        question_text: Text of the question
        filename: Base filename for saving
        save_path: Path to save the generated figure
        show_significance: Whether to show significance markers (default: True)
    """
    # Map Likert responses to numeric values
    likert_mapping = {
        'Strongly disagree': 1, 'Disagree': 2, 'Neutral': 3, 'Agree': 4, 'Strongly agree': 5,
        'Very Poor': 1, 'Poor': 2, 'Fair': 3, 'Good': 4, 'Excellent': 5
    }
    
    # Add numeric values for responses
    response_df['Value'] = response_df['Response'].map(likert_mapping)
    response_df = response_df.dropna(subset=['Value'])
    
    # Calculate statistics by treatment
    stats_df = response_df.groupby('Treatment')['Value'].agg(['mean', 'count', 'std']).reset_index()
    
    # Calculate confidence intervals (95%)
    stats_df['ci'] = stats_df.apply(
        lambda x: 1.96 * x['std'] / np.sqrt(x['count']) if x['count'] > 0 else 0, 
        axis=1
    )
    
    # Map treatment names to readable labels
    treatment_mapping = {
        'baseline_5': 'Control',
        'chat_5': 'Chat',
        'conversation_5': 'Conversation',
        'feedback_5': 'Feedback',
        'suggestions_5': 'Suggestions'
    }
    
    stats_df['Treatment_Label'] = stats_df['Treatment'].map(treatment_mapping)
    
    # Initialize ordered_treatments to ensure it's always defined
    ordered_treatments = []
    
    # Reorder treatments so Control (baseline_5) comes first
    all_treatments = list(stats_df['Treatment'].unique())
    if 'baseline_5' in all_treatments:
        all_treatments.remove('baseline_5')
        ordered_treatments = ['baseline_5'] + sorted(all_treatments)
        stats_df['Treatment'] = pd.Categorical(
            stats_df['Treatment'], 
            categories=ordered_treatments, 
            ordered=True
        )
        stats_df = stats_df.sort_values('Treatment')
    else:
        ordered_treatments = sorted(all_treatments)
        stats_df['Treatment'] = pd.Categorical(
            stats_df['Treatment'], 
            categories=ordered_treatments, 
            ordered=True
        )
    
    # Calculate p-values (comparing each treatment with baseline) if show_significance is True
    # and baseline_5 exists in the data
    if show_significance and 'baseline_5' in response_df['Treatment'].values:
        baseline_values = response_df[response_df['Treatment'] == 'baseline_5']['Value'].values
        p_values = []
        for treatment in stats_df['Treatment']:
            if treatment == 'baseline_5':
                p_values.append(1.0)  # Not applicable for baseline
            else:
                treatment_values = response_df[response_df['Treatment'] == treatment]['Value'].values
                _, p_val = stats.ttest_ind(treatment_values, baseline_values, equal_var=False)
                p_values.append(p_val)
        
        stats_df['p_value'] = p_values
    else:
        # No significance testing
        stats_df['p_value'] = 1.0  # Set all p-values to 1.0 (not significant)
    
    # Create the figure with FIXED dimensions for consistency
    fig, ax = plt.subplots(figsize=(4.5, 3.0), dpi=300)
    
    # Set high-quality Nature-appropriate styling
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.linewidth'] = 0.6
    plt.rcParams['xtick.major.width'] = 0.6
    plt.rcParams['ytick.major.width'] = 0.6
    plt.rcParams['xtick.minor.width'] = 0.6
    plt.rcParams['ytick.minor.width'] = 0.6
    
    # Use a more scientific color palette
    colors = ['#2878B5', '#9AC9DB', '#C82423', '#F8AC8C', '#6F4C9B']
    color_dict = {t: colors[i % len(colors)] for i, t in enumerate(ordered_treatments)}
    
    # Create the point plot with error bars
    for i, row in stats_df.iterrows():
        ax.errorbar(
            i, row['mean'], 
            yerr=row['ci'], 
            fmt='o', 
            capsize=4, 
            color=color_dict[row['Treatment']], 
            markersize=8,  # Slightly smaller for consistency
            ecolor=color_dict[row['Treatment']],
            elinewidth=2.0,  # Slightly thinner
            capthick=2,
            alpha=0.8
        )
    
    # Add significance markers if show_significance is True
    if show_significance:
        for i, row in stats_df.iterrows():
            if row['Treatment'] != 'baseline_5':
                p_val = row['p_value']
                if p_val < 0.001:
                    sig_marker = '***'
                elif p_val < 0.01:
                    sig_marker = '**'
                elif p_val < 0.05:
                    sig_marker = '*'
                elif p_val < 0.1:
                    sig_marker = '†'
                else:
                    sig_marker = ''
                    
                ax.annotate(
                    sig_marker, 
                    xy=(i, row['mean']), 
                    xytext=(0, 12),  # Reduced from 15 for tighter layout
                    textcoords='offset points',
                    ha='center', 
                    va='bottom',
                    fontsize=9,
                    fontweight='bold'
                )
    
    # Annotate points with sample sizes
    for i, row in stats_df.iterrows():
        ax.annotate(
            f"n={row['count']}", 
            xy=(i, row['mean']), 
            xytext=(0, -12),
            textcoords='offset points',
            ha='center', 
            va='top',
            fontsize=8,  # Slightly smaller
            alpha=0.95
        )
    
    # Set axis labels and title with controlled sizing
    ax.set_ylabel('Mean Response (1-5)', fontsize=9, fontweight='bold')
    ax.set_xlabel('', fontsize=9)
    
    # Truncate title if too long to maintain consistent layout
    if len(question_text) > 60:
        title_text = question_text[:57] + "..."
    else:
        title_text = question_text
    
    ax.set_title(title_text, fontsize=10, fontweight='bold', pad=8, wrap=True)
    
    # Set x-ticks with mapped treatment labels
    ax.set_xticks(range(len(stats_df)))
    ax.set_xticklabels([row['Treatment_Label'] for _, row in stats_df.iterrows()], 
                       fontsize=8)  # Consistent font size
    
    # Customize y-axis with fixed limits
    ax.set_ylim(1, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    
    # Add Likert scale labels
    if 'Strongly disagree' in response_df['Response'].values:
        ax.set_yticklabels(['Strongly\ndisagree', 'Disagree', 'Neutral', 'Agree', 'Strongly\nagree'], 
                          fontsize=7)
    else:
        ax.set_yticklabels(['Very Poor', 'Poor', 'Fair', 'Good', 'Excellent'], 
                          fontsize=7)
    
    # Add grid lines for better readability
    ax.grid(axis='both', linestyle='--', alpha=0.05, linewidth=0.3)

    # Add reference lines with better styling
    for y in range(1, 6):
        ax.axhline(y=y, linestyle='--', alpha=0.3, color='gray', linewidth=0.7)
    
    # Add vertical grid lines at each treatment position
    for x in range(len(stats_df)):
        ax.axvline(x=x, linestyle='--', alpha=0.3, color='gray', linewidth=0.7)
    
    # Add box around the plot
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(0.5)
    
    # Rotate x-tick labels for better readability
    plt.xticks(rotation=0, ha='center')
    
    # Use FIXED subplot parameters for consistent layout
    plt.subplots_adjust(
        left=0.15,    # Fixed left margin
        bottom=0.15,  # Fixed bottom margin  
        right=0.95,   # Fixed right margin
        top=0.85      # Fixed top margin
    )
    
    # Save the figure in multiple formats
    for ext in ['pdf']:  # 'pdf', 'svg', 'eps'
        output_path = os.path.join(save_path, f"{filename}.{ext}")
        plt.savefig(output_path, bbox_inches=None, dpi=300)  # Don't use tight_bbox
    
    plt.close()

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
    return ""

def create_categorical_plot(response_df, question_text, filename, save_path, show_significance=True):
    """
    Create a publication-quality plot for categorical questions.
    
    Args:
        response_df: DataFrame with responses
        question_text: Text of the question
        filename: Base filename for saving
        save_path: Path to save the generated figure
        show_significance: Whether to show significance markers (default: True)
    """
    # Count responses by treatment and response
    counts = response_df.groupby(['Treatment', 'Response']).size().reset_index(name='Count')
    
    # Calculate total counts by treatment for percentages
    totals = response_df.groupby('Treatment').size().reset_index(name='Total')
    counts = counts.merge(totals, on='Treatment')
    counts['Percentage'] = 100 * counts['Count'] / counts['Total']
    
    # Map treatment names to readable labels
    treatment_mapping = {
        'baseline_5': 'Control',
        'chat_5': 'Chat',
        'conversation_5': 'Conversation',
        'feedback_5': 'Feedback',
        'suggestions_5': 'Suggestions'
    }
    
    counts['Treatment_Label'] = counts['Treatment'].map(treatment_mapping)
    
    # Filter out 'EMPTY' responses for cleaner plots
    if 'EMPTY' in counts['Response'].values:
        counts = counts[counts['Response'] != 'EMPTY']
    
    # Filter out 'Under 18' responses for age demographic question
    if question_text == "What is your age group?":
        counts = counts[counts['Response'] != 'Under 18']
    
    # Simplify long text responses
    if question_text == "What is the highest level of education you have completed?":
        counts['Response'] = counts['Response'].apply(
            lambda x: "Postgraduate degree" if isinstance(x, str) and "Postgraduate degree" in x else x
        )
    
    if question_text == "What is your political party affiliation?":
        counts['Response'] = counts['Response'].apply(
            lambda x: "Other" if isinstance(x, str) and "Other" in x else x
        )
    
    # Sort responses by frequency for non-age questions
    if question_text == "What is your age group?":
        age_order = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
        # Filter to only include ages that are present in the data
        response_order = [age for age in age_order if age in counts['Response'].unique()]
    else:
        response_order = counts.groupby('Response')['Count'].sum().sort_values(ascending=False).index.tolist()
    
    # Limit to top 7 responses for readability
    if len(response_order) > 7:
        top_responses = response_order[:7]
        counts = counts[counts['Response'].isin(top_responses)]
        response_order = top_responses
    
    # Initialize ordered_treatments to ensure it's always defined
    ordered_treatments = []
    
    # Reorder treatments so Control (baseline_5) comes first
    all_treatments = list(counts['Treatment'].unique())
    if 'baseline_5' in all_treatments:
        all_treatments.remove('baseline_5')
        ordered_treatments = ['baseline_5'] + sorted(all_treatments)
        counts['Treatment'] = pd.Categorical(
            counts['Treatment'], 
            categories=ordered_treatments, 
            ordered=True
        )
    else:
        ordered_treatments = sorted(all_treatments)
        counts['Treatment'] = pd.Categorical(
            counts['Treatment'], 
            categories=ordered_treatments, 
            ordered=True
        )
    
    # Create figure with FIXED dimensions for consistency
    fig, ax = plt.subplots(figsize=(5.5, 3.0), dpi=300)  # Slightly wider for categorical
    
    # Set high-quality Nature-appropriate styling
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.linewidth'] = 0.6
    plt.rcParams['xtick.major.width'] = 0.6
    plt.rcParams['ytick.major.width'] = 0.6
    
    # Use a scientific color palette
    colors = ['#2878B5', '#9AC9DB', '#C82423', '#F8AC8C', '#6F4C9B']
    color_dict = {treatment: colors[i % len(colors)] for i, treatment in enumerate(ordered_treatments)}
    
    # Calculate significance relative to baseline for each response
    # Only if show_significance is True and baseline_5 is in the data
    p_values = {}
    if show_significance and 'baseline_5' in counts['Treatment'].values:
        baseline_proportions = {}
        for response in response_order:
            baseline_data = counts[(counts['Treatment'] == 'baseline_5') & (counts['Response'] == response)]
            if not baseline_data.empty:
                baseline_count = baseline_data['Count'].values[0]
                baseline_total = baseline_data['Total'].values[0]
                baseline_proportions[response] = (baseline_count, baseline_total)
        
        # Create a nested dictionary to store p-values
        for response in response_order:
            p_values[response] = {}
            if response in baseline_proportions:
                baseline_count, baseline_total = baseline_proportions[response]
                for treatment in ordered_treatments:
                    if treatment != 'baseline_5':
                        treatment_data = counts[(counts['Treatment'] == treatment) & (counts['Response'] == response)]
                        if not treatment_data.empty:
                            treat_count = treatment_data['Count'].values[0]
                            treat_total = treatment_data['Total'].values[0]
                            
                            # Use Fisher's exact test for proportions
                            table = [[treat_count, treat_total - treat_count], 
                                     [baseline_count, baseline_total - baseline_count]]
                            _, p_val = stats.fisher_exact(table)
                            p_values[response][treatment] = p_val
    
    # Plot each response as a grouped bar chart
    bar_width = 0.15
    index = np.arange(len(response_order))
    
    for i, treatment in enumerate(ordered_treatments):
        treatment_data = counts[counts['Treatment'] == treatment]
        percentages = []
        response_counts = []
        
        for response in response_order:
            response_data = treatment_data[treatment_data['Response'] == response]
            if not response_data.empty:
                percentages.append(response_data['Percentage'].values[0])
                response_counts.append(response_data['Count'].values[0])
            else:
                percentages.append(0)
                response_counts.append(0)
        
        # Calculate position
        pos = index + (i - len(ordered_treatments)/2 + 0.5) * bar_width
        
        # Create bars
        ax.bar(
            pos, 
            percentages, 
            bar_width, 
            alpha=0.8, 
            color=color_dict[treatment],
            edgecolor='black',
            linewidth=0.5,
            label=treatment_mapping.get(treatment, treatment)
        )

    # Add plot labels and legend with controlled sizing
    ax.set_ylabel('Percentage (%)', fontsize=9, fontweight='bold')
    ax.set_xlabel('Response', fontsize=9, fontweight='bold')
    
    # Truncate title if too long to maintain consistent layout
    if len(question_text) > 50:  # Shorter limit for categorical plots
        title_text = question_text[:47] + "..."
    else:
        title_text = question_text
    
    ax.set_title(title_text, fontsize=10, fontweight='bold', pad=8)
    
    # Set x-axis tick labels to the response names with consistent formatting
    ax.set_xticks(index)
    # Truncate long response labels
    truncated_labels = []
    for response in response_order:
        if len(str(response)) > 12:
            truncated_labels.append(str(response)[:9] + "...")
        else:
            truncated_labels.append(str(response))
    
    ax.set_xticklabels(truncated_labels, rotation=45, ha='right', fontsize=8)
    
    # Add a legend with consistent positioning
    ax.legend(
        title='Treatment',
        loc='upper right',
        fontsize=8,
        title_fontsize=8
    )
    
    # Add grid lines for better readability
    ax.grid(axis='both', linestyle='--', alpha=0.3, linewidth=0.7)

    # Add horizontal reference lines
    for y in [20, 40, 60, 80]:
        ax.axhline(y=y, linestyle='--', alpha=0.3, color='gray', linewidth=0.7)
    
    # Add vertical grid lines for each response
    for x in index:
        ax.axvline(x=x, linestyle='--', alpha=0.3, color='gray', linewidth=0.7)
    
    # Add box around the plot
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(0.5)
    
    # Use FIXED subplot parameters for consistent layout
    plt.subplots_adjust(
        left=0.12,    # Fixed left margin
        bottom=0.25,  # More space for rotated labels
        right=0.95,   # Fixed right margin
        top=0.85      # Fixed top margin
    )
    
    # Save the figure
    for ext in ['pdf']:  # 'pdf', 'svg', 'eps'
        output_path = os.path.join(save_path, f"{filename}.{ext}")
        plt.savefig(output_path, bbox_inches=None, dpi=300)  # Don't use tight_bbox
    
    plt.close()

# Add a function to call from the Streamlit interface
def generate_nature_si_plots():
    """Generate publication-quality plots for Nature SI and save them to disk"""
    with st.spinner("Generating Nature SI publication plots..."):
        create_nature_si_plots(df)
        st.success("✅ Nature SI plots generated successfully in 'figures/initial-survey' folder")
    
def is_likert_question(responses):
    """Check if the question uses Likert scale responses (supports both default and new scales)"""
    default_likert = {
        'Strongly disagree', 'Disagree', 'Neutral', 
        'Agree', 'Strongly agree', 'Prefer not to answer'
    }
    new_likert = {
        'Very Poor', 'Poor', 'Fair', 'Good', 'Excellent', 'Prefer not to answer'
    }
    unique_responses = set(responses['Response'].unique())
    return (len(unique_responses.intersection(default_likert)) >= 3 or
           (len(unique_responses.intersection(new_likert)) >= 3))

def get_likert_type(responses):
    """Determine the likert scale type used.
    Returns 'new' if the new scale is detected; otherwise returns 'default'."""
    new_likert = {
        'Very Poor', 'Poor', 'Fair', 'Good', 'Excellent', 'Prefer not to answer'
    }
    unique_responses = set(responses['Response'].unique())
    if len(unique_responses.intersection(new_likert)) >= 3:
        return 'new'
    return 'default'

def likert_to_numeric(response, scale_type='default'):
    """Convert Likert scale responses to numeric values based on the scale type"""
    if scale_type == 'new':
        mapping = {
            'Very Poor': 1,
            'Poor': 2,
            'Fair': 3,
            'Good': 4,
            'Excellent': 5
        }
    else:
        mapping = {
            'Strongly disagree': 1,
            'Disagree': 2,
            'Neutral': 3,
            'Agree': 4,
            'Strongly agree': 5
        }
    # For responses like "Prefer not to answer" or any unmapped response, return None
    return mapping.get(response, None)



# New function to create and save publication-quality plots for Exit Survey
def create_exit_survey_si_plots(df, save_path='figures/exit-survey'):
    """
    Create publication-quality plots for the Exit Survey formatted for Nature SI.
    
    Args:
        df: DataFrame with survey data
        save_path: Path to save the generated figures
    """
    # Ensure the output directory exists
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # Set up the style for publication-quality plots
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 8,
        'axes.titlesize': 10,
        'axes.labelsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 8,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })
    
    # Define questions for each category in the exit survey
    experience_questions = {
        'overallExperience': 'Rate your overall experience with the platform.',
        'participation': 'I participated in the discussions more\nthan I usually do on social media.',
        'engagement': 'It was easy for me to engage in the conversations.',
        'informative': 'I found the comments from other participants\nto be informative and high-quality.',
        'trust': 'I trust the information provided by other participants.',
        'agreement': 'I tended to agree with the other participants.',
        'time': 'I feel like I had enough time for each conversation.',
        'barriers': 'Did you face any barriers when posting content?',
    }
    
    ai_eval_questions = {
        'easyParticipation': 'The AI made it easier for me to participate in\nthe discussion, compared to my\n usual experience on social media.',
        'naturalDiscussion': 'The AI felt natural in the context of the discussions.',
        'contentQuality': 'The content I created using the AI was high-quality.',
        'usefulness': 'I can imagine situations in which\nI would use this AI if it was available on social media.',
    }
    
    social_media_questions = {
        'aiComfortability': 'I feel comfortable with AI being\nused on social media platforms.',
        'aiParticipation': 'AI suggestions can make it more likely for\nme to participate in online discussions.',
        'aiLessToxic': 'AI can make online discussions\nmore positive and less toxic.',
        'aiLessPolarizing': 'AI can make discussions less polarizing.',
        'aiReducesMisinformation': 'AI can help reduce misinformation on social media.',
        'aiAccuracy': 'AI-generated content is accurate and reliable.',
        'aiRegulation': 'AI should be regulated to prevent\nmisuse and ensure ethical use.',
    }
    
    # Process and create plots for each category
    process_exit_survey_category(df, 'overallExperience', experience_questions, "Overall Experience", save_path)
    process_exit_survey_category(df, 'aiEvaluation', ai_eval_questions, "AI Evaluation", save_path)
    process_exit_survey_category(df, 'socialMediaAI', social_media_questions, "Social Media AI Perceptions", save_path)
    

def process_exit_survey_category(df, category, questions_dict, title, save_path):
    """
    Process each exit survey category and create publication-quality plots for each question.
    
    Args:
        df: DataFrame with survey data
        category: Category of questions to process
        questions_dict: Dictionary of question keys and text
        title: Title for the category
        save_path: Path to save the generated figures
    """
    # Process each question in the category
    for question_key, question_text in questions_dict.items():
        # Create a clean filename from the question key
        filename = f"{category}_{question_key}"
        
        # Get responses for this question
        responses_data = []
        
        for _, row in df.iterrows():
            treatment = row['treatmentName']
            if isinstance(row['exitSurvey'], dict) and category in row['exitSurvey']:
                response = row['exitSurvey'][category].get(question_key, 'EMPTY')
                
                # Handle list responses (e.g., for multiple choice questions)
                if isinstance(response, list):
                    if not response:  # Empty list
                        responses_data.append({'Treatment': treatment, 'Response': 'EMPTY'})
                    else:
                        for item in response:
                            responses_data.append({'Treatment': treatment, 'Response': item if item else 'EMPTY'})
                else:
                    # Handle single responses
                    responses_data.append({
                        'Treatment': treatment, 
                        'Response': response if response and response != '' else 'EMPTY'
                    })
        
        response_df = pd.DataFrame(responses_data)
        
        # Check if this is a Likert-scale question
        is_likert = check_if_likert(response_df)
        
        if is_likert:
            # Create a publication-quality plot for Likert-scale questions
            # For aiEvaluation category, don't show significance markers because there's no control group
            show_significance = category != 'aiEvaluation'
            create_likert_plot(response_df, question_text, filename, save_path, show_significance=show_significance)
        else:
            # Create a publication-quality plot for categorical questions
            # For aiEvaluation category, don't show significance markers because there's no control group
            show_significance = category != 'aiEvaluation'
            create_categorical_plot(response_df, question_text, filename, save_path, show_significance=show_significance)

# Add a function to generate exit survey plots from the Streamlit interface
def generate_exit_survey_si_plots():
    """Generate publication-quality plots for Exit Survey and save them to disk"""
    create_exit_survey_si_plots(df)

# Add a function to create treatment effects plots for Nature publication
def create_treatment_effects_plots(treatment_data, save_path='figures/treatment-effects', 
                                  control_group='baseline_5', 
                                  confidence=0.95, 
                                  n_bootstrap=10000):
    """
    Create publication-quality treatment effects plots for Nature SI.
    
    Args:
        treatment_data: Dictionary of treatment data
        save_path: Path to save the generated figures
        control_group: Control group name
        confidence: Confidence level for intervals (default: 0.95)
        n_bootstrap: Number of bootstrap samples (default: 10000)
    """
    from players_helpers.ate import QUESTION_PAIRS, QUESTION_LABELS, calculate_treatment_stats
    
    # Ensure the output directory exists
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # Set up the style for publication-quality plots
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 8,
        'axes.titlesize': 10,
        'axes.labelsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 8,
        'figure.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })
    
    # Process each question
    for question in QUESTION_PAIRS.keys():
        # Get statistics
        stats_df = calculate_treatment_stats(
            treatment_data, 
            control_group, 
            question,
            True,  # use bootstrap
            confidence,
            n_bootstrap,
            'None'  # no demographic grouping
        )
        
        # Clean filename based on question
        question_text = QUESTION_LABELS[question]
        filename = f"treatment_effect_{question.replace(' ', '_')}"
        
        # Reorder treatments so Control (baseline_5) comes first
        all_treatments = list(stats_df['treatment'].unique())
        if control_group in all_treatments:
            all_treatments.remove(control_group)
            ordered_treatments = [control_group] + sorted(all_treatments)
            stats_df['treatment'] = pd.Categorical(
                stats_df['treatment'], 
                categories=ordered_treatments,
                ordered=True
            )
            stats_df = stats_df.sort_values('treatment')
        else:
            ordered_treatments = sorted(all_treatments)
        
        # Map treatment names to readable labels
        treatment_mapping = {
            'baseline_5': 'Control',
            'chat_5': 'Chat',
            'conversation_5': 'Conversation',
            'feedback_5': 'Feedback',
            'suggestions_5': 'Suggestions'
        }
        
        stats_df['Treatment_Label'] = stats_df['treatment'].map(treatment_mapping)
        
        # Create the figure with appropriate dimensions for Nature SI
        fig, ax = plt.subplots(figsize=(4.5, 3.0), dpi=300)  # Standardized dimensions
        
        # Set high-quality Nature-appropriate styling
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 8
        plt.rcParams['axes.linewidth'] = 0.8
        plt.rcParams['xtick.major.width'] = 0.8
        plt.rcParams['ytick.major.width'] = 0.8
        plt.rcParams['xtick.minor.width'] = 0.6
        plt.rcParams['ytick.minor.width'] = 0.6
        
        # Use a more scientific color palette
        colors = ['#2878B5', '#9AC9DB', '#C82423', '#F8AC8C', '#6F4C9B']
        color_dict = {t: colors[i % len(colors)] for i, t in enumerate(ordered_treatments)}
        
        # Create the point plot with error bars
        for i, row in stats_df.iterrows():
            ax.errorbar(
                i, row['mean'], 
                yerr=row['ci'], 
                fmt='o', 
                capsize=4, 
                color=color_dict[row['Treatment']], 
                markersize=8,  # Standardized smaller size
                ecolor=color_dict[row['Treatment']],
                elinewidth=2.0,  # Standardized thinner line
                capthick=2,
                alpha=0.8
            )
        
        # Add significance markers
        for i, row in stats_df.iterrows():
            if row['Treatment'] != 'baseline_5':
                p_val = row['p_value']
                if p_val < 0.001:
                    sig_marker = '***'
                elif p_val < 0.01:
                    sig_marker = '**'
                elif p_val < 0.05:
                    sig_marker = '*'
                elif p_val < 0.1:
                    sig_marker = '†'
                else:
                    sig_marker = ''
                    
                ax.annotate(
                    sig_marker, 
                    xy=(i, row['mean']), 
                    xytext=(0, 12),  # Reduced spacing for consistency
                    textcoords='offset points',
                    ha='center', 
                    va='bottom',
                    fontsize=9,
                    fontweight='bold'
                )
        
        # Annotate points with sample sizes
        for i, row in stats_df.iterrows():
            ax.annotate(
                f"n={row['count']}", 
                xy=(i, row['mean']), 
                xytext=(0, -12),  # Reduced spacing for consistency
                textcoords='offset points',
                ha='center', 
                va='top',
                fontsize=8,  # Smaller font for consistency
                alpha=0.95
            )
        
        # Set axis labels and title with controlled sizing
        ax.set_ylabel('Average Rating (1-5)', fontsize=9, fontweight='bold')
        ax.set_xlabel('', fontsize=9)
        ax.set_title(question_title, fontsize=10, fontweight='bold', pad=8)
        
        # Set x-ticks with mapped treatment labels and consistent font size
        ax.set_xticks(range(len(stats_df)))
        ax.set_xticklabels([row['Treatment_Label'] for _, row in stats_df.iterrows()],
                          fontsize=8)
        
        # Customize y-axis with fixed limits
        ax.set_ylim(1, 5)
        ax.set_yticks([1, 2, 3, 4, 5])
        
        # Add grid
        ax.grid(axis='both', linestyle='--', alpha=0.3, linewidth=0.7)
        
        # Add reference lines with better styling
        for y in range(1, 6):
            ax.axhline(y=y, linestyle='--', alpha=0.3, color='gray', linewidth=0.7)
            
        # Add vertical grid lines for each treatment
        for x in range(len(stats_df)):
            ax.axvline(x=x, linestyle='--', alpha=0.3, color='gray', linewidth=0.7)
        
        # Add box around the plot with consistent styling
        for spine in ['top', 'right', 'left', 'bottom']:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_linewidth(0.5)
        
        # Use FIXED subplot parameters for consistent layout
        plt.subplots_adjust(
            left=0.15,    # Fixed left margin
            bottom=0.15,  # Fixed bottom margin  
            right=0.95,   # Fixed right margin
            top=0.85      # Fixed top margin
        )
        
        # Save the figure with consistent parameters
        for ext in ['pdf']:
            output_path = os.path.join(save_path, f"participant_{question_key}_ratings.{ext}")
            plt.savefig(output_path, bbox_inches=None, dpi=300)  # No tight_bbox for consistency
        
        plt.close()

# Add a function to generate treatment effects plots from the streamlit interface
def generate_treatment_effects_si_plots():
    """Generate publication-quality treatment effects plots for Nature SI"""
    from players_helpers.ate import get_treatment_data
    
    treatment_data = get_treatment_data(df_ate)
    create_treatment_effects_plots(treatment_data)

# Add a function to create user ratings plots for Nature publication
def create_user_ratings_plots(df, save_path='figures/user-ratings'):
    """
    Create publication-quality user ratings plots for Nature SI.
    
    Args:
        df: DataFrame with survey data
        save_path: Path to save the generated figures
    """
    # Ensure the output directory exists
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # Set up the style for publication-quality plots
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 8,
        'axes.titlesize': 10,
        'axes.labelsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 8,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'grid.linestyle': '--',
        'grid.linewidth': 0.7,
        'grid.alpha': 0.3,
    })
    
    # Create plots for comment ratings
    create_comment_ratings_plot(df, save_path)
    
    # Create plots for participant ratings (positivity, engagement, politeness)
    create_participant_ratings_plots(df, save_path)
    
    # Create plots for categorical ratings (politicalAffiliation, sharedValues, isBot, usedAI)
    create_categorical_ratings_plots(df, save_path)
    

def create_comment_ratings_plot(df, save_path):
    """
    Create a publication-quality plot for comment ratings.
    
    Args:
        df: DataFrame with survey data
        save_path: Path to save the generated figure
    """
    # Extract and process comment ratings
    comment_ratings = []
    for _, row in df.iterrows():
        if isinstance(row['exitSurvey'], dict) and 'userRating' in row['exitSurvey']:
            # Process ratings
            ratings = row['exitSurvey']['userRating'].get('commentRatings', {})
            for comment_id, rating_data in ratings.items():
                if 'valueComment' in rating_data:
                    try:
                        rating = float(rating_data['valueComment'])
                        comment_ratings.append({
                            'Treatment': row['treatmentName'],
                            'Rating': rating
                        })
                    except (ValueError, TypeError):
                        continue

    ratings_df = pd.DataFrame(comment_ratings)
    
    if ratings_df.empty:
        return
    
    # Calculate statistics by treatment
    stats_df = ratings_df.groupby('Treatment')['Rating'].agg(['mean', 'count', 'std']).reset_index()
    
    # Calculate confidence intervals (95%)
    stats_df['ci'] = stats_df.apply(
        lambda x: 1.96 * x['std'] / np.sqrt(x['count']) if x['count'] > 0 else 0, 
        axis=1
    )
    
    # Map treatment names to readable labels
    treatment_mapping = {
        'baseline_5': 'Control',
        'chat_5': 'Chat',
        'conversation_5': 'Conversation',
        'feedback_5': 'Feedback',
        'suggestions_5': 'Suggestions'
    }
    
    stats_df['Treatment_Label'] = stats_df['Treatment'].map(treatment_mapping)
    
    # Reorder treatments so Control (baseline_5) comes first
    all_treatments = list(stats_df['Treatment'].unique())
    if 'baseline_5' in all_treatments:
        all_treatments.remove('baseline_5')
        ordered_treatments = ['baseline_5'] + sorted(all_treatments)
        stats_df['Treatment'] = pd.Categorical(
            stats_df['Treatment'], 
            categories=ordered_treatments, 
            ordered=True
        )
        stats_df = stats_df.sort_values('Treatment')
    else:
        ordered_treatments = sorted(all_treatments)
        stats_df['Treatment'] = pd.Categorical(
            stats_df['Treatment'], 
            categories=ordered_treatments, 
            ordered=True
        )
    
    # Calculate p-values (comparing each treatment with baseline)
    # and baseline_5 exists in the data
    if 'baseline_5' in ratings_df['Treatment'].values:
        baseline_values = ratings_df[ratings_df['Treatment'] == 'baseline_5']['Rating'].values
        p_values = []
        for treatment in stats_df['Treatment']:
            if treatment == 'baseline_5':
                p_values.append(1.0)  # Not applicable for baseline
            else:
                treatment_values = ratings_df[ratings_df['Treatment'] == treatment]['Rating'].values
                _, p_val = stats.ttest_ind(treatment_values, baseline_values, equal_var=False)
                p_values.append(p_val)
        
        stats_df['p_value'] = p_values
    else:
        # No significance testing
        stats_df['p_value'] = 1.0  # Set all p-values to 1.0 (not significant)
    
    # Create the figure with FIXED dimensions for consistency
    fig, ax = plt.subplots(figsize=(4.5, 3.0), dpi=300)
    
    # Set high-quality Nature-appropriate styling
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8
    plt.rcParams['xtick.minor.width'] = 0.6
    plt.rcParams['ytick.minor.width'] = 0.6
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.linewidth'] = 0.7
    plt.rcParams['grid.alpha'] = 0.3
    
    # Use a more scientific color palette
    colors = ['#2878B5', '#9AC9DB', '#C82423', '#F8AC8C', '#6F4C9B']
    color_dict = {t: colors[i % len(colors)] for i, t in enumerate(ordered_treatments)}
    
    # Create the point plot with error bars
    for i, row in stats_df.iterrows():
        ax.errorbar(
            i, row['mean'], 
            yerr=row['ci'], 
            fmt='o', 
            capsize=4, 
            color=color_dict[row['Treatment']], 
            markersize=8,
            ecolor=color_dict[row['Treatment']],
            elinewidth=2.0,
            capthick=2,
            alpha=0.8
        )
    
    # Add significance markers
    for i, row in stats_df.iterrows():
        if row['Treatment'] != 'baseline_5':
            p_val = row['p_value']
            if p_val < 0.001:
                sig_marker = '***'
            elif p_val < 0.01:
                sig_marker = '**'
            elif p_val < 0.05:
                sig_marker = '*'
            elif p_val < 0.1:
                sig_marker = '†'
            else:
                sig_marker = 'ns'
                
            ax.annotate(
                sig_marker, 
                xy=(i, row['mean']), 
                xytext=(0, 12),
                textcoords='offset points',
                ha='center', 
                va='bottom',
                fontsize=9,
                fontweight='bold'
            )
    
    # Annotate points with sample sizes
    for i, row in stats_df.iterrows():
        ax.annotate(
            f"n={row['count']}", 
            xy=(i, row['mean']), 
            xytext=(0, -12),
            textcoords='offset points',
            ha='center', 
            va='top',
            fontsize=8,
            alpha=0.95
        )
    
    # Set axis labels and title with controlled sizing
    ax.set_ylabel('Average Rating (1-5)', fontsize=9, fontweight='bold')
    ax.set_xlabel('', fontsize=9)
    ax.set_title('Comment Ratings by Treatment', fontsize=10, fontweight='bold', pad=8)
    
    # Set x-ticks with mapped treatment labels
    ax.set_xticks(range(len(stats_df)))
    ax.set_xticklabels([row['Treatment_Label'] for _, row in stats_df.iterrows()],
                       fontsize=8)
    
    # Customize y-axis with fixed limits
    ax.set_ylim(1, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    
    # Add grid
    ax.grid(axis='both', linestyle='--', alpha=0.3, linewidth=0.7)
    
    # Add reference lines with better styling
    for y in range(1, 6):
        ax.axhline(y=y, linestyle='--', alpha=0.3, color='gray', linewidth=0.7)
        
    # Add vertical grid lines for each treatment
    for x in range(len(stats_df)):
        ax.axvline(x=x, linestyle='--', alpha=0.3, color='gray', linewidth=0.7)
    
    # Add box around the plot
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(0.5)
    
    # Rotate x-tick labels for better readability
    plt.xticks(rotation=0, ha='center')
    
    # Use FIXED subplot parameters for consistent layout
    plt.subplots_adjust(
        left=0.15,    # Fixed left margin
        bottom=0.15,  # Fixed bottom margin  
        right=0.95,   # Fixed right margin
        top=0.85      # Fixed top margin
    )
    
    # Save the figure
    for ext in ['pdf']:
        output_path = os.path.join(save_path, f"comment_ratings.{ext}")
        plt.savefig(output_path, bbox_inches=None, dpi=300)  # Don't use tight_bbox
    
    plt.close()

def create_participant_ratings_plots(df, save_path):
    """
    Create publication-quality plots for participant ratings.
    
    Args:
        df: DataFrame with survey data
        save_path: Path to save the generated figures
    """
    # Likert-scale questions to analyze
    likert_questions = {
        'positivity': 'Participant Positivity Ratings',
        'engagement': 'Participant Engagement Ratings',
        'politeness': 'Participant Politeness Ratings'
    }
    
    for question_key, question_title in likert_questions.items():
        # Extract and process participant ratings
        participant_ratings = []
        for _, row in df.iterrows():
            if isinstance(row['exitSurvey'], dict) and 'userRating' in row['exitSurvey']:
                ratings = row['exitSurvey']['userRating'].get('participantRatings', {})
                for participant_data in ratings.values():
                    if question_key in participant_data:
                        try:
                            rating = float(participant_data[question_key])
                            participant_ratings.append({
                                'Treatment': row['treatmentName'],
                                'Rating': rating
                            })
                        except (ValueError, TypeError):
                            continue
        
        ratings_df = pd.DataFrame(participant_ratings)
        
        if ratings_df.empty:
            continue
        
        # Calculate statistics by treatment
        stats_df = ratings_df.groupby('Treatment')['Rating'].agg(['mean', 'count', 'std']).reset_index()
        
        # Calculate confidence intervals (95%)
        stats_df['ci'] = stats_df.apply(
            lambda x: 1.96 * x['std'] / np.sqrt(x['count']) if x['count'] > 0 else 0, 
            axis=1
        )
        
        # Map treatment names to readable labels
        treatment_mapping = {
            'baseline_5': 'Control',
            'chat_5': 'Chat',
            'conversation_5': 'Conversation',
            'feedback_5': 'Feedback',
            'suggestions_5': 'Suggestions'
        }
        
        stats_df['Treatment_Label'] = stats_df['Treatment'].map(treatment_mapping)
        
        # Reorder treatments so Control (baseline_5) comes first
        all_treatments = list(stats_df['Treatment'].unique())
        if 'baseline_5' in all_treatments:
            all_treatments.remove('baseline_5')
            ordered_treatments = ['baseline_5'] + sorted(all_treatments)
            stats_df['Treatment'] = pd.Categorical(
                stats_df['Treatment'], 
                categories=ordered_treatments, 
                ordered=True
            )
            stats_df = stats_df.sort_values('Treatment')
        else:
            ordered_treatments = sorted(all_treatments)
            stats_df['Treatment'] = pd.Categorical(
                stats_df['Treatment'], 
                categories=ordered_treatments, 
                ordered=True
            )
        
        # Calculate p-values (comparing each treatment with baseline)
        if 'baseline_5' in ratings_df['Treatment'].values:
            baseline_values = ratings_df[ratings_df['Treatment'] == 'baseline_5']['Rating'].values
            p_values = []
            for treatment in stats_df['Treatment']:
                if treatment == 'baseline_5':
                    p_values.append(1.0)  # Not applicable for baseline
                else:
                    treatment_values = ratings_df[ratings_df['Treatment'] == treatment]['Rating'].values
                    _, p_val = stats.ttest_ind(treatment_values, baseline_values, equal_var=False)
                    p_values.append(p_val)
            
            stats_df['p_value'] = p_values
        else:
            # No significance testing
            stats_df['p_value'] = 1.0  # Set all p-values to 1.0 (not significant)
        
        # Create the figure with appropriate dimensions for Nature SI
        fig, ax = plt.subplots(figsize=(6, 3.5), dpi=300)
        
        # Set high-quality Nature-appropriate styling
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 8
        plt.rcParams['axes.linewidth'] = 0.8
        plt.rcParams['xtick.major.width'] = 0.8
        plt.rcParams['ytick.major.width'] = 0.8
        plt.rcParams['xtick.minor.width'] = 0.6
        plt.rcParams['ytick.minor.width'] = 0.6
        
        # Use a more scientific color palette
        colors = ['#2878B5', '#9AC9DB', '#C82423', '#F8AC8C', '#6F4C9B']
        color_dict = {t: colors[i % len(colors)] for i, t in enumerate(ordered_treatments)}
        
        # Create the point plot with error bars
        for i, row in stats_df.iterrows():
            ax.errorbar(
                i, row['mean'], 
                yerr=row['ci'], 
                fmt='o', 
                capsize=4, 
                color=color_dict[row['Treatment']], 
                markersize=9,
                ecolor=color_dict[row['Treatment']],
                elinewidth=2.5,
                capthick=2,
                alpha=0.8
            )
        
        # Add significance markers
        for i, row in stats_df.iterrows():
            if row['Treatment'] != 'baseline_5':
                p_val = row['p_value']
                if p_val < 0.001:
                    sig_marker = '***'
                elif p_val < 0.01:
                    sig_marker = '**'
                elif p_val < 0.05:
                    sig_marker = '*'
                elif p_val < 0.1:
                    sig_marker = '†'
                else:
                    sig_marker = ''
                    
                ax.annotate(
                    sig_marker, 
                    xy=(i, row['mean']), 
                    xytext=(0, 15),
                    textcoords='offset points',
                    ha='center', 
                    va='bottom',
                    fontsize=10,
                    fontweight='bold'
                )
        
        # Annotate points with sample sizes
        for i, row in stats_df.iterrows():
            ax.annotate(
                f"n={row['count']}", 
                xy=(i, row['mean']), 
                xytext=(0, -15),
                textcoords='offset points',
                ha='center', 
                va='top',
                fontsize=9,
                alpha=0.95
            )
        
        # Set axis labels and title with better formatting
        ax.set_ylabel('Average Rating (1-5)', fontsize=9, fontweight='bold')
        ax.set_xlabel('', fontsize=9)
        ax.set_title(question_title, fontsize=10, fontweight='bold', pad=10)
        
        # Set x-ticks with mapped treatment labels
        ax.set_xticks(range(len(stats_df)))
        ax.set_xticklabels([row['Treatment_Label'] for _, row in stats_df.iterrows()])
        
        # Customize y-axis
        ax.set_ylim(1, 5)
        ax.set_yticks([1, 2, 3, 4, 5])
        
        # Add grid
        ax.grid(axis='both', linestyle='--', alpha=0.3, linewidth=0.7)
        
        # Add reference lines with better styling
        for y in range(1, 6):
            ax.axhline(y=y, linestyle='--', alpha=0.3, color='gray', linewidth=0.7)
            
        # Add vertical grid lines for each treatment
        for x in range(len(stats_df)):
            ax.axvline(x=x, linestyle='--', alpha=0.3, color='gray', linewidth=0.7)
        
        # Add box around the plot
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['top'].set_linewidth(0.5)
        ax.spines['right'].set_linewidth(0.5)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure
        for ext in ['pdf']:
            output_path = os.path.join(save_path, f"participant_{question_key}_ratings.{ext}")
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
        
        plt.close()

def create_categorical_ratings_plots(df, save_path):
    """
    Create publication-quality scatter plots for categorical participant ratings.
    
    Args:
        df: DataFrame with survey data
        save_path: Path to save the generated figures
    """
    # Categorical questions to analyze
    categorical_questions = {
        'politicalAffiliation': 'Perceived Political Affiliation',
        'sharedValues': 'Shared Values with Participants',
        'isBot': 'Perceived as Bot',
        'usedAI': 'Perceived as Using AI'
    }
    
    for question_key, question_title in categorical_questions.items():
        # Extract and process categorical ratings
        categorical_data = []
        for _, row in df.iterrows():
            if isinstance(row['exitSurvey'], dict) and 'userRating' in row['exitSurvey']:
                ratings = row['exitSurvey']['userRating'].get('participantRatings', {})
                for participant_data in ratings.values():
                    if question_key in participant_data:
                        categorical_data.append({
                            'Treatment': row['treatmentName'],
                            'Response': str(participant_data[question_key])  # Convert to string to handle various types
                        })
        
        df_cat = pd.DataFrame(categorical_data)
        
        if df_cat.empty:
            continue
        
        # Get unique responses for this question
        response_order = df_cat.groupby('Response').size().sort_values(ascending=False).index.tolist()
        if len(response_order) > 5:
            # Limit to top 5 responses for readability
            top_responses = response_order[:5]
            response_order = top_responses
            # Filter data to only include top responses
            df_cat = df_cat[df_cat['Response'].isin(top_responses)]
        
        # Map treatment names to readable labels
        treatment_mapping = {
            'baseline_5': 'Control',
            'chat_5': 'Chat',
            'conversation_5': 'Conversation',
            'feedback_5': 'Feedback',
            'suggestions_5': 'Suggestions'
        }
        
        # Reorder treatments so Control (baseline_5) comes first
        all_treatments = list(df_cat['Treatment'].unique())
        if 'baseline_5' in all_treatments:
            all_treatments.remove('baseline_5')
            ordered_treatments = ['baseline_5'] + sorted(all_treatments)
        else:
            ordered_treatments = sorted(all_treatments)
        
        # Aggregate data to calculate proportions and statistics
        agg_data = []
        
        for treatment in ordered_treatments:
            treatment_data = df_cat[df_cat['Treatment'] == treatment]
            total = len(treatment_data)
            
            for response in response_order:
                response_count = sum(treatment_data['Response'] == response)
                if total > 0:
                    proportion = response_count / total * 100
                    
                    # Calculate bootstrap confidence intervals
                    binary_data = (treatment_data['Response'] == response).astype(int).values
                    
                    # Bootstrap CI calculation
                    np.random.seed(42)
                    n_bootstrap = 10000
                    bootstrap_props = np.zeros(n_bootstrap)
                    
                    for i in range(n_bootstrap):
                        bootstrap_sample = np.random.choice(binary_data, size=len(binary_data), replace=True)
                        bootstrap_props[i] = np.mean(bootstrap_sample) * 100
                    
                    ci_lower, ci_upper = np.percentile(bootstrap_props, [2.5, 97.5])
                    
                    agg_data.append({
                        'Treatment': treatment,
                        'Response': response,
                        'Count': response_count,
                        'Total': total,
                        'Proportion': proportion,
                        'CI_Lower': proportion - ci_lower,
                        'CI_Upper': ci_upper - proportion
                    })
        
        # Convert to DataFrame
        agg_data = pd.DataFrame(agg_data)
        
        # Compare proportions to baseline (control) using t-test
        if 'baseline_5' in ordered_treatments:
            for response in response_order:
                # Get baseline data for this response
                baseline_rows = agg_data[(agg_data['Treatment'] == 'baseline_5') & 
                                       (agg_data['Response'] == response)]
                if not baseline_rows.empty:
                    baseline_row = baseline_rows.iloc[0]
                    baseline_prop = baseline_row['Proportion']
                    baseline_count = baseline_row['Count']
                    baseline_total = baseline_row['Total']
                    
                    # Create the binary data for baseline
                    baseline_binary = np.zeros(baseline_total)
                    baseline_binary[:baseline_count] = 1
                    
                    # Compare each treatment to baseline
                    for treatment in ordered_treatments:
                        if treatment != 'baseline_5':
                            treatment_rows = agg_data[(agg_data['Treatment'] == treatment) & 
                                                    (agg_data['Response'] == response)]
                            if not treatment_rows.empty:
                                treatment_row = treatment_rows.iloc[0]
                                treatment_count = treatment_row['Count']
                                treatment_total = treatment_row['Total']
                                
                                # Create binary data for treatment
                                treatment_binary = np.zeros(treatment_total)
                                treatment_binary[:treatment_count] = 1
                                
                                # Calculate p-value using t-test
                                _, p_value = stats.ttest_ind(treatment_binary, baseline_binary, equal_var=False)
                                
                                # Update agg_data with p-value
                                agg_data.loc[(agg_data['Treatment'] == treatment) & 
                                           (agg_data['Response'] == response), 'p_value'] = p_value
        
        # Create the figure - wider format to accommodate responses on x-axis
        fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=300)  # More standardized size
        
        # Set high-quality Nature-appropriate styling
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 8  # Smaller font for consistency
        plt.rcParams['axes.linewidth'] = 0.8
        plt.rcParams['xtick.major.width'] = 0.8
        plt.rcParams['ytick.major.width'] = 0.8
        
        # Use a scientific color palette for treatments
        colors = ['#2878B5', '#9AC9DB', '#C82423', '#F8AC8C', '#6F4C9B']
        treatment_color_map = {treatment: colors[i % len(colors)] for i, treatment in enumerate(ordered_treatments)}
        
        # Plot each response as a separate group on x-axis
        x_positions = {}
        x_ticks = []
        x_labels = []
        
        for i, response in enumerate(response_order):
            response_stats = agg_data[agg_data['Response'] == response]
            
            # Calculate positions for this response group
            base_pos = i * (len(ordered_treatments) + 2.0)  # +1 for spacing between groups
            x_ticks.append(base_pos + (len(ordered_treatments) - 1) / 2)  # Center of this group
            x_labels.append(response)
            
            # Plot each treatment within this response group
            for j, treatment in enumerate(ordered_treatments):
                treatment_data = response_stats[response_stats['Treatment'] == treatment]
                if not treatment_data.empty:
                    # Calculate position
                    x_pos = base_pos + j
                    x_positions[(response, treatment)] = x_pos
                    
                    row = treatment_data.iloc[0]
                    # Plot point with error bars
                    ax.errorbar(
                        x_pos, row['Proportion'],
                        yerr=[[row['CI_Lower']], [row['CI_Upper']]],
                        fmt='o',
                        capsize=3,
                        color=treatment_color_map[treatment],
                        markersize=7,
                        elinewidth=1.5,
                        capthick=1.5,
                        alpha=0.8,
                        label=treatment_mapping.get(treatment, treatment) if i == 0 else None  # Only label once
                    )
                    
                    # Add sample size annotation below point with increased spacing
                    ax.annotate(
                        f"n={row['Count']}",
                        xy=(x_pos, row['Proportion']),
                        xytext=(0, -25),  # Increased from -15 to -25
                        textcoords='offset points',
                        ha='center',
                        va='top',
                        fontsize=7
                    )
                    
                    # Add significance markers if not baseline
                    if treatment != 'baseline_5' and 'p_value' in row:
                        p_val = row['p_value']
                        sig_marker = get_stars(p_val)
                        ax.annotate(
                            sig_marker,
                            xy=(x_pos, row['Proportion']),
                            xytext=(0, 25),  # Increased from 15 to 25
                            textcoords='offset points',
                            ha='center',
                            va='bottom',
                            fontsize=9,
                            fontweight='bold'
                        )
        
        # Set axis labels and title
        ax.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
        ax.set_title(question_title, fontsize=12, fontweight='bold', pad=10)
        
        # Set x-ticks at the center of each response group
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=12)
        
        # Add vertical separators between response groups
        # for i in range(len(response_order) - 1):
        #     separator_pos = (i + 1) * (len(ordered_treatments) + 1) - 0.5
        #     ax.axvline(x=separator_pos, color='gray', linestyle='--', alpha=0.3, linewidth=0.7)
        
        # Add horizontal and vertical grid lines
        # ax.grid(axis='both', linestyle='--', alpha=0.3, linewidth=0.7)
        
        # Add legend for treatments
        ax.legend(
            title="Treatments",
            loc='upper right',
            fontsize=12,
            # title_fontsize=10
        )
        
        # Add box around the plot
        for spine in ['top', 'right', 'left', 'bottom']:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_linewidth(0.5)
        
        # Add significance legend
        # ax.annotate("†p<0.1, *p<0.05, **p<0.01, ***p<0.001, ns: not significant",
        #           xy=(1.0, -0.12), 
        #           xycoords='axes fraction',
        #           ha='right', 
        #           fontsize=7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure
        filename = f"categorical_{question_key}"
        for ext in ['pdf']:
            output_path = os.path.join(save_path, f"{filename}.{ext}")
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
        
        plt.close()

# Add a function to generate user ratings plots from the streamlit interface
def generate_user_ratings_si_plots():
    """Generate publication-quality user ratings plots for Nature SI"""
    create_user_ratings_plots(df)

# Main dashboard
def main():
    from players_helpers.ate import get_treatment_data
    create_user_ratings_plots(df)

if __name__ == "__main__":
    main()