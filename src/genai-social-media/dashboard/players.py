import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
import os
from players_helpers.ate import get_treatment_data, plot_treatment_effects, calculate_cohens_d
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
import re
from statsmodels.miscmodels.ordinal_model import OrderedModel

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

    # For making files
    flattened_df_player = flatten_survey_data(df_player)
    # Write to csv
    flattened_df_player.to_csv('data/combining/combined/flattened_player_data.csv', index=False)
    
    return df_player, df_ate

df, df_ate = load_data()


# Key Metrics
def display_key_metrics():
    st.header("Key Metrics")
    
    # First row: Basic metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Number of Players", len(df))
    
    with col2:
        # Treatment distribution
        treatment_counts = df['treatmentName'].value_counts().sort_index()
        st.subheader("Players by Treatment")
        
        # Create a simple table showing the distribution
        treatment_df = pd.DataFrame({
            'Treatment': treatment_counts.index,
            'Count': treatment_counts.values,
            'Percentage': (treatment_counts.values / len(df) * 100).round(1)
        })
        
        # Format the display
        treatment_df['Display'] = treatment_df.apply(
            lambda row: f"{row['Count']} ({row['Percentage']}%)", axis=1
        )
        
        # Show as metrics or table
        for _, row in treatment_df.iterrows():
            st.metric(row['Treatment'], row['Display'])
    
    # Second row: Demographics
    st.subheader("Demographics Overview")
    
    # Extract demographic data from initialSurvey
    gender_data = []
    age_data = []
    education_data = []
    occupation_data = []
    party_data = []
    
    for _, row in df.iterrows():
        if isinstance(row['initialSurvey'], dict) and 'demographicInfo' in row['initialSurvey']:
            demo_info = row['initialSurvey']['demographicInfo']
            
            # Extract gender
            gender = demo_info.get('gender', 'Unknown')
            if gender and gender != 'Prefer not to answer':
                gender_data.append(gender)
            else:
                gender_data.append('Unknown/Prefer not to answer')
            
            # Extract age
            age = demo_info.get('age', 'Unknown')
            if age and age != 'Prefer not to answer':
                age_data.append(age)
            else:
                age_data.append('Unknown/Prefer not to answer')
            
            # Extract education
            education = demo_info.get('education', 'Unknown')
            if education and education != 'Prefer not to answer':
                education_data.append(education)
            else:
                education_data.append('Unknown/Prefer not to answer')
            
            # Extract occupation
            occupation = demo_info.get('occupation', 'Unknown')
            # Handle occupation as list or string
            if isinstance(occupation, list) and len(occupation) > 0:
                occupation = occupation[0]
            if occupation and occupation != 'Prefer not to answer':
                occupation_data.append(str(occupation))
            else:
                occupation_data.append('Unknown/Prefer not to answer')
            
            # Extract party affiliation
            party = demo_info.get('partyAffiliation', 'Unknown')
            if party and party != 'Prefer not to answer':
                # Simplify party names for cleaner display
                if 'Republican' in party:
                    party_data.append('Republican')
                elif 'Democrat' in party:
                    party_data.append('Democrat')
                elif 'Independent' in party:
                    party_data.append('Independent')
                else:
                    party_data.append(str(party))
            else:
                party_data.append('Unknown/Prefer not to answer')
        else:
            gender_data.append('Unknown/Prefer not to answer')
            age_data.append('Unknown/Prefer not to answer')
            education_data.append('Unknown/Prefer not to answer')
            occupation_data.append('Unknown/Prefer not to answer')
            party_data.append('Unknown/Prefer not to answer')
    
    # First row: Gender and Age
    demo_row1_col1, demo_row1_col2 = st.columns(2)
    
    with demo_row1_col1:
        st.subheader("Gender")
        gender_counts = pd.Series(gender_data).value_counts()
        gender_percentages = (gender_counts / len(df) * 100).round(1)
        
        for gender, count in gender_counts.items():
            percentage = gender_percentages[gender]
            st.metric(gender, f"{count} ({percentage}%)")
    
    with demo_row1_col2:
        st.subheader("Age")
        age_counts = pd.Series(age_data).value_counts()
        age_percentages = (age_counts / len(df) * 100).round(1)
        
        for age, count in age_counts.items():
            percentage = age_percentages[age]
            st.metric(age, f"{count} ({percentage}%)")
    
    # Second row: Education and Occupation
    demo_row2_col1, demo_row2_col2 = st.columns(2)
    
    with demo_row2_col1:
        st.subheader("Education")
        education_counts = pd.Series(education_data).value_counts()
        education_percentages = (education_counts / len(df) * 100).round(1)
        
        for education, count in education_counts.items():
            percentage = education_percentages[education]
            # Truncate long education labels for better display
            display_education = education if len(education) <= 25 else education[:22] + "..."
            st.metric(display_education, f"{count} ({percentage}%)")
    
    with demo_row2_col2:
        st.subheader("Occupation")
        occupation_counts = pd.Series(occupation_data).value_counts()
        occupation_percentages = (occupation_counts / len(df) * 100).round(1)
        
        for occupation, count in occupation_counts.items():
            percentage = occupation_percentages[occupation]
            # Truncate long occupation labels for better display
            display_occupation = occupation if len(occupation) <= 20 else occupation[:17] + "..."
            st.metric(display_occupation, f"{count} ({percentage}%)")
    
    # Third row: Party Affiliation (centered in single column)
    demo_row3_col1, demo_row3_col2, demo_row3_col3 = st.columns([1, 2, 1])
    
    with demo_row3_col2:
        st.subheader("Party Affiliation")
        party_counts = pd.Series(party_data).value_counts()
        party_percentages = (party_counts / len(df) * 100).round(1)
        
        for party, count in party_counts.items():
            percentage = party_percentages[party]
            st.metric(party, f"{count} ({percentage}%)")
    
    # Optional: Add a visualization
    st.subheader("Treatment Distribution Visualization")
    fig = px.pie(
        values=treatment_counts.values,
        names=treatment_counts.index,
        title="Distribution of Players Across Treatments"
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

def display_initial_survey():
    st.header("Initial Survey Results")
    st.markdown("This section presents the results from the initial survey completed by participants.")
    
    # Add filters
    # df, selected_game_ids, selected_treatments, _ = create_filters(
    #     df, 
    #     include_participant=False, 
    #     key_prefix="initial_survey_tab"
    # )

    # Calculate the number of participants who finished the intro questionnaire
    total_participants = len(df)
    
    # More explicit check for survey completion
    answered_survey = df['initialSurveyFinished'].fillna(False).astype(bool).sum()
    
    fraction_answered_survey = answered_survey / total_participants if total_participants > 0 else 0
    st.metric("Intro Survey Completion Rate", f"{fraction_answered_survey:.2%}")
    
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
        'healthcare': 'I seek information about health and wellbeing practices on online social media platforms.',
        'usPolitics': 'I stay informed about current events and issues in US politics.',
        'onlineParticipation': 'I often participate in online discussions on social media.',
        'easyEngagement': 'It is easy for me to engage in online conversations',
        'informativeDiscussion': 'I find discussions on social media informative and high-quality',
        'trustInformation': 'I trust the information shared by other users on social media.',
        'changeOpinion': 'I recall changing my opinion based on interactions on social media.',
        'agreeOpinions': 'I tend to agree with the opinions I see on social media.',
        'barriersPosting': 'Do you face any barriers when posting content on social media?'
    }
    
    ai_questions = {
        'comfortableWithAI': 'I feel comfortable with AI being used on social media platforms.',
        'aiSuggestions': 'AI suggestions can make it more likely for me to participate in online discussions.',
        'lessToxic': 'AI can make online discussions more positive and less toxic.',
        'lessPolarizing': 'AI can make discussions less polarizing.',
        'reduceMisinformation': 'AI can help reduce misinformation on social media.',
        'aiContentAccurate': 'AI-generated content is accurate and reliable.',
        'aiRegulation': 'AI should be regulated to prevent misuse and ensure ethical use.'
    }
    
    st.divider()

    # Plot each category
    plot_category_questions(df, 'demographicInfo', demographic_questions, "Demographics")
    plot_category_questions(df, 'socialMediaInfo', social_media_questions, "Social Media Usage")
    plot_category_questions(df, 'aiAssessmentInfo', ai_questions, "AI Assessment")
    
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

def plot_category_questions(df, category, questions_dict, title):
    st.subheader(title)
    
    # Define the desired order for the default Likert-scale responses (used for bar charts)
    likert_order = [
        'Strongly disagree', 
        'Disagree', 
        'Neutral', 
        'Agree', 
        'Strongly agree', 
        'Prefer not to answer', 
        'EMPTY'
    ]
    
    # Add demographic grouping option
    demographic_options = ['None', 'age', 'gender', 'education', 'occupation', 'partyAffiliation']
    selected_demographic = st.selectbox(
        "Group by demographic (optional):", 
        demographic_options,
        key=f"demographic_select_{category}"
    )
    
    # Get all possible responses for consistent coloring
    all_possible_responses = set()
    for question_key in questions_dict.keys():
        responses = df['initialSurvey'].apply(
            lambda x: x.get(category, {}).get(question_key, 'EMPTY')
            if isinstance(x, dict) else 'EMPTY'
        ).replace('', 'EMPTY').fillna('EMPTY')
        
        if responses.apply(lambda x: isinstance(x, list)).any():
            responses = [item if item else 'EMPTY'
                         for sublist in responses
                         if isinstance(sublist, list)
                         for item in (sublist if sublist else ['EMPTY'])]
        
        all_possible_responses.update(responses)
    
    color_sequence = px.colors.qualitative.D3[:len(df['treatmentName'].unique())]
    color_map = dict(zip(sorted(df['treatmentName'].unique()), color_sequence))
    
    # Plot each question
    for question_key, question_text in questions_dict.items():
        data_list = []
        
        # First, collect the responses for this question
        for treatment in df['treatmentName'].unique():
            treatment_df = df[df['treatmentName'] == treatment]
            
            responses = treatment_df['initialSurvey'].apply(
                lambda x: x.get(category, {}).get(question_key, 'EMPTY')
                if isinstance(x, dict) else 'EMPTY'
            ).replace('', 'EMPTY').fillna('EMPTY')
            
            if responses.apply(lambda x: isinstance(x, list)).any():
                flattened_data = [item if item else 'EMPTY'
                                  for sublist in responses
                                  if isinstance(sublist, list)
                                  for item in (sublist if sublist else ['EMPTY'])]
                responses = pd.Series(flattened_data)
            
            # Add demographic information if selected
            if selected_demographic != 'None':
                for i, response in enumerate(responses):
                    if i < len(treatment_df):
                        demographic_value = 'Unknown'
                        if selected_demographic in ['age', 'gender', 'education', 'occupation', 'partyAffiliation']:
                            # Extract demographic from initialSurvey
                            demo_value = treatment_df.iloc[i]['initialSurvey'].get('demographicInfo', {}).get(selected_demographic, '')
                            
                            # Special handling for party affiliation
                            if selected_demographic == 'partyAffiliation' and demo_value:
                                if 'Republican' in demo_value:
                                    demo_value = 'Republican'
                                elif 'Democrat' in demo_value:
                                    demo_value = 'Democrat'
                            
                            if demo_value and demo_value != 'Prefer not to answer':
                                demographic_value = demo_value
                        
                        data_list.append({
                            'Treatment': treatment,
                            'Response': response,
                            'Demographic': demographic_value
                        })
                    else:
                        data_list.append({
                            'Treatment': treatment,
                            'Response': response,
                            'Demographic': 'Unknown'
                        })
            else:
                for response in responses:
                    data_list.append({
                        'Treatment': treatment,
                        'Response': response,
                        'Demographic': 'All'  # Default when no demographic is selected
                    })
        
        # Create initial DataFrame with collected responses
        plot_df = pd.DataFrame(data_list)
        
        # Filter out 'Unknown' and 'Prefer not to answer' demographics
        if selected_demographic != 'None':
            plot_df = plot_df[~plot_df['Demographic'].isin(['Unknown', 'Prefer not to answer', ''])]
        
        # Check whether the question is Likert-based (either default or new scale)
        if is_likert_question(plot_df):
            # Determine which Likert scale is being used
            scale_type = get_likert_type(plot_df)
            point_data_dict = {}

            np.random.seed(42)
            n_bootstrap = 10000

            # Compute mean, bootstrap confidence intervals, and hold bootstrap distributions
            if selected_demographic != 'None':
                # Group by both treatment and demographic
                for treatment in plot_df['Treatment'].unique():
                    for demographic in plot_df[plot_df['Treatment'] == treatment]['Demographic'].unique():
                        group_key = f"{treatment}_{demographic}"
                        group_responses = plot_df[(plot_df['Treatment'] == treatment) &
                                                 (plot_df['Demographic'] == demographic)]['Response']

                        numeric_responses = group_responses.apply(lambda x: likert_to_numeric(x, scale_type))
                        numeric_responses = numeric_responses.dropna()

                        if len(numeric_responses) > 0:
                            n = len(numeric_responses)
                            mean_val = numeric_responses.mean()

                            bootstrap_samples = np.random.choice(numeric_responses, size=(n_bootstrap, n), replace=True)
                            bootstrap_means = np.mean(bootstrap_samples, axis=1)

                            ci_lower, ci_upper = np.percentile(bootstrap_means, [2.5, 97.5])

                            point_data_dict[group_key] = {
                                'Treatment': treatment,
                                'Demographic': demographic,
                                'Mean': mean_val,
                                'CI_lower': mean_val - ci_lower,
                                'CI_upper': ci_upper - mean_val,
                                'Count': n,
                                'Raw_CI_lower': ci_lower,
                                'Raw_CI_upper': ci_upper,
                                'bootstrap_means': bootstrap_means,
                                'raw_numeric': numeric_responses.values  # Store for Cohen's d
                            }
            else:
                # Original code for treatment-only grouping
                for treatment in plot_df['Treatment'].unique():
                    treatment_responses = plot_df[plot_df['Treatment'] == treatment]['Response']
                    numeric_responses = treatment_responses.apply(lambda x: likert_to_numeric(x, scale_type))
                    numeric_responses = numeric_responses.dropna()
                    
                    if len(numeric_responses) > 0:
                        n = len(numeric_responses)
                        mean_val = numeric_responses.mean()
                        
                        bootstrap_samples = np.random.choice(numeric_responses, size=(n_bootstrap, n), replace=True)
                        bootstrap_means = np.mean(bootstrap_samples, axis=1)
                        
                        ci_lower, ci_upper = np.percentile(bootstrap_means, [2.5, 97.5])

                        point_data_dict[treatment] = {
                            'Treatment': treatment,
                            'Demographic': 'All',
                            'Mean': mean_val,
                            'CI_lower': mean_val - ci_lower,
                            'CI_upper': ci_upper - mean_val,
                            'Count': n,
                            'Raw_CI_lower': ci_lower,
                            'Raw_CI_upper': ci_upper,
                            'bootstrap_means': bootstrap_means,
                            'raw_numeric': numeric_responses.values  # Store for Cohen's d
                        }

            # Compare each group to the control group "baseline_5" if available
            if selected_demographic != 'None':
                # For demographic grouping, compare within each demographic group
                for demographic in plot_df['Demographic'].unique():
                    baseline_key = f"baseline_5_{demographic}"
                    if baseline_key in point_data_dict:
                        baseline_bootstrap = point_data_dict[baseline_key]['bootstrap_means']
                        baseline_raw = point_data_dict[baseline_key].get('raw_numeric', [])
                        for treatment in plot_df['Treatment'].unique():
                            if treatment != 'baseline_5':
                                treatment_key = f"{treatment}_{demographic}"
                                if treatment_key in point_data_dict:
                                    treatment_bootstrap = point_data_dict[treatment_key]['bootstrap_means']
                                    diff = treatment_bootstrap - baseline_bootstrap
                                    p_value = 2 * min(np.mean(diff > 0), np.mean(diff < 0))
                                    point_data_dict[treatment_key]['p_value'] = p_value

                                    # Calculate Cohen's d with CI
                                    treatment_raw = point_data_dict[treatment_key].get('raw_numeric', [])
                                    if len(treatment_raw) > 0 and len(baseline_raw) > 0:
                                        effect = calculate_cohens_d(list(treatment_raw), list(baseline_raw))
                                        point_data_dict[treatment_key]['cohens_d'] = effect['d']
                                        point_data_dict[treatment_key]['d_ci_lower'] = effect['ci_lower']
                                        point_data_dict[treatment_key]['d_ci_upper'] = effect['ci_upper']
                                        point_data_dict[treatment_key]['effect_interpretation'] = effect['interpretation']
            else:
                # Original code for treatment-only comparison
                if 'baseline_5' in point_data_dict:
                    baseline_bootstrap = point_data_dict['baseline_5']['bootstrap_means']
                    baseline_raw = point_data_dict['baseline_5'].get('raw_numeric', [])
                    for treatment, data in point_data_dict.items():
                        if treatment != 'baseline_5':
                            treatment_bootstrap = data['bootstrap_means']
                            diff = treatment_bootstrap - baseline_bootstrap
                            p_value = 2 * min(np.mean(diff > 0), np.mean(diff < 0))
                            data['p_value'] = p_value

                            # Calculate Cohen's d with CI
                            treatment_raw = data.get('raw_numeric', [])
                            if len(treatment_raw) > 0 and len(baseline_raw) > 0:
                                effect = calculate_cohens_d(list(treatment_raw), list(baseline_raw))
                                data['cohens_d'] = effect['d']
                                data['d_ci_lower'] = effect['ci_lower']
                                data['d_ci_upper'] = effect['ci_upper']
                                data['effect_interpretation'] = effect['interpretation']
                        else:
                            data['p_value'] = None
                            data['cohens_d'] = 0.0
                            data['d_ci_lower'] = 0.0
                            data['d_ci_upper'] = 0.0
                            data['effect_interpretation'] = 'reference'

            point_df = pd.DataFrame(list(point_data_dict.values()))

            # Create the plot
            fig = go.Figure()

            # Reorder treatments so that 'baseline_5' is always first
            treatment_order = list(point_df['Treatment'].unique())
            if 'baseline_5' in treatment_order:
                treatment_order.remove('baseline_5')
                treatment_order = ['baseline_5'] + sorted(treatment_order)
            
            # Create color map for demographics if needed
            if selected_demographic != 'None':
                demographic_values = sorted(point_df['Demographic'].unique())
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
                    demo_df = point_df[point_df['Demographic'] == demographic]
                    
                    # Sort by treatment order
                    demo_df = demo_df.sort_values(by='Treatment', key=lambda x: x.map({t: i for i, t in enumerate(treatment_order)}))
                    
                    # Create a list of dictionaries with treatment and position
                    positions = []
                    for i, treatment in enumerate(demo_df['Treatment']):
                        # Store the original treatment name and the position with offset
                        positions.append({
                            'original': treatment,
                            'position': i + offset_map[demographic]  # Use numeric position + offset
                        })
                    
                    fig.add_trace(go.Scatter(
                        x=[p['position'] for p in positions],  # Use numeric positions with offset
                        y=demo_df['Mean'],
                        error_y=dict(
                            type='data',
                            symmetric=False,
                            array=demo_df['CI_upper'],
                            arrayminus=demo_df['CI_lower'],
                            visible=True
                        ),
                        mode='markers',
                        marker=dict(size=10, color=demographic_color_map[demographic]),
                        name=demographic,
                        text=demo_df['Treatment'],  # Store original treatment names for hover
                        hovertemplate='Treatment: %{text}<br>Mean: %{y:.2f}<extra></extra>'
                    ))
                    
                    # Add annotations for each point
                    for i, (pos, row) in enumerate(zip(positions, demo_df.iterrows())):
                        mean_val = row[1]['Mean']
                        ci_upper = row[1]['CI_upper']
                        ci_lower = row[1]['CI_lower']
                        annotation_text = f'n={row[1]["Count"]}<br>μ={mean_val:.2f} [{mean_val - ci_lower:.2f}, {mean_val + ci_upper:.2f}]'

                        if row[1]["Treatment"] != 'baseline_5' and row[1].get("p_value") is not None:
                            p_val = row[1]["p_value"]
                            stars = get_stars(p_val)
                            annotation_text += f'<br>p={p_val:.3f} {stars}'
                            # Add Cohen's d with CI if available
                            d_val = row[1].get("cohens_d")
                            d_ci_lower = row[1].get("d_ci_lower")
                            d_ci_upper = row[1].get("d_ci_upper")
                            effect_interp = row[1].get("effect_interpretation", "")
                            if d_val is not None:
                                if d_ci_lower is not None and d_ci_upper is not None:
                                    annotation_text += f'<br>d={d_val:.2f} [{d_ci_lower:.2f}, {d_ci_upper:.2f}] ({effect_interp})'
                                else:
                                    annotation_text += f'<br>d={d_val:.2f} ({effect_interp})'
                        fig.add_annotation(
                            x=pos['position'],
                            y=row[1]['Mean'],
                            text=annotation_text,
                            yshift=70,
                            showarrow=False,
                            font=dict(size=8)
                        )
                
                # Update x-axis to show treatment names at the right positions
                fig.update_layout(
                    xaxis=dict(
                        tickmode='array',
                        tickvals=list(range(len(treatment_order))),
                        ticktext=treatment_order
                    )
                )
            else:
                # Original code for treatment-only plot
                # Sort by treatment order
                point_df = point_df.sort_values(by='Treatment', key=lambda x: x.map({t: i for i, t in enumerate(treatment_order)}))
                
                fig.add_trace(go.Scatter(
                    x=point_df['Treatment'],
                    y=point_df['Mean'],
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array=point_df['CI_upper'],
                        arrayminus=point_df['CI_lower'],
                        visible=True
                    ),
                    mode='markers',
                    marker=dict(size=10),
                    name='Mean Response'
                ))
                
                # Add annotations including sample size, CI, and significance stars
                for i, row in point_df.iterrows():
                    mean_val = row['Mean']
                    ci_upper = row['CI_upper']
                    ci_lower = row['CI_lower']
                    annotation_text = f'n={row["Count"]}<br>μ={mean_val:.2f} [{mean_val - ci_lower:.2f}, {mean_val + ci_upper:.2f}]'

                    if row["Treatment"] != 'baseline_5' and row.get("p_value") is not None:
                        p_val = row["p_value"]
                        stars = get_stars(p_val)
                        annotation_text += f'<br>p={p_val:.3f} {stars}'
                        # Add Cohen's d with CI if available
                        d_val = row.get("cohens_d")
                        d_ci_lower = row.get("d_ci_lower")
                        d_ci_upper = row.get("d_ci_upper")
                        effect_interp = row.get("effect_interpretation", "")
                        if d_val is not None:
                            if d_ci_lower is not None and d_ci_upper is not None:
                                annotation_text += f'<br>d={d_val:.2f} [{d_ci_lower:.2f}, {d_ci_upper:.2f}] ({effect_interp})'
                            else:
                                annotation_text += f'<br>d={d_val:.2f} ({effect_interp})'
                    fig.add_annotation(
                        x=row['Treatment'],
                        y=row['Mean'],
                        text=annotation_text,
                        yshift=70,
                        showarrow=False,
                        font=dict(size=8)
                    )
            
            # Set y-axis tick labels according to the detected Likert scale
            if scale_type == 'new':
                tick_text = ['Very Poor', 'Poor', 'Fair', 'Good', 'Excellent']
            else:
                tick_text = ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree']
            
            # Update layout with appropriate title based on demographic selection
            title_text = question_text
            if selected_demographic != 'None':
                title_text = f"{question_text} (by {selected_demographic})"
                
            fig.update_layout(
                title=title_text,
                width=2000,  # Increased from 800 to 1000
                height=500,
                xaxis_title="Treatment",
                yaxis_title="Average Response (1-5)",
                xaxis=dict(
                    categoryorder='array',
                    categoryarray=treatment_order
                ),
                yaxis=dict(
                    ticktext=tick_text,
                    tickvals=[1, 2, 3, 4, 5],
                    range=[0.5, 5.5]
                ),
                showlegend=(selected_demographic != 'None')  # Only show legend when using demographics
            )
            
        else:
            # For non-Likert questions, keep the existing bar chart code
            total_responses = plot_df.groupby('Treatment').size()
            plot_df = plot_df.groupby(['Treatment', 'Response']).size().reset_index(name='Count')
            plot_df['Total'] = plot_df['Treatment'].map(total_responses)
            plot_df['Percentage'] = (plot_df['Count'] / plot_df['Total']) * 100
            
            if is_likert_question(plot_df):
                plot_df['Response'] = pd.Categorical(plot_df['Response'],
                                                     categories=likert_order,
                                                     ordered=True)
                response_order = likert_order
            else:
                unique_responses = plot_df['Response'].unique()
                response_order = sorted([r for r in unique_responses if r != 'EMPTY']) + ['EMPTY']
            
            fig = px.bar(plot_df,
                         x='Response',
                         y='Percentage',
                         color='Treatment',
                         barmode='group',
                         title=question_text,
                         color_discrete_map=color_map,
                         category_orders={'Response': response_order},
                         custom_data=['Count', 'Total'])
            
            fig.update_traces(
                hovertemplate="<br>".join([
                    "<b>%{x}</b>",
                    "Treatment: %{fullData.name}",
                    "Percentage: %{y:.1f}%",
                    "Count: %{customdata[0]}",
                    "Total Responses: %{customdata[1]}",
                    "<extra></extra>"
                ])
            )
            
            fig.update_layout(
                width=800,
                height=500,
                xaxis_title="Response",
                yaxis_title="Percentage of Responses (%)",
                legend_title="Treatment",
                bargap=0.2,
                bargroupgap=0.1,
                xaxis_tickangle=-45
            )
            
        st.plotly_chart(fig)

# Exit Survey Dashboard
def display_exit_survey():
    st.header("Exit Survey Results")
    st.markdown("This section presents the results from the exit survey completed by participants.")

    # Calculate completion rate
    total_participants = len(df)
    completed_exit_survey = df['exitStepDone'].fillna(False).astype(bool).sum()
    completion_rate = completed_exit_survey / total_participants if total_participants > 0 else 0
    st.metric("Exit Survey Completion Rate", f"{completion_rate:.2%}")

    # Parse exitSurvey JSON
    # df['exitSurvey'] = df['exitSurvey'].apply(lambda x: json.loads(x) if isinstance(x, str) and x else {})

    # Overall Experience Section
    experience_questions = {
        'overallExperience': 'Rate your overall experience with the platform.',
        'participation': 'I participated in the discussions more than I usually do on social media.',
        'engagement': 'It was easy for me to engage in the conversations.',
        'informative': 'I found the comments from other participants to be informative and high-quality.',
        'trust': 'I trust the information provided by other participants.',
        'agreement': 'I tended to agree with the other participants.',
        'time': 'I feel like I had enough time for each conversation.',
        'barriers': 'Did you face any barriers when posting content?',
    }
    plot_exit_survey_category(df, 'overallExperience', experience_questions)

    # AI Evaluation Section
    st.subheader("AI Evaluation")
    ai_eval_questions = {
        'easyParticipation': 'The AI made it easier for me to participate in the discussion, compared to my usual experience on social media.',
        'naturalDiscussion': 'The AI felt natural in the context of the discussions.',
        'contentQuality': 'The content I created using the AI was high-quality.',
        'usefulness': 'I can imagine situations in which I would use this AI if it was available on social media.',
    }
    plot_exit_survey_category(df, 'aiEvaluation', ai_eval_questions)

    # Social Media AI Section
    st.subheader("Social Media AI Perceptions")
    social_media_questions = {
        'aiComfortability': 'I feel comfortable with AI being used on social media platforms.',
        'aiParticipation': 'AI suggestions can make it more likely for me to participate in online discussions.',
        'aiLessToxic': 'AI can make online discussions more positive and less toxic.',
        'aiLessPolarizing': 'AI can make discussions less polarizing.',
        'aiReducesMisinformation': 'AI can help reduce misinformation on social media.',
        'aiAccuracy': 'AI-generated is accurate and reliable.',
        'aiRegulation': 'AI should be regulated to prevent misuse and ensure ethical use.',
    }
    plot_exit_survey_category(df, 'socialMediaAI', social_media_questions)


def display_barriers_contingency():
    """
    Create a contingency table comparing barriers in initial and exit surveys.
    This shows how many users faced barriers before and after for each treatment group.
    """
    st.subheader("Barriers to Posting: Initial vs Exit Survey")
    st.markdown("This table shows how participants' experiences with barriers changed between the initial survey (pre-experiment) and exit survey (post-experiment).")
    
    # Define barrier options
    barrier_options = [
        "None",
        "Lack of ideas",
        "Time constraints",
        "Fear of negative feedback",
        "Technical difficulties",
        "Privacy concerns",
        "Other (please specify)"
    ]
    
    # Initialize data structure to store results by treatment
    barrier_data = {barrier: {} for barrier in barrier_options}
    
    # Process each user's data
    for _, row in df.iterrows():
        treatment = row['treatmentName']
        
        # Skip if we don't have both initial and exit survey data
        if not isinstance(row['initialSurvey'], dict) or not isinstance(row['exitSurvey'], dict):
            continue
        if 'socialMediaInfo' not in row['initialSurvey'] or 'overallExperience' not in row['exitSurvey']:
            continue
        
        # Get barriers from initial survey (socialMediaInfo.barriersPosting)
        initial_barriers = []
        if 'socialMediaInfo' in row['initialSurvey']:
            barriers_posting = row['initialSurvey']['socialMediaInfo'].get('barriersPosting', [])
            if barriers_posting:
                initial_barriers = barriers_posting if isinstance(barriers_posting, list) else [barriers_posting]
                # Replace empty list or ['None'] with ['None']
                if not initial_barriers or (len(initial_barriers) == 1 and initial_barriers[0] == 'None'):
                    initial_barriers = ['None']
        else:
            initial_barriers = ['None']  # Default if missing
        
        # Get barriers from exit survey (overallExperience.barriers)
        exit_barriers = []
        if 'overallExperience' in row['exitSurvey']:
            barriers_exit = row['exitSurvey']['overallExperience'].get('barriers', [])
            if barriers_exit:
                exit_barriers = barriers_exit if isinstance(barriers_exit, list) else [barriers_exit]
                # Replace empty list or ['None'] with ['None']
                if not exit_barriers or (len(exit_barriers) == 1 and exit_barriers[0] == 'None'):
                    exit_barriers = ['None']
        else:
            exit_barriers = ['None']  # Default if missing
        
        # Analyze each barrier type
        for barrier in barrier_options:
            had_initial = barrier in initial_barriers
            had_exit = barrier in exit_barriers
            
            # Determine change category
            if had_initial and had_exit:
                category = "Maintained"
            elif had_initial and not had_exit:
                category = "Removed"
            elif not had_initial and had_exit:
                category = "Gained"
            else:
                category = "Never Had"
            
            # Add to the data structure
            if treatment not in barrier_data[barrier]:
                barrier_data[barrier][treatment] = {
                    "Maintained": 0,
                    "Removed": 0,
                    "Gained": 0,
                    "Never Had": 0,
                    "Total": 0
                }
            
            barrier_data[barrier][treatment][category] += 1
            barrier_data[barrier][treatment]["Total"] += 1
    
    # Create tabs for overall summary and specific barrier types
    barrier_tabs = st.tabs(["Overall Summary"] + barrier_options)
    
    # Overall Summary Tab
    with barrier_tabs[0]:
        st.subheader("Barriers Summary (All Types)")
        
        # Create summary data across all barrier types (except "None")
        summary_data = {}
        
        for treatment in set(t for barrier in barrier_data.values() for t in barrier.keys()):
            summary_data[treatment] = {
                "Barriers Maintained": 0,
                "Barriers Removed": 0,
                "Barriers Gained": 0,
                "No Barriers (Both Surveys)": 0,
                "Total Users": 0
            }
        
        # First pass to get total users
        for treatment in summary_data:
            if "None" in barrier_data and treatment in barrier_data["None"]:
                none_data = barrier_data["None"][treatment]
                summary_data[treatment]["Total Users"] = none_data["Total"]
                
                # Count users with no barriers in both surveys
                summary_data[treatment]["No Barriers (Both Surveys)"] = none_data["Maintained"]
                
                # Adjust for "None" barrier specifically
                if treatment in barrier_data["None"]:
                    # Users who gained "None" = removed actual barriers
                    summary_data[treatment]["Barriers Removed"] += none_data["Gained"]
                    # Users who removed "None" = gained actual barriers
                    summary_data[treatment]["Barriers Gained"] += none_data["Removed"]
            else:
                # If treatment not in "None" data, get total from any other barrier
                for barrier in barrier_options:
                    if barrier != "None" and treatment in barrier_data[barrier]:
                        summary_data[treatment]["Total Users"] = barrier_data[barrier][treatment]["Total"]
                        break
        
        # Second pass to count specific barrier changes (excluding "None")
        for barrier in barrier_options:
            if barrier == "None":
                continue
                
            for treatment in summary_data.keys():
                if treatment in barrier_data[barrier]:
                    # Count maintained barriers
                    summary_data[treatment]["Barriers Maintained"] += barrier_data[barrier][treatment]["Maintained"]
                    # Count removed barriers (not the same as gained "None")
                    summary_data[treatment]["Barriers Removed"] += barrier_data[barrier][treatment]["Removed"]
                    # Count gained barriers (not the same as removed "None")
                    summary_data[treatment]["Barriers Gained"] += barrier_data[barrier][treatment]["Gained"]
        
        # Create table for display
        summary_table = []
        for treatment, counts in summary_data.items():
            total = counts["Total Users"]
            if total > 0:
                summary_table.append({
                    "Treatment": treatment,
                    "Barriers Maintained": f"{counts['Barriers Maintained']} ({counts['Barriers Maintained']/total:.1%})",
                    "Barriers Removed": f"{counts['Barriers Removed']} ({counts['Barriers Removed']/total:.1%})",
                    "Barriers Gained": f"{counts['Barriers Gained']} ({counts['Barriers Gained']/total:.1%})",
                    "No Barriers (Both)": f"{counts['No Barriers (Both Surveys)']} ({counts['No Barriers (Both Surveys)']/total:.1%})",
                    "Total Users": total
                })
        
        if summary_table:
            # Convert to DataFrame and sort treatments (with baseline_5 first)
            df_summary = pd.DataFrame(summary_table)
            if "baseline_5" in df_summary["Treatment"].values:
                baseline_row = df_summary[df_summary["Treatment"] == "baseline_5"]
                other_rows = df_summary[df_summary["Treatment"] != "baseline_5"].sort_values("Treatment")
                df_summary = pd.concat([baseline_row, other_rows]).reset_index(drop=True)
            else:
                df_summary = df_summary.sort_values("Treatment").reset_index(drop=True)
            
            # Display the table
            st.table(df_summary)
            
            # Create visualization
            st.subheader("Summary Visualization")
            
            # Prepare data for plot
            summary_plot_data = []
            for treatment, counts in summary_data.items():
                total = counts["Total Users"]
                if total > 0:
                    summary_plot_data.extend([
                        {"Treatment": treatment, "Category": "Barriers Maintained", 
                         "Count": counts["Barriers Maintained"], "Percentage": (counts["Barriers Maintained"]/total)*100},
                        {"Treatment": treatment, "Category": "Barriers Removed", 
                         "Count": counts["Barriers Removed"], "Percentage": (counts["Barriers Removed"]/total)*100},
                        {"Treatment": treatment, "Category": "Barriers Gained", 
                         "Count": counts["Barriers Gained"], "Percentage": (counts["Barriers Gained"]/total)*100},
                        {"Treatment": treatment, "Category": "No Barriers (Both)", 
                         "Count": counts["No Barriers (Both Surveys)"], "Percentage": (counts["No Barriers (Both Surveys)"]/total)*100}
                    ])
            
            df_summary_plot = pd.DataFrame(summary_plot_data)
            
            # Define order for categories and treatments
            category_order = ["Barriers Maintained", "Barriers Removed", "Barriers Gained", "No Barriers (Both)"]
            
            treatment_order = list(summary_data.keys())
            if "baseline_5" in treatment_order:
                treatment_order.remove("baseline_5")
                treatment_order = ["baseline_5"] + sorted(treatment_order)
            else:
                treatment_order = sorted(treatment_order)
            
            # Create the plot
            fig = px.bar(
                df_summary_plot, 
                x="Treatment", 
                y="Percentage", 
                color="Category",
                barmode="group",
                title="Barriers to Posting: Summary Across All Barrier Types",
                category_orders={
                    "Category": category_order,
                    "Treatment": treatment_order
                },
                color_discrete_sequence=px.colors.qualitative.Set2,
                hover_data=["Count"]
            )
            
            fig.update_layout(
                xaxis_title="Treatment Group",
                yaxis_title="Percentage of Users",
                legend_title="Barrier Status Change",
                height=500
            )
            
            st.plotly_chart(fig)
    
    # Individual Barrier Tabs
    for i, barrier in enumerate(barrier_options):
        with barrier_tabs[i+1]:
            st.subheader(f"Barrier Analysis: {barrier}")
            
            # Create table for this specific barrier
            barrier_table = []
            for treatment, counts in barrier_data[barrier].items():
                total = counts["Total"]
                if total > 0:
                    barrier_table.append({
                        "Treatment": treatment,
                        "Maintained": f"{counts['Maintained']} ({counts['Maintained']/total:.1%})",
                        "Removed": f"{counts['Removed']} ({counts['Removed']/total:.1%})",
                        "Gained": f"{counts['Gained']} ({counts['Gained']/total:.1%})",
                        "Never Had": f"{counts['Never Had']} ({counts['Never Had']/total:.1%})",
                        "Total Users": total
                    })
            
            if barrier_table:
                # Convert to DataFrame and sort treatments (with baseline_5 first)
                df_barrier = pd.DataFrame(barrier_table)
                if "baseline_5" in df_barrier["Treatment"].values:
                    baseline_row = df_barrier[df_barrier["Treatment"] == "baseline_5"]
                    other_rows = df_barrier[df_barrier["Treatment"] != "baseline_5"].sort_values("Treatment")
                    df_barrier = pd.concat([baseline_row, other_rows]).reset_index(drop=True)
                else:
                    df_barrier = df_barrier.sort_values("Treatment").reset_index(drop=True)
                
                # Display the table
                st.table(df_barrier)
                
                # Create visualization
                st.subheader(f"Visualization: {barrier}")
                
                # Prepare data for plot
                barrier_plot_data = []
                for treatment, counts in barrier_data[barrier].items():
                    total = counts["Total"]
                    if total > 0:
                        barrier_plot_data.extend([
                            {"Treatment": treatment, "Category": "Maintained", 
                             "Count": counts["Maintained"], "Percentage": (counts["Maintained"]/total)*100},
                            {"Treatment": treatment, "Category": "Removed", 
                             "Count": counts["Removed"], "Percentage": (counts["Removed"]/total)*100},
                            {"Treatment": treatment, "Category": "Gained", 
                             "Count": counts["Gained"], "Percentage": (counts["Gained"]/total)*100},
                            {"Treatment": treatment, "Category": "Never Had", 
                             "Count": counts["Never Had"], "Percentage": (counts["Never Had"]/total)*100}
                        ])
                
                df_barrier_plot = pd.DataFrame(barrier_plot_data)
                
                # Define order for categories and treatments
                category_order = ["Maintained", "Removed", "Gained", "Never Had"]
                
                treatment_order = list(barrier_data[barrier].keys())
                if "baseline_5" in treatment_order:
                    treatment_order.remove("baseline_5")
                    treatment_order = ["baseline_5"] + sorted(treatment_order)
                else:
                    treatment_order = sorted(treatment_order)
                
                # Create the plot
                fig = px.bar(
                    df_barrier_plot, 
                    x="Treatment", 
                    y="Percentage", 
                    color="Category",
                    barmode="group",
                    title=f"Barrier Analysis: {barrier}",
                    category_orders={
                        "Category": category_order,
                        "Treatment": treatment_order
                    },
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    hover_data=["Count"]
                )
                
                fig.update_layout(
                    xaxis_title="Treatment Group",
                    yaxis_title="Percentage of Users",
                    legend_title="Status Change",
                    height=500
                )
                
                st.plotly_chart(fig)
                
                # Add interpretation for this barrier
                if barrier == "None":
                    st.markdown("""
                    ### Interpretation:
                    - **Maintained**: Users who reported no barriers in both surveys
                    - **Removed**: Users who reported no barriers initially but reported barriers in the exit survey
                    - **Gained**: Users who reported barriers initially but reported no barriers in the exit survey
                    - **Never Had**: Users who reported barriers in both surveys
                    
                    A higher percentage of "Gained" indicates that users who initially faced barriers no longer 
                    experience them after the treatment.
                    """)
                else:
                    st.markdown(f"""
                    ### Interpretation for "{barrier}":
                    - **Maintained**: Users who reported this barrier in both surveys
                    - **Removed**: Users who reported this barrier initially but not in the exit survey
                    - **Gained**: Users who didn't report this barrier initially but reported it in the exit survey
                    - **Never Had**: Users who never reported this barrier in either survey
                    
                    A higher percentage of "Removed" indicates that the treatment may have been effective at
                    addressing this specific barrier.
                    """)
            else:
                st.warning(f"Insufficient data for barrier type: {barrier}")
    
    # Add overall interpretation
    st.markdown("""
    ### Key Metrics to Consider:
    - **Barriers Removed**: Higher values indicate the treatment helped users overcome existing barriers
    - **Barriers Gained**: Higher values indicate the treatment may have introduced new barriers
    - **Comparison with Control**: Compare treatment groups with the baseline_5 control group to assess intervention effects
    """)

def plot_exit_survey_category(df, category, questions_dict):
    """Helper function to plot exit survey questions for a specific category"""
    # Create color map for treatments
    color_sequence = px.colors.qualitative.D3[:len(df['treatmentName'].unique())]
    color_map = dict(zip(sorted(df['treatmentName'].unique()), color_sequence))
    
    # Add demographic grouping option
    demographic_options = ['None', 'age', 'gender', 'education', 'occupation', 'partyAffiliation']
    selected_demographic = st.selectbox(
        "Group by demographic (optional):", 
        demographic_options,
        key=f"demographic_select_exit_{category}"
    )
    
    likert_data = {}  # Dictionary to store all Likert questions data
    
    # Plot each question
    for question_key, question_text in questions_dict.items():
        data_list = []
        
        # First collect all responses
        for treatment in df['treatmentName'].unique():
            treatment_df = df[df['treatmentName'] == treatment]
            
            responses = treatment_df['exitSurvey'].apply(
                lambda x: x.get(category, {}).get(question_key, 'EMPTY') 
                if isinstance(x, dict) and category in x else 'EMPTY'
            ).replace('', 'EMPTY').fillna('EMPTY')
            
            if responses.apply(lambda x: isinstance(x, list)).any():
                flattened_data = [item if item else 'EMPTY' 
                                for sublist in responses 
                                if isinstance(sublist, list) 
                                for item in (sublist if sublist else ['EMPTY'])]
                responses = pd.Series(flattened_data)
            
            # Add demographic information if selected
            if selected_demographic != 'None':
                for i, response in enumerate(responses):
                    if i < len(treatment_df):
                        demographic_value = 'Unknown'
                        if selected_demographic in ['age', 'gender', 'education', 'occupation', 'partyAffiliation']:
                            # Extract demographic from initialSurvey
                            demo_value = treatment_df.iloc[i]['initialSurvey'].get('demographicInfo', {}).get(selected_demographic, '')
                            
                            # Special handling for party affiliation
                            if selected_demographic == 'partyAffiliation' and demo_value:
                                if 'Republican' in demo_value:
                                    demo_value = 'Republican'
                                elif 'Democrat' in demo_value:
                                    demo_value = 'Democrat'
                            
                            if demo_value and demo_value != 'Prefer not to answer':
                                demographic_value = demo_value
                        
                        data_list.append({
                            'Treatment': treatment,
                            'Response': response,
                            'Demographic': demographic_value
                        })
                    else:
                        data_list.append({
                            'Treatment': treatment,
                            'Response': response,
                            'Demographic': 'Unknown'
                        })
            else:
                # Original code without demographic grouping
                for response in responses:
                    data_list.append({
                        'Treatment': treatment,
                        'Response': response,
                        'Demographic': 'All'  # Default when no demographic is selected
                    })
        
        # Create initial DataFrame
        plot_df = pd.DataFrame(data_list)
        
        # Filter out 'Unknown' and 'Prefer not to answer' demographics
        if selected_demographic != 'None':
            plot_df = plot_df[~plot_df['Demographic'].isin(['Unknown', 'Prefer not to answer', ''])]
        
        # Now check if it's a Likert question
        if is_likert_question(plot_df):  # For Likert questions, create point plot
            scale_type = get_likert_type(plot_df)
            point_data_dict = {}

            # Set random seed for reproducibility and define bootstrap iterations
            np.random.seed(42)
            n_bootstrap = 10000

            # Compute mean, bootstrap confidence intervals, and hold bootstrap distributions
            if selected_demographic != 'None':
                # Group by both treatment and demographic
                for treatment in plot_df['Treatment'].unique():
                    for demographic in plot_df[plot_df['Treatment'] == treatment]['Demographic'].unique():
                        group_key = f"{treatment}_{demographic}"
                        group_responses = plot_df[(plot_df['Treatment'] == treatment) &
                                                 (plot_df['Demographic'] == demographic)]['Response']

                        numeric_responses = group_responses.apply(lambda x: likert_to_numeric(x, scale_type))
                        numeric_responses = numeric_responses.dropna()

                        if len(numeric_responses) > 0:
                            n = len(numeric_responses)
                            mean_val = numeric_responses.mean()

                            bootstrap_samples = np.random.choice(numeric_responses, size=(n_bootstrap, n), replace=True)
                            bootstrap_means = np.mean(bootstrap_samples, axis=1)

                            ci_lower, ci_upper = np.percentile(bootstrap_means, [2.5, 97.5])

                            point_data_dict[group_key] = {
                                'Treatment': treatment,
                                'Demographic': demographic,
                                'Mean': mean_val,
                                'CI_lower': mean_val - ci_lower,
                                'CI_upper': ci_upper - mean_val,
                                'Count': n,
                                'Raw_CI_lower': ci_lower,
                                'Raw_CI_upper': ci_upper,
                                'bootstrap_means': bootstrap_means,
                                'raw_numeric': numeric_responses.values  # Store for Cohen's d
                            }
            else:
                # Original code for treatment-only grouping
                for treatment in plot_df['Treatment'].unique():
                    treatment_responses = plot_df[plot_df['Treatment'] == treatment]['Response']
                    numeric_responses = treatment_responses.apply(lambda x: likert_to_numeric(x, scale_type))
                    numeric_responses = numeric_responses.dropna()

                    if len(numeric_responses) > 0:
                        n = len(numeric_responses)
                        mean_val = numeric_responses.mean()

                        bootstrap_samples = np.random.choice(numeric_responses, size=(n_bootstrap, n), replace=True)
                        bootstrap_means = np.mean(bootstrap_samples, axis=1)

                        ci_lower, ci_upper = np.percentile(bootstrap_means, [2.5, 97.5])

                        point_data_dict[treatment] = {
                            'Treatment': treatment,
                            'Demographic': 'All',
                            'Mean': mean_val,
                            'CI_lower': mean_val - ci_lower,
                            'CI_upper': ci_upper - mean_val,
                            'Count': n,
                            'Raw_CI_lower': ci_lower,
                            'Raw_CI_upper': ci_upper,
                            'bootstrap_means': bootstrap_means,
                            'raw_numeric': numeric_responses.values  # Store for Cohen's d
                        }

            # Compare each group to the control group "baseline_5" if available
            if selected_demographic != 'None':
                # For demographic grouping, compare within each demographic group
                for demographic in plot_df['Demographic'].unique():
                    baseline_key = f"baseline_5_{demographic}"
                    if baseline_key in point_data_dict:
                        baseline_bootstrap = point_data_dict[baseline_key]['bootstrap_means']
                        baseline_raw = point_data_dict[baseline_key].get('raw_numeric', [])
                        for treatment in plot_df['Treatment'].unique():
                            if treatment != 'baseline_5':
                                treatment_key = f"{treatment}_{demographic}"
                                if treatment_key in point_data_dict:
                                    treatment_bootstrap = point_data_dict[treatment_key]['bootstrap_means']
                                    diff = treatment_bootstrap - baseline_bootstrap
                                    p_value = 2 * min(np.mean(diff > 0), np.mean(diff < 0))
                                    point_data_dict[treatment_key]['p_value'] = p_value

                                    # Calculate Cohen's d with CI
                                    treatment_raw = point_data_dict[treatment_key].get('raw_numeric', [])
                                    if len(treatment_raw) > 0 and len(baseline_raw) > 0:
                                        effect = calculate_cohens_d(list(treatment_raw), list(baseline_raw))
                                        point_data_dict[treatment_key]['cohens_d'] = effect['d']
                                        point_data_dict[treatment_key]['d_ci_lower'] = effect['ci_lower']
                                        point_data_dict[treatment_key]['d_ci_upper'] = effect['ci_upper']
                                        point_data_dict[treatment_key]['effect_interpretation'] = effect['interpretation']
            else:
                # Original code for treatment-only comparison
                if 'baseline_5' in point_data_dict:
                    baseline_bootstrap = point_data_dict['baseline_5']['bootstrap_means']
                    baseline_raw = point_data_dict['baseline_5'].get('raw_numeric', [])
                    for treatment, data in point_data_dict.items():
                        if treatment != 'baseline_5':
                            treatment_bootstrap = data['bootstrap_means']
                            diff = treatment_bootstrap - baseline_bootstrap
                            p_value = 2 * min(np.mean(diff > 0), np.mean(diff < 0))
                            data['p_value'] = p_value

                            # Calculate Cohen's d with CI
                            treatment_raw = data.get('raw_numeric', [])
                            if len(treatment_raw) > 0 and len(baseline_raw) > 0:
                                effect = calculate_cohens_d(list(treatment_raw), list(baseline_raw))
                                data['cohens_d'] = effect['d']
                                data['d_ci_lower'] = effect['ci_lower']
                                data['d_ci_upper'] = effect['ci_upper']
                                data['effect_interpretation'] = effect['interpretation']
                        else:
                            data['p_value'] = None
                            data['cohens_d'] = 0.0
                            data['d_ci_lower'] = 0.0
                            data['d_ci_upper'] = 0.0
                            data['effect_interpretation'] = 'reference'

            point_df = pd.DataFrame(list(point_data_dict.values()))

            # Create the plot
            fig = go.Figure()

            # Reorder treatments so that baseline_5 is always displayed first
            treatment_order = list(point_df['Treatment'].unique())
            if 'baseline_5' in treatment_order:
                treatment_order.remove('baseline_5')
                treatment_order = ['baseline_5'] + sorted(treatment_order)
            
            # Create color map for demographics if needed
            if selected_demographic != 'None':
                demographic_values = sorted(point_df['Demographic'].unique())
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
                    demo_df = point_df[point_df['Demographic'] == demographic]
                    
                    # Sort by treatment order
                    demo_df = demo_df.sort_values(by='Treatment', key=lambda x: x.map({t: i for i, t in enumerate(treatment_order)}))
                    
                    # Create a list of dictionaries with treatment and position
                    positions = []
                    for i, treatment in enumerate(demo_df['Treatment']):
                        # Store the original treatment name and the position with offset
                        positions.append({
                            'original': treatment,
                            'position': i + offset_map[demographic]  # Use numeric position + offset
                        })
                    
                    fig.add_trace(go.Scatter(
                        x=[p['position'] for p in positions],  # Use numeric positions with offset
                        y=demo_df['Mean'],
                        error_y=dict(
                            type='data',
                            symmetric=False,
                            array=demo_df['CI_upper'],
                            arrayminus=demo_df['CI_lower'],
                            visible=True
                        ),
                        mode='markers',
                        marker=dict(size=10, color=demographic_color_map[demographic]),
                        name=demographic,
                        text=demo_df['Treatment'],  # Store original treatment names for hover
                        hovertemplate='Treatment: %{text}<br>Mean: %{y:.2f}<extra></extra>'
                    ))
                    
                    # Add annotations for each point
                    for i, (pos, row) in enumerate(zip(positions, demo_df.iterrows())):
                        mean_val = row[1]['Mean']
                        ci_upper = row[1]['CI_upper']
                        ci_lower = row[1]['CI_lower']
                        annotation_text = f'n={row[1]["Count"]}<br>μ={mean_val:.2f} [{mean_val - ci_lower:.2f}, {mean_val + ci_upper:.2f}]'

                        if row[1]["Treatment"] != 'baseline_5' and row[1].get("p_value") is not None:
                            p_val = row[1]["p_value"]
                            stars = get_stars(p_val)
                            annotation_text += f'<br>p={p_val:.3f} {stars}'
                            # Add Cohen's d with CI if available
                            d_val = row[1].get("cohens_d")
                            d_ci_lower = row[1].get("d_ci_lower")
                            d_ci_upper = row[1].get("d_ci_upper")
                            effect_interp = row[1].get("effect_interpretation", "")
                            if d_val is not None:
                                if d_ci_lower is not None and d_ci_upper is not None:
                                    annotation_text += f'<br>d={d_val:.2f} [{d_ci_lower:.2f}, {d_ci_upper:.2f}] ({effect_interp})'
                                else:
                                    annotation_text += f'<br>d={d_val:.2f} ({effect_interp})'
                        fig.add_annotation(
                            x=pos['position'],
                            y=row[1]['Mean'],
                            text=annotation_text,
                            yshift=70,
                            showarrow=False,
                            font=dict(size=8)
                        )
                
                # Update x-axis to show treatment names at the right positions
                fig.update_layout(
                    xaxis=dict(
                        tickmode='array',
                        tickvals=list(range(len(treatment_order))),
                        ticktext=treatment_order
                    )
                )
            else:
                # Original code for treatment-only plot
                # Sort by treatment order
                point_df = point_df.sort_values(by='Treatment', key=lambda x: x.map({t: i for i, t in enumerate(treatment_order)}))
                
                fig.add_trace(go.Scatter(
                    x=point_df['Treatment'],
                    y=point_df['Mean'],
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array=point_df['CI_upper'],
                        arrayminus=point_df['CI_lower'],
                        visible=True
                    ),
                    mode='markers',
                    marker=dict(size=10),
                    name='Mean Response'
                ))
                
                # Add annotations including sample size, CI, and significance stars
                for i, row in point_df.iterrows():
                    mean_val = row['Mean']
                    ci_upper = row['CI_upper']
                    ci_lower = row['CI_lower']
                    annotation_text = f'n={row["Count"]}<br>μ={mean_val:.2f} [{mean_val - ci_lower:.2f}, {mean_val + ci_upper:.2f}]'

                    if row["Treatment"] != 'baseline_5' and row.get("p_value") is not None:
                        p_val = row["p_value"]
                        stars = get_stars(p_val)
                        annotation_text += f'<br>p={p_val:.3f} {stars}'
                        # Add Cohen's d with CI if available
                        d_val = row.get("cohens_d")
                        d_ci_lower = row.get("d_ci_lower")
                        d_ci_upper = row.get("d_ci_upper")
                        effect_interp = row.get("effect_interpretation", "")
                        if d_val is not None:
                            if d_ci_lower is not None and d_ci_upper is not None:
                                annotation_text += f'<br>d={d_val:.2f} [{d_ci_lower:.2f}, {d_ci_upper:.2f}] ({effect_interp})'
                            else:
                                annotation_text += f'<br>d={d_val:.2f} ({effect_interp})'
                    fig.add_annotation(
                        x=row['Treatment'],
                        y=row['Mean'],
                        text=annotation_text,
                        yshift=70,
                        showarrow=False,
                        font=dict(size=8)
                    )
            
            # Set y-axis tick labels based on the scale in use
            if scale_type == 'new':
                tick_text = ['Very Poor', 'Poor', 'Fair', 'Good', 'Excellent']
            else:
                tick_text = ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree']
            
            # Update layout with appropriate title based on demographic selection
            title_text = question_text
            if selected_demographic != 'None':
                title_text = f"{question_text} (by {selected_demographic})"
            
            fig.update_layout(
                title=title_text,
                width=2000,  # Increased from 800 to 1500
                height=500,
                xaxis_title="Treatment",
                yaxis_title="Average Response (1-5)",
                xaxis=dict(
                    categoryorder='array',
                    categoryarray=treatment_order
                ),
                yaxis=dict(
                    ticktext=tick_text,
                    tickvals=[1, 2, 3, 4, 5],
                    range=[0.5, 5.5]
                ),
                showlegend=(selected_demographic != 'None')  # Only show legend when using demographics
            )
            
            # After calculating all statistics, store in likert_data
            question_data = {
                "x_vals": list(range(1, len(point_data_dict) + 1)),
                "y_vals": [data['Mean'] for data in point_data_dict.values()],
                "err": [data['CI_upper'] for data in point_data_dict.values()],
                "significance": [
                    get_stars(data.get('p_value', 1)) if data.get('p_value') is not None else ""
                    for data in point_data_dict.values()
                ],
                "n_values": [data['Count'] for data in point_data_dict.values()],
                "treatment_labels": [data['Treatment'] for data in point_data_dict.values()],
                "question_label": question_text,
                "scale_type": scale_type
            }
            
            likert_data[question_key] = question_data
        
        else:  # Keep existing bar chart code for non-Likert questions
            total_responses = plot_df.groupby('Treatment').size()
            plot_df = plot_df.groupby(['Treatment', 'Response']).size().reset_index(name='Count')
            plot_df['Total'] = plot_df['Treatment'].map(total_responses)
            plot_df['Percentage'] = (plot_df['Count'] / plot_df['Total']) * 100
            
            unique_responses = plot_df['Response'].unique()
            response_order = sorted([r for r in unique_responses if r != 'EMPTY']) + ['EMPTY']
            
            fig = px.bar(plot_df, 
                         x='Response', 
                         y='Percentage',
                         color='Treatment',
                         barmode='group',
                         title=question_text,
                         color_discrete_map=color_map,
                         category_orders={'Response': response_order},
                         custom_data=['Count', 'Total'])
            
            # Update hover template for bar charts
            fig.update_traces(
                hovertemplate="<br>".join([
                    "<b>%{x}</b>",
                    "Treatment: %{fullData.name}",
                    "Percentage: %{y:.1f}%",
                    "Count: %{customdata[0]}",
                    "Total Responses: %{customdata[1]}",
                    "<extra></extra>"
                ])
            )
            
            fig.update_layout(
                width=800,
                height=500,
                xaxis_title="Response",
                yaxis_title="Percentage of Responses (%)",
                legend_title="Treatment",
                bargap=0.2,
                bargroupgap=0.1,
                xaxis_tickangle=-45
            )
        
        st.plotly_chart(fig)

    # Save to JSON file after processing all questions
    # if likert_data:  # Only save if we have Likert data
    #     import json
    #     with open(f'data/likert_{category}.json', 'w') as f:
    #         json.dump(likert_data, f, indent=2)

# Open-ended Feedback
def display_open_ended_feedback():
    st.header("Open-Ended Feedback")
    
    # Create a copy of the dataframe and parse JSON
    feedback_df = df.copy()
    
    # Count AI and Platform feedback entries
    ai_feedback_count = sum(1 for row in feedback_df.iterrows() 
                          if row[1]['exitSurvey'].get('aiEvaluation', {}).get('feedbackAI', '').strip())
    platform_feedback_count = sum(1 for row in feedback_df.iterrows() 
                                if isinstance(row[1]['exitSurvey'], dict) 
                                and row[1]['exitSurvey'].get('openFeedback', {}).get('feedbackPlatform', '').strip())
    
    # Display total counts in columns
    col1, col2 = st.columns(2)
    with col1:
        st.metric("AI Feedback Responses", ai_feedback_count)
    with col2:
        st.metric("Platform Feedback Responses", platform_feedback_count)
    
    # Group feedback counts by treatment
    treatment_ai_counts = {}
    treatment_platform_counts = {}
    
    for _, row in feedback_df.iterrows():
        treatment = row['treatmentName']
        
        # Count AI feedback by treatment
        if row['exitSurvey'].get('aiEvaluation', {}).get('feedbackAI', '').strip():
            treatment_ai_counts[treatment] = treatment_ai_counts.get(treatment, 0) + 1
        
        # Count platform feedback by treatment
        if isinstance(row['exitSurvey'], dict) and row['exitSurvey'].get('openFeedback', {}).get('feedbackPlatform', '').strip():
            treatment_platform_counts[treatment] = treatment_platform_counts.get(treatment, 0) + 1
    
    # Display counts by treatment
    st.subheader("Feedback Counts by Treatment")
    
    # Create dataframes for the counts
    ai_counts_df = pd.DataFrame({
        'Treatment': list(treatment_ai_counts.keys()),
        'AI Feedback Count': list(treatment_ai_counts.values())
    }).sort_values('Treatment')
    
    platform_counts_df = pd.DataFrame({
        'Treatment': list(treatment_platform_counts.keys()),
        'Platform Feedback Count': list(treatment_platform_counts.values())
    }).sort_values('Treatment')
    
    # Merge the dataframes
    counts_df = pd.merge(ai_counts_df, platform_counts_df, on='Treatment', how='outer').fillna(0)
    counts_df = counts_df.astype({'AI Feedback Count': int, 'Platform Feedback Count': int})
    
    # Display as a table
    st.table(counts_df)
    
    st.divider()
    
    # Display AI Feedback
    st.header("AI Feedback")
    
    # Collect AI feedback with treatment info
    ai_feedback_list = []
    for _, row in feedback_df.iterrows():
        ai_feedback = row['exitSurvey'].get('aiEvaluation', {}).get('feedbackAI')
        if ai_feedback and isinstance(ai_feedback, str) and ai_feedback.strip():
            ai_feedback_list.append({
                'participantID': row['participantID'],
                'treatmentName': row['treatmentName'],
                'feedback': ai_feedback
            })
    
    # Sort by treatment name
    ai_feedback_list = sorted(ai_feedback_list, key=lambda x: x['treatmentName'])
    
    # Display sorted feedback
    for item in ai_feedback_list:
        st.markdown(f"**Participant**: {item['participantID']}  \n"
                  f"**Treatment**: {item['treatmentName']}  \n"
                  f"**Feedback**: {item['feedback']}")
        st.markdown("---")
    
    # Display Platform Feedback
    st.header("Platform Feedback")
    
    # Collect platform feedback with treatment info
    platform_feedback_list = []
    for _, row in feedback_df.iterrows():
        if not isinstance(row['exitSurvey'], dict):
            continue
            
        platform_feedback = row['exitSurvey'].get('openFeedback', {}).get('feedbackPlatform')
        if platform_feedback and isinstance(platform_feedback, str) and platform_feedback.strip():
            platform_feedback_list.append({
                'participantID': row['participantID'],
                'treatmentName': row['treatmentName'],
                'feedback': platform_feedback
            })
    
    # Sort by treatment name
    platform_feedback_list = sorted(platform_feedback_list, key=lambda x: x['treatmentName'])
    
    # Display sorted feedback
    for item in platform_feedback_list:
        st.markdown(f"**Participant**: {item['participantID']}  \n"
                  f"**Treatment**: {item['treatmentName']}  \n"
                  f"**Feedback**: {item['feedback']}")
        st.markdown("---")

def bootstrap_mean_ci(data, n_bootstrap=10000):
    """Calculate confidence intervals using bootstrap"""
    n = len(data)
    bootstrap_means = np.zeros(n_bootstrap)
    np.random.seed(42)
    
    for i in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means[i] = np.mean(bootstrap_sample)
        
    ci_lower, ci_upper = np.percentile(bootstrap_means, [2.5, 97.5])
    mean = np.mean(data)
    return mean, ci_lower, ci_upper

def calculate_p_value(treatment_data, baseline_data, n_bootstrap=10000):
    """Calculate p-value using t-test on original data"""
    from scipy import stats
    
    # Perform t-test on the original data
    _, p_value = stats.ttest_ind(treatment_data, baseline_data)
    
    return p_value

def get_stars(p_value):
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    elif p_value < 0.1:
        return "†"
    return "ns"

def add_significance_markers(fig, i, max_y, p_value):
    stars = get_stars(p_value)
    if stars:  # Only add annotation if there are stars
        fig.add_annotation(
            x=i,
            y=max_y + 0.1,
            text=stars,
            showarrow=False,
            font=dict(size=14)
        )
    else:  # Add 'ns' if not significant
        fig.add_annotation(
            x=i,
            y=max_y + 0.1,
            text='ns',
            showarrow=False,
            font=dict(size=14)
        )

def display_user_ratings():
    st.header("User Ratings Analysis")
    

    # Comment Ratings Analysis
    st.subheader("Comment Ratings")
    
    # Add demographic selector before the plot
    demographic_options = ['None', 'age', 'gender', 'education', 'occupation', 'partyAffiliation']
    selected_demographic = st.selectbox(
        "Group by demographic (optional):", 
        demographic_options,
        key="demographic_select_comments"
    )

    # Extract and process comment ratings with demographics
    comment_ratings = []
    for _, row in df.iterrows():
        if isinstance(row['exitSurvey'], dict) and 'userRating' in row['exitSurvey']:
            # Extract demographic info
            demographic_info = row['initialSurvey'].get('demographicInfo', {})
            demographic_value = 'Unknown'
            
            if selected_demographic != 'None':
                if selected_demographic in demographic_info:
                    demo_value = demographic_info[selected_demographic]
                    
                    # Special handling for party affiliation
                    if selected_demographic == 'partyAffiliation' and demo_value:
                        if 'Republican' in demo_value:
                            demo_value = 'Republican'
                        elif 'Democrat' in demo_value:
                            demo_value = 'Democrat'
                    
                    if demo_value and demo_value != 'Prefer not to answer':
                        demographic_value = demo_value

            # Process ratings
            ratings = row['exitSurvey']['userRating'].get('commentRatings', {})
            for comment_id, rating_data in ratings.items():
                if 'valueComment' in rating_data:
                    try:
                        rating = float(rating_data['valueComment'])
                        comment_ratings.append({
                            'treatment': row['treatmentName'],
                            'rating': rating,
                            'demographic': demographic_value
                        })
                    except (ValueError, TypeError):
                        continue

    ratings_df = pd.DataFrame(comment_ratings)

    # Filter out 'Unknown' and 'Prefer not to answer' demographics if a demographic is selected
    if selected_demographic != 'None':
        ratings_df = ratings_df[~ratings_df['demographic'].isin(['Unknown', 'Prefer not to answer', ''])]

    # Calculate statistics and p-values for each treatment-demographic combination
    treatment_stats = []
    baseline_ratings_by_demo = {}

    # First pass to get baseline for each demographic
    if selected_demographic != 'None':
        for demographic in ratings_df['demographic'].unique():
            demo_baseline = ratings_df[
                (ratings_df['treatment'] == 'baseline_5') & 
                (ratings_df['demographic'] == demographic)
            ]['rating'].values
            if len(demo_baseline) > 0:
                baseline_ratings_by_demo[demographic] = {
                    'ratings': demo_baseline,
                    'mean': np.mean(demo_baseline)
                }
    else:
        baseline_data = ratings_df[ratings_df['treatment'] == 'baseline_5']
        if not baseline_data.empty:
            baseline_ratings_by_demo['All'] = {
                'ratings': baseline_data['rating'].values,
                'mean': baseline_data['rating'].mean()
            }

    # Second pass for all treatments and their statistics
    if selected_demographic != 'None':
        for treatment in ratings_df['treatment'].unique():
            for demographic in ratings_df['demographic'].unique():
                treatment_ratings = ratings_df[
                    (ratings_df['treatment'] == treatment) & 
                    (ratings_df['demographic'] == demographic)
                ]['rating'].values
                
                if len(treatment_ratings) > 0:
                    mean, ci_lower, ci_upper = bootstrap_mean_ci(treatment_ratings)
                    
                    stat = {
                        'treatment': treatment,
                        'demographic': demographic,
                        'mean': mean,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'n': len(treatment_ratings),
                        'ratings': treatment_ratings
                    }
                    
                    # Calculate p-value and effect size if not baseline and we have baseline data for this demographic
                    if treatment != 'baseline_5' and demographic in baseline_ratings_by_demo:
                        baseline_ratings = baseline_ratings_by_demo[demographic]['ratings']
                        p_value = calculate_p_value(treatment_ratings, baseline_ratings)
                        stat['p_value'] = p_value
                        stat['stars'] = get_stars(p_value)

                        # Calculate Cohen's d with CI
                        effect_result = calculate_cohens_d(treatment_ratings, baseline_ratings)
                        stat['cohens_d'] = effect_result['d']
                        stat['d_ci_lower'] = effect_result['ci_lower']
                        stat['d_ci_upper'] = effect_result['ci_upper']
                        stat['effect_interpretation'] = effect_result['interpretation']

                    treatment_stats.append(stat)
    else:
        # Original code for no demographic filtering
        for treatment in ratings_df['treatment'].unique():
            treatment_ratings = ratings_df[ratings_df['treatment'] == treatment]['rating'].values
            mean, ci_lower, ci_upper = bootstrap_mean_ci(treatment_ratings)
            
            stat = {
                'treatment': treatment,
                'demographic': 'All',
                'mean': mean,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'n': len(treatment_ratings),
                'ratings': treatment_ratings
            }
            
            if treatment != 'baseline_5' and 'All' in baseline_ratings_by_demo:
                baseline_ratings = baseline_ratings_by_demo['All']['ratings']
                p_value = calculate_p_value(treatment_ratings, baseline_ratings)
                stat['p_value'] = p_value
                stat['stars'] = get_stars(p_value)

                # Calculate Cohen's d with CI
                effect_result = calculate_cohens_d(treatment_ratings, baseline_ratings)
                stat['cohens_d'] = effect_result['d']
                stat['d_ci_lower'] = effect_result['ci_lower']
                stat['d_ci_upper'] = effect_result['ci_upper']
                stat['effect_interpretation'] = effect_result['interpretation']

            treatment_stats.append(stat)

    # Create plot with significance stars
    fig = go.Figure()

    # Create color maps and ensure consistent treatment ordering
    treatment_order = list(ratings_df['treatment'].unique())
    if 'baseline_5' in treatment_order:
        treatment_order.remove('baseline_5')
        treatment_order = ['baseline_5'] + sorted(treatment_order)  # Sort other treatments alphabetically

    if selected_demographic != 'None':
        demographic_values = sorted(ratings_df['demographic'].unique())
        demographic_colors = px.colors.qualitative.Plotly[:len(demographic_values)]
        demographic_color_map = dict(zip(demographic_values, demographic_colors))
        
        # Create offsets for each demographic to prevent overlap
        offset_amount = 0.15
        offsets = np.linspace(-offset_amount * (len(demographic_values)-1)/2, 
                             offset_amount * (len(demographic_values)-1)/2, 
                             len(demographic_values))
        offset_map = dict(zip(demographic_values, offsets))
    else:
        color_sequence = px.colors.qualitative.D3[:len(treatment_order)]  # Use treatment_order length
        color_map = dict(zip(treatment_order, color_sequence))  # Map colors to ordered treatments

    # Find max y value for positioning stars
    max_y = max([stat['mean'] + (stat['ci_upper'] - stat['mean']) for stat in treatment_stats])

    # Add traces for each demographic group
    if selected_demographic != 'None':
        for demographic in demographic_values:
            # Get all stats for this demographic
            demo_stats = [s for s in treatment_stats if s['demographic'] == demographic]
            
            # Create a dictionary mapping treatment to stat for easier lookup
            treatment_to_stat = {s['treatment']: s for s in demo_stats}
            
            # Create positions list for this demographic, ensuring we follow treatment_order
            positions = []
            demo_means = []
            demo_ci_upper = []
            demo_ci_lower = []
            stats_to_plot = []
            
            for i, treatment in enumerate(treatment_order):
                if treatment in treatment_to_stat:
                    stat = treatment_to_stat[treatment]
                    positions.append({
                        'original': treatment,
                        'position': i + offset_map[demographic]
                    })
                    demo_means.append(stat['mean'])
                    demo_ci_upper.append(stat['ci_upper'] - stat['mean'])
                    demo_ci_lower.append(stat['mean'] - stat['ci_lower'])
                    stats_to_plot.append(stat)
            
            # Only add trace if we have data for this demographic
            if positions:
                fig.add_trace(go.Scatter(
                    x=[p['position'] for p in positions],
                    y=demo_means,
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array=demo_ci_upper,
                        arrayminus=demo_ci_lower,
                        visible=True
                    ),
                    mode='markers',
                    marker=dict(size=10, color=demographic_color_map[demographic]),
                    name=demographic,
                    text=[p['original'] for p in positions],
                    hovertemplate=(
                        "<b>Treatment:</b> %{text}<br>"
                        "<b>Demographic:</b> " + demographic + "<br>"
                        "<b>Mean Rating:</b> %{y:.2f}<br>"
                        "<extra></extra>"
                    )
                ))
                
                # Add annotations for each point
                for pos, stat in zip(positions, stats_to_plot):
                    mean_val = stat['mean']
                    ci_lower = stat['ci_lower']
                    ci_upper = stat['ci_upper']
                    annotation_text = f'n={stat["n"]}<br>μ={mean_val:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]'

                    if stat['treatment'] != 'baseline_5' and 'p_value' in stat:
                        p_val = stat['p_value']
                        stars = stat['stars']
                        annotation_text += f'<br>p={p_val:.3f} {stars}'
                        if 'cohens_d' in stat and stat['cohens_d'] is not None:
                            d_val = stat['cohens_d']
                            d_ci_lower = stat.get('d_ci_lower')
                            d_ci_upper = stat.get('d_ci_upper')
                            effect_interp = stat.get('effect_interpretation', '')
                            if d_ci_lower is not None and d_ci_upper is not None:
                                annotation_text += f'<br>d={d_val:.2f} [{d_ci_lower:.2f}, {d_ci_upper:.2f}] ({effect_interp})'
                            else:
                                annotation_text += f'<br>d={d_val:.2f} ({effect_interp})'

                    fig.add_annotation(
                        x=pos['position'],
                        y=stat['mean'],
                        text=annotation_text,
                        yshift=70,
                        showarrow=False,
                        font=dict(size=9)
                    )
    else:
        # Original code for no demographic filtering, but ensure treatment order
        treatment_to_stat = {s['treatment']: s for s in treatment_stats}
        
        for i, treatment in enumerate(treatment_order):
            if treatment in treatment_to_stat:
                stat = treatment_to_stat[treatment]
                
                fig.add_trace(go.Scatter(
                    x=[i],  # Use numeric position for consistent ordering
                    y=[stat['mean']],
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array=[stat['ci_upper'] - stat['mean']],
                        arrayminus=[stat['mean'] - stat['ci_lower']],
                        visible=True
                    ),
                    mode='markers',
                    marker=dict(size=12, color=color_map[treatment]),
                    name=treatment,
                    text=[treatment],  # Store treatment name for hover
                    hovertemplate=(
                        "<b>Treatment:</b> %{text}<br>"
                        "<b>Mean Rating:</b> %{y:.2f}<br>"
                        "<b>Sample Size:</b> " + str(stat['n']) +
                        "<extra></extra>"
                    )
                ))
                
                # Add annotations
                mean_val = stat['mean']
                ci_lower = stat['ci_lower']
                ci_upper = stat['ci_upper']
                annotation_text = f'n={stat["n"]}<br>μ={mean_val:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]'

                if stat['treatment'] != 'baseline_5' and 'p_value' in stat:
                    p_val = stat['p_value']
                    stars = stat['stars']
                    annotation_text += f'<br>p={p_val:.3f} {stars}'
                    if 'cohens_d' in stat and stat['cohens_d'] is not None:
                        d_val = stat['cohens_d']
                        d_ci_lower = stat.get('d_ci_lower')
                        d_ci_upper = stat.get('d_ci_upper')
                        effect_interp = stat.get('effect_interpretation', '')
                        if d_ci_lower is not None and d_ci_upper is not None:
                            annotation_text += f'<br>d={d_val:.2f} [{d_ci_lower:.2f}, {d_ci_upper:.2f}] ({effect_interp})'
                        else:
                            annotation_text += f'<br>d={d_val:.2f} ({effect_interp})'

                fig.add_annotation(
                    x=i,  # Use numeric position
                    y=stat['mean'],
                    text=annotation_text,
                    yshift=70,
                    showarrow=False,
                    font=dict(size=9)
                )

    # Update layout
    title_text = "Comment Ratings by Treatment"
    if selected_demographic != 'None':
        title_text += f" (by {selected_demographic})"

    # Add horizontal grid lines at specific values
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
        yaxis_title="Average Rating",
        xaxis=dict(
            tickmode='array',
            ticktext=treatment_order,  # Use consistent treatment order
            tickvals=list(range(len(treatment_order))),
            tickangle=45,
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            zeroline=True,
            zerolinewidth=2.5,
            zerolinecolor='black',
            gridcolor='lightgray',
            range=[0, max_y + 0.5],
            # Add more prominent grid lines
            dtick=1.0,  # Major grid lines every 1.0 units
            minor_showgrid=True,
            minor_dtick=0.5,  # Minor grid lines every 0.5 units
        ),
        showlegend=(selected_demographic != 'None'),
        legend=dict(
            title=dict(text=selected_demographic if selected_demographic != 'None' else ''),
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=1.15
        ),
        height=600,
        template='simple_white',
        margin=dict(
            t=100,
            b=120,
            r=150,
            l=80
        )
    )

    # Add horizontal reference lines at key values
    for y_val in [1, 2, 3, 4, 5]:
        fig.add_shape(
            type="line",
            x0=-0.5,
            y0=y_val,
            x1=len(treatment_order) - 0.5,
            y1=y_val,
            line=dict(
                color="rgba(150, 150, 150, 0.5)",
                width=1,
                dash="dash",
            )
        )

    # Add significance legend
    fig.add_annotation(
        xref='paper',
        yref='paper',
        x=1.15,
        y=1.0,
        text='*p<0.1, **p<0.05, ***p<0.01, ns: not significant',
        showarrow=False,
        font=dict(size=10),
        xanchor='right',
        yanchor='top'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Participant Ratings Analysis
    st.subheader("Participant Ratings")

    # Add demographic selector before the plot
    demographic_options = ['None', 'age', 'gender', 'education', 'occupation', 'partyAffiliation']
    selected_demographic = st.selectbox(
        "Group by demographic (optional):", 
        demographic_options,
        key="demographic_select_participant_ratings"
    )
    
    # Process Likert-scale questions
    likert_questions = ['positivity', 'engagement', 'politeness']
    
    for question in likert_questions:
        participant_ratings = []
        for _, row in df.iterrows():
            if isinstance(row['exitSurvey'], dict) and 'userRating' in row['exitSurvey']:
                # Extract demographic info
                demographic_value = 'Unknown'
                if selected_demographic != 'None':
                    demographic_info = row['initialSurvey'].get('demographicInfo', {})
                    if selected_demographic in demographic_info:
                        demo_value = demographic_info[selected_demographic]
                        
                        # Special handling for party affiliation
                        if selected_demographic == 'partyAffiliation' and demo_value:
                            if 'Republican' in demo_value:
                                demo_value = 'Republican'
                            elif 'Democrat' in demo_value:
                                demo_value = 'Democrat'
                        
                        if demo_value and demo_value != 'Prefer not to answer':
                            demographic_value = demo_value
                
                ratings = row['exitSurvey']['userRating'].get('participantRatings', {})
                for participant_data in ratings.values():
                    if question in participant_data:
                        try:
                            rating = float(participant_data[question])
                            participant_ratings.append({
                                'treatment': row['treatmentName'],
                                'rating': rating,
                                'demographic': demographic_value
                            })
                        except (ValueError, TypeError):
                            continue
        
        ratings_df = pd.DataFrame(participant_ratings)
        
        # Filter out 'Unknown' and 'Prefer not to answer' demographics if a demographic is selected
        if selected_demographic != 'None':
            ratings_df = ratings_df[~ratings_df['demographic'].isin(['Unknown', 'Prefer not to answer', ''])]
        
        if not ratings_df.empty:
            treatment_stats = []
            baseline_ratings_by_demo = {}
            
            # First pass to get baseline for each demographic
            if selected_demographic != 'None':
                for demographic in ratings_df['demographic'].unique():
                    demo_baseline = ratings_df[
                        (ratings_df['treatment'] == 'baseline_5') & 
                        (ratings_df['demographic'] == demographic)
                    ]['rating'].values
                    if len(demo_baseline) > 0:
                        baseline_ratings_by_demo[demographic] = demo_baseline
            else:
                baseline_ratings = ratings_df[ratings_df['treatment'] == 'baseline_5']['rating'].values
                if len(baseline_ratings) > 0:
                    baseline_ratings_by_demo['All'] = baseline_ratings
            
            # Second pass for all treatments and their statistics
            if selected_demographic != 'None':
                for treatment in ratings_df['treatment'].unique():
                    for demographic in ratings_df['demographic'].unique():
                        treatment_ratings = ratings_df[
                            (ratings_df['treatment'] == treatment) & 
                            (ratings_df['demographic'] == demographic)
                        ]['rating'].values
                        
                        if len(treatment_ratings) > 0:
                            mean, ci_lower, ci_upper = bootstrap_mean_ci(treatment_ratings)
                            
                            stat = {
                                'treatment': treatment,
                                'demographic': demographic,
                                'mean': mean,
                                'ci_lower': ci_lower,
                                'ci_upper': ci_upper,
                                'n': len(treatment_ratings),
                                'ratings': treatment_ratings
                            }
                            
                            # Calculate p-value if not baseline and we have baseline data for this demographic
                            if treatment != 'baseline_5' and demographic in baseline_ratings_by_demo:
                                p_value = calculate_p_value(treatment_ratings, baseline_ratings_by_demo[demographic])
                                stat['p_value'] = p_value
                                stat['stars'] = get_stars(p_value)
                            
                            treatment_stats.append(stat)
            else:
                # Original code for no demographic filtering
                for treatment in ratings_df['treatment'].unique():
                    treatment_ratings = ratings_df[ratings_df['treatment'] == treatment]['rating'].values
                    mean, ci_lower, ci_upper = bootstrap_mean_ci(treatment_ratings)
                    
                    stat = {
                        'treatment': treatment,
                        'demographic': 'All',
                        'mean': mean,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'n': len(treatment_ratings),
                        'ratings': treatment_ratings
                    }
                    
                    if treatment != 'baseline_5' and 'All' in baseline_ratings_by_demo:
                        p_value = calculate_p_value(treatment_ratings, baseline_ratings_by_demo['All'])
                        stat['p_value'] = p_value
                        stat['stars'] = get_stars(p_value)
                    
                    treatment_stats.append(stat)
            
            # Create plot with significance stars
            fig = go.Figure()
            
            # Reorder treatments so that baseline_5 is always displayed first
            treatment_order = list(ratings_df['treatment'].unique())
            if 'baseline_5' in treatment_order:
                treatment_order.remove('baseline_5')
                treatment_order = ['baseline_5'] + sorted(treatment_order)
            
            # Find max y value for positioning stars
            max_y = max([stat['mean'] + (stat['ci_upper'] - stat['mean']) for stat in treatment_stats])
            
            if selected_demographic != 'None':
                demographic_values = sorted(ratings_df['demographic'].unique())
                demographic_colors = px.colors.qualitative.Plotly[:len(demographic_values)]
                demographic_color_map = dict(zip(demographic_values, demographic_colors))
                
                # Create offsets for each demographic to prevent overlap
                offset_amount = 0.15
                offsets = np.linspace(-offset_amount * (len(demographic_values)-1)/2, 
                                     offset_amount * (len(demographic_values)-1)/2, 
                                     len(demographic_values))
                offset_map = dict(zip(demographic_values, offsets))
                
                # Group by demographic for the plot
                for demographic in demographic_values:
                    # Get all stats for this demographic
                    demo_stats = [s for s in treatment_stats if s['demographic'] == demographic]
                    
                    # Create a dictionary mapping treatment to stat for easier lookup
                    treatment_to_stat = {s['treatment']: s for s in demo_stats}
                    
                    # Create positions list for this demographic, ensuring we follow treatment_order
                    positions = []
                    demo_means = []
                    demo_ci_upper = []
                    demo_ci_lower = []
                    stats_to_plot = []
                    
                    for i, treatment in enumerate(treatment_order):
                        if treatment in treatment_to_stat:
                            stat = treatment_to_stat[treatment]
                            positions.append({
                                'original': treatment,
                                'position': i + offset_map[demographic]
                            })
                            demo_means.append(stat['mean'])
                            demo_ci_upper.append(stat['ci_upper'] - stat['mean'])
                            demo_ci_lower.append(stat['mean'] - stat['ci_lower'])
                            stats_to_plot.append(stat)
                    
                    # Only add trace if we have data for this demographic
                    if positions:
                        fig.add_trace(go.Scatter(
                            x=[p['position'] for p in positions],
                            y=demo_means,
                            error_y=dict(
                                type='data',
                                symmetric=False,
                                array=demo_ci_upper,
                                arrayminus=demo_ci_lower,
                                visible=True
                            ),
                            mode='markers',
                            marker=dict(size=10, color=demographic_color_map[demographic]),
                            name=demographic,
                            text=[p['original'] for p in positions],
                            hovertemplate=(
                                "<b>Treatment:</b> %{text}<br>"
                                "<b>Demographic:</b> " + demographic + "<br>"
                                "<b>Mean Rating:</b> %{y:.2f}<br>"
                                "<extra></extra>"
                            )
                        ))
                        
                        # Add annotations for each point
                        for pos, stat in zip(positions, stats_to_plot):
                            annotation_text = f'n={stat["n"]}'
                            if stat['treatment'] != 'baseline_5' and 'p_value' in stat:
                                annotation_text += f'<br>{stat["stars"]}'
                            
                            fig.add_annotation(
                                x=pos['position'],
                                y=stat['mean'],
                                text=annotation_text,
                                yshift=40,
                                showarrow=False,
                                font=dict(size=10)
                            )
                
                # Update x-axis to show treatment names at the right positions
                fig.update_layout(
                    xaxis=dict(
                        tickmode='array',
                        tickvals=list(range(len(treatment_order))),
                        ticktext=treatment_order
                    )
                )
            else:
                # Original code for treatment-only plot
                color_sequence = px.colors.qualitative.D3[:len(treatment_order)]
                color_map = dict(zip(treatment_order, color_sequence))
                
                for stat in treatment_stats:
                    fig.add_trace(go.Scatter(
                        x=[stat['treatment']],
                        y=[stat['mean']],
                        error_y=dict(
                            type='data',
                            symmetric=False,
                            array=[stat['ci_upper'] - stat['mean']],
                            arrayminus=[stat['mean'] - stat['ci_lower']],
                            visible=True
                        ),
                        mode='markers',
                        marker=dict(size=12, color=color_map[stat['treatment']]),
                        name=stat['treatment'],
                        hovertemplate=(
                            "<b>Treatment:</b> %{x}<br>"
                            "<b>Mean Rating:</b> %{y:.2f}<br>"
                            "<b>Sample Size:</b> " + str(stat['n']) +
                            "<extra></extra>"
                        )
                    ))
                    
                    # Add sample sizes
                    fig.add_annotation(
                        x=stat['treatment'],
                        y=0.3,
                        text=f"n={stat['n']}",
                        showarrow=False,
                        font=dict(size=10),
                        yanchor='top'
                    )
                    
                    # Add significance markers
                    if stat['treatment'] != 'baseline_5' and 'p_value' in stat:
                        add_significance_markers(fig, stat['treatment'], max_y, stat['p_value'])
                
                # Add baseline reference line
                baseline_mean = next((s['mean'] for s in treatment_stats if s['treatment'] == 'baseline_5'), None)
                if baseline_mean is not None:
                    fig.add_hline(
                        y=baseline_mean,
                        line_dash="dash",
                        line_color="red",
                        line_width=2,
                        annotation_text="Control Group",
                        annotation_position="top left",
                        annotation_font_size=12,
                        annotation_font_color="red"
                    )
            
            # Update layout
            title_text = f"Participant {question.capitalize()} Ratings by Treatment"
            if selected_demographic != 'None':
                title_text += f" (by {selected_demographic})"
                
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
                yaxis_title=f"Average {question.capitalize()} Rating",
                xaxis=dict(
                    tickangle=45,
                    tickfont=dict(size=12)
                ),
                yaxis=dict(
                    zeroline=True,
                    zerolinewidth=2.5,
                    zerolinecolor='black',
                    gridcolor='lightgray',
                    range=[0, max_y + 0.5]
                ),
                showlegend=(selected_demographic != 'None'),
                legend=dict(
                    title=dict(text=selected_demographic if selected_demographic != 'None' else ''),
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=1.15
                ),
                height=600,
                template='simple_white',
                margin=dict(
                    t=100,
                    b=120,
                    r=150,
                    l=80
                )
            )
            
            # Add significance legend
            fig.add_annotation(
                xref='paper',
                yref='paper',
                x=1.15,
                y=1.0,
                text='*p<0.1, **p<0.05, ***p<0.01, ns: not significant',
                showarrow=False,
                font=dict(size=10),
                xanchor='right',
                yanchor='top'
            )
            
            st.plotly_chart(fig, use_container_width=True)

    # Categorical questions analysis
    categorical_questions = ['politicalAffiliation', 'sharedValues', 'isBot', 'usedAI']
    
    def bootstrap_proportion_ci(successes, total, n_bootstrap=10000):
        """Calculate confidence interval for proportion using bootstrap"""
        data = np.concatenate([np.ones(successes), np.zeros(total - successes)])
        bootstrap_props = np.zeros(n_bootstrap)
        
        for i in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=total, replace=True)
            bootstrap_props[i] = np.mean(bootstrap_sample)
        
        ci_lower, ci_upper = np.percentile(bootstrap_props, [2.5, 97.5])
        return ci_lower * 100, ci_upper * 100
    
    def calculate_categorical_p_value(treatment_data, baseline_data, response, n_bootstrap=10000):
        """Calculate bootstrap p-value for difference in proportions"""
        # Convert to binary data (1 for selected response, 0 for others)
        t_binary = (treatment_data == response).astype(int)
        b_binary = (baseline_data == response).astype(int)
        
        observed_diff = np.mean(t_binary) - np.mean(b_binary)
        combined = np.concatenate([t_binary, b_binary])
        
        null_diffs = np.zeros(n_bootstrap)
        for i in range(n_bootstrap):
            perm = np.random.permutation(len(combined))
            boot_treat = combined[perm[:len(t_binary)]]
            boot_base = combined[perm[len(t_binary):]]
            null_diffs[i] = np.mean(boot_treat) - np.mean(boot_base)
        
        p_value = np.mean(np.abs(null_diffs) >= np.abs(observed_diff))
        return p_value
    
    for question in categorical_questions:
        categorical_data = []
        for _, row in df.iterrows():
            if isinstance(row['exitSurvey'], dict) and 'userRating' in row['exitSurvey']:
                ratings = row['exitSurvey']['userRating'].get('participantRatings', {})
                for participant_data in ratings.values():
                    if question in participant_data:
                        categorical_data.append({
                            'treatment': row['treatmentName'],
                            'response': participant_data[question]
                        })
        
        df_cat = pd.DataFrame(categorical_data)
        
        if not df_cat.empty:
            # Calculate proportions
            props = df_cat.groupby(['treatment', 'response']).size().reset_index(name='count')
            totals = df_cat.groupby('treatment').size().reset_index(name='total')
            props = props.merge(totals, on='treatment')
            props['percentage'] = props['count'] / props['total'] * 100
            
            # Calculate confidence intervals using bootstrap
            props['ci_lower'], props['ci_upper'] = zip(*props.apply(
                lambda row: bootstrap_proportion_ci(row['count'], row['total']), 
                axis=1
            ))
            
            # Calculate significance vs baseline for each response
            baseline_data = df_cat[df_cat['treatment'] == 'baseline_5']['response']
            
            # Create offset mapping for responses
            unique_responses = sorted(props['response'].unique())
            offsets = np.linspace(-0.2, 0.2, len(unique_responses))
            offset_map = dict(zip(unique_responses, offsets))
            
            # Create plot
            fig = go.Figure()
            
            # Create color map for responses
            color_sequence = px.colors.qualitative.Set2[:len(unique_responses)]
            color_map = dict(zip(unique_responses, color_sequence))
            
            # Find max y for positioning stars
            max_y = props['percentage'].max() + 10
            
            for response in unique_responses:
                response_data = props[props['response'] == response]
                
                fig.add_trace(go.Scatter(
                    x=[t + offset_map[response] for t in range(len(response_data['treatment']))],
                    y=response_data['percentage'],
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array=response_data['ci_upper'] - response_data['percentage'],
                        arrayminus=response_data['percentage'] - response_data['ci_lower'],
                        visible=True
                    ),
                    mode='markers',
                    name=response,
                    marker=dict(
                        size=10,
                        color=color_map[response]
                    ),
                    hovertemplate=(
                        "<b>Treatment:</b> %{customdata}<br>"
                        "<b>Response:</b> " + response + "<br>" +
                        "<b>Percentage:</b> %{y:.1f}%<br>" +
                        "<b>Count:</b> %{text}" +
                        "<extra></extra>"
                    ),
                    customdata=response_data['treatment'],
                    text=response_data['count']
                ))
                
                # Add significance stars for each treatment vs baseline
                for treatment in props['treatment'].unique():
                    if treatment != 'baseline_5':
                        treatment_data = df_cat[df_cat['treatment'] == treatment]['response']
                        p_value = calculate_categorical_p_value(treatment_data, baseline_data, response)
                        stars = get_stars(p_value)
                        if stars or stars == 'ns':  # Add both stars and 'ns'
                            fig.add_annotation(
                                x=list(props['treatment'].unique()).index(treatment) + offset_map[response],
                                y=max_y,
                                text=stars if stars else 'ns',
                                showarrow=False,
                                font=dict(size=14)
                            )
            
            # Update layout
            fig.update_layout(
                title=dict(
                    text=f"{question.capitalize()} Responses by Treatment",
                    x=0.5,
                    y=0.95,
                    xanchor='center',
                    yanchor='top',
                    font=dict(size=20)
                ),
                xaxis=dict(
                    tickmode='array',
                    ticktext=props['treatment'].unique(),
                    tickvals=list(range(len(props['treatment'].unique()))),
                    tickangle=45,
                    title="Treatment Group"
                ),
                yaxis=dict(
                    title="Percentage",
                    range=[0, max_y + 5],  # Add space for stars
                    gridcolor='lightgray'
                ),
                showlegend=True,
                legend=dict(
                    title=dict(text="Response"),
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=1.15
                ),
                height=600,
                template='simple_white',
                margin=dict(
                    t=100,
                    b=120,
                    r=150,
                    l=80
                )
            )
            
            # Add significance legend
            fig.add_annotation(
                xref='paper',
                yref='paper',
                x=1.15,
                y=1.0,
                text='*p<0.1, **p<0.05, ***p<0.01, ns: not significant',
                showarrow=False,
                font=dict(size=10),
                xanchor='right',
                yanchor='top'
            )
            
            st.plotly_chart(fig, use_container_width=True)

def display_demographic_regression(treatment_data):
    st.subheader("Demographic Regression")
    st.markdown("This analysis shows how demographic factors and treatment assignment affect user responses.")
    
    # Add options for regression analysis
    st.markdown("### Regression Configuration")
    
    # Option for dependent variable
    dv_options = [
        "Average Rating", 
        "informative",
        "positivity",
        "engagement",
        "politeness",
        "∆AI Comfort", 
        "∆AI Suggestions", 
        "∆Less Toxic", 
        "∆Less Polarizing",
        "∆Reduce Misinfo", 
        "∆AI Accuracy", 
        "∆AI Regulation"
    ]
    
    selected_dv = st.selectbox(
        "Select dependent variable:",
        dv_options,
        key="selected_dv",
        index=0
    )
    
    # Option for treatment variable encoding
    treatment_encoding = st.radio(
        "Treatment variable encoding:",
        ["Binary (Treatment vs Control)", "Categorical (Specific Treatments)"],
        key="treatment_encoding",
        index=1
    )
    
    # Prepare data for regression
    regression_data = prepare_regression_data(treatment_data, treatment_encoding, selected_dv, use_aggregated_demographics=True)
    
    if regression_data is not None and not regression_data.empty:
        # Run regression and display results
        run_and_display_regression(regression_data, selected_dv)
    else:
        st.warning("Insufficient data for regression analysis.")

def prepare_regression_data(treatment_data, treatment_encoding, selected_dv, use_aggregated_demographics=False):
    """
    Prepare data for regression analysis.
    
    Args:
        treatment_data: Dictionary containing treatment data
        treatment_encoding: How to encode the treatment variable
        selected_dv: The dependent variable to use in the regression
        use_aggregated_demographics: Whether to use aggregated demographic categories
    """
    regression_rows = []
    
    # Map selected_dv to the actual data field
    dv_mapping = {
        "Average Rating": "avg_rating",
        "informative": "informative",
        "positivity": "positivity",
        "engagement": "engagement",
        "politeness": "politeness",
        "∆AI Comfort": "comfortableWithAI",
        "∆AI Suggestions": "aiSuggestions",
        "∆Less Toxic": "lessToxic",
        "∆Less Polarizing": "lessPolarizing",
        "∆Reduce Misinfo": "reduceMisinformation",
        "∆AI Accuracy": "aiContentAccurate",
        "∆AI Regulation": "aiRegulation"
    }
    
    dv_field = dv_mapping[selected_dv]
    
    # Process each treatment group
    for treatment_name, users in treatment_data.items():
        for user_id, user_data in users.items():
            # Skip if missing required data
            if selected_dv in ["Average Rating", "informative", "positivity", "engagement", "politeness"]:
                # Get the user's participantID
                participant_id = user_id
                
                # Find all ratings for this participant in df
                user_rows = df[df['id'] == participant_id]
                
                if user_rows.empty:
                    continue
                
                if selected_dv == "Average Rating":
                    # Get the user's ratings from their row
                    user_ratings = []
                    for _, user_row in user_rows.iterrows():
                        if isinstance(user_row['exitSurvey'], dict) and 'userRating' in user_row['exitSurvey']:
                            ratings = user_row['exitSurvey']['userRating'].get('commentRatings', {})
                            for comment_id, rating_data in ratings.items():
                                if 'valueComment' in rating_data:
                                    try:
                                        rating = float(rating_data['valueComment'])
                                        user_ratings.append(rating)
                                    except (ValueError, TypeError):
                                        continue
                    
                    if not user_ratings:
                        continue
                    
                    dv_value = sum(user_ratings) / len(user_ratings)
                
                elif selected_dv == "informative":
                    try:
                        exit_survey = user_rows["exitSurvey"].iloc[0]
                        if 'overallExperience' in exit_survey and 'informative' in exit_survey['overallExperience']:
                            user_informative = likert_to_numeric(exit_survey['overallExperience']["informative"])
                            dv_value = user_informative
                        else:
                            # Skip this user if the required data is missing
                            continue
                    except (KeyError, TypeError):
                        # Handle case where the structure isn't as expected
                        continue
                elif selected_dv in ["positivity", "engagement", "politeness"]:
                    # Get the user's ratings for the selected metric
                    user_ratings = []
                    for _, user_row in user_rows.iterrows():
                        if isinstance(user_row['exitSurvey'], dict) and 'userRating' in user_row['exitSurvey']:
                            ratings = user_row['exitSurvey']['userRating'].get('participantRatings', {})
                            for participant_data in ratings.values():
                                if selected_dv in participant_data:
                                    try:
                                        rating = float(participant_data[selected_dv])
                                        user_ratings.append(rating)
                                    except (ValueError, TypeError):
                                        continue
                    
                    if not user_ratings:
                        continue
                    
                    dv_value = sum(user_ratings) / len(user_ratings)
            else:
                # For difference metrics, we need the differences data
                if 'differences' not in user_data or dv_field not in user_data['differences']:
                    continue
                
                try:
                    dv_value = float(user_data['differences'][dv_field])
                except (ValueError, TypeError):
                    continue
            
            # Extract demographic information
            demographics = user_data.get('demographics', {})
            if not demographics:
                continue
            
            # Create data row
            data_row = {
                'participantID': user_id,
                'dv_value': dv_value,
                'treatment': treatment_name
            }
            
            # Add binary treatment indicator
            data_row['is_treatment'] = 1 if treatment_name != 'baseline_5' else 0
            
            # Process demographics with aggregation options
            
            # Age
            age = demographics.get('age')
            if age and age not in ['Prefer not to answer', 'Under 18']:
                if use_aggregated_demographics:
                    # Aggregate age into young/old
                    if age in ["18-24", "25-34", "35-44"]:
                        data_row['age'] = 'young'
                    else:
                        data_row['age'] = 'old'
                else:
                    data_row['age'] = str(age)
            
            # Gender
            gender = demographics.get('gender')
            if gender and gender not in ['Prefer not to answer', 'Non-binary']:
                gender = str(gender)
                if use_aggregated_demographics:
                    # Aggregate gender into female/not female
                    if gender.lower() == 'female':
                        data_row['gender'] = 'female'
                    else:
                        data_row['gender'] = 'not_female'
                else:
                    # Map gender values consistently
                    if gender.lower() == 'female':
                        data_row['gender'] = 'Female'
                    elif gender.lower() == 'male':
                        data_row['gender'] = 'Male'
                    elif gender.lower() in ['non-binary', 'nonbinary', 'non binary']:
                        data_row['gender'] = 'Non-binary'
            
            # Education
            education = demographics.get('education')
            if education and education != 'Prefer not to answer':
                education = str(education)
                if use_aggregated_demographics:
                    # Aggregate education into educated/not educated
                    if education in ["4-year college degree", "Postgraduate degree (MA, MBA, JD, PhD, etc)"]:
                        data_row['education'] = 'educated'
                    else:
                        data_row['education'] = 'not_educated'
                else:
                    data_row['education'] = education
            
            # Occupation
            occupation = demographics.get('occupation')
            # Handle occupation as list or string
            if isinstance(occupation, list) and len(occupation) > 0:
                occupation = occupation[0]
            occupation = str(occupation)
            if occupation and occupation not in ['Prefer not to answer', 'Other (please specify)']:

                # if 'Other' in occupation:
                #     occupation = 'Other (please specify)'
                
                if use_aggregated_demographics:
                    # Aggregate occupation into full-time/not full-time
                    if occupation == "Employed full-time":
                        data_row['occupation'] = 'full_time'
                    else:
                        data_row['occupation'] = 'not_full_time'
                else:
                    # Fix specific occupation values
                    if 'Self' in occupation:
                        occupation = 'Self-employed'
                    data_row['occupation'] = occupation
            
            # Party Affiliation - keep granularity even with aggregation
            party = demographics.get('partyAffiliation')
            if party and party not in ['Prefer not to answer', 'Other']:
                party = str(party)
                # Simplify party mapping to ensure single values
                if party == "Strong Republican":
                    # data_row['party'] = 'Strong_Republican'
                    data_row['party'] = 'Republican'
                elif party == "Moderate Republican":
                    # data_row['party'] = 'Moderate_Republican'
                    data_row['party'] = 'Republican'
                elif party == "Republican":
                    data_row['party'] = 'Republican'
                elif party == "Strong Democrat":
                    # data_row['party'] = 'Strong_Democrat'
                    data_row['party'] = 'Democrat'
                elif party == "Moderate Democrat":
                    # data_row['party'] = 'Moderate_Democrat'
                    data_row['party'] = 'Democrat'
                elif party == "Democrat":
                    data_row['party'] = 'Democrat'
                elif party == "Independent":
                    data_row['party'] = 'Independent'
            
            regression_rows.append(data_row)
    
    # Create DataFrame
    if not regression_rows:
        return None
    
    regression_df = pd.DataFrame(regression_rows)
    
    # Only convert the dv_value and is_treatment columns to numeric
    # Leave categorical columns as they are until dummy encoding
    for col in ['dv_value', 'is_treatment']:
        if col in regression_df.columns:
            try:
                # Convert boolean columns to integers
                if regression_df[col].dtype == bool:
                    regression_df[col] = regression_df[col].astype(int)
                # Convert other columns to float
                else:
                    regression_df[col] = pd.to_numeric(regression_df[col], errors='coerce')
            except Exception as e:
                st.error(f"Could not convert column {col} to numeric. Error: {str(e)}")
                if col != 'dv_value':  # Don't drop dv_value as it's essential
                    regression_df = regression_df.drop(col, axis=1)
    
    # Drop any rows with NaN values in essential columns only
    regression_df = regression_df.dropna(subset=['dv_value'])
    
    # Ensure treatment columns are properly encoded
    if treatment_encoding == "Categorical (Specific Treatments)":
        # Explicitly use baseline_5 as the reference category
        treatment_dummies = pd.get_dummies(regression_df['treatment'], prefix='treatment')
        if 'treatment_baseline_5' in treatment_dummies.columns:
            treatment_dummies = treatment_dummies.drop('treatment_baseline_5', axis=1)
        regression_df = pd.concat([regression_df, treatment_dummies], axis=1)
    
    # Create dummy variables for categorical demographic variables with specific reference categories
    demographic_cols = ['age', 'gender', 'education', 'occupation', 'party']
    if use_aggregated_demographics:     
        reference_categories = {
            'age': 'young',
            'gender': 'female',
            'education': 'educated',
            'occupation': 'not_full_time',
            'party': 'Democrat'
        }
    else:
        reference_categories = {
            'age': '25_34',
            'gender': 'Female',
            'education': '4_year_college_degree',
            'occupation': 'Employed_full_time',
            'party': 'Democrat'
        }
    for col in demographic_cols:
        if col in regression_df.columns:
            # Create dummies without dropping any category
            dummies = pd.get_dummies(regression_df[col], prefix=col)
            
            # Clean column names to remove ALL special characters
            clean_columns = []
            for column in dummies.columns:
                # Replace all special characters with underscores
                clean_col = re.sub(r'[^a-zA-Z0-9_]', '_', column)
                clean_columns.append(clean_col)
            
            dummies.columns = clean_columns
            
            # Drop the reference category column if it exists
            ref_col = f"{col}_{reference_categories[col]}"
            if ref_col in dummies.columns:
                dummies = dummies.drop(ref_col, axis=1)
                
            regression_df = pd.concat([regression_df, dummies], axis=1)
    
    # Remove non-numeric columns except those needed for identification
    cols_to_keep = ['participantID', 'dv_value', 'treatment', 'is_treatment']
    
    # Add treatment dummy columns
    cols_to_keep += [col for col in regression_df.columns if col.startswith('treatment_')]
    
    # Add demographic dummy columns
    for prefix in ['age_', 'gender_', 'education_', 'occupation_', 'party_']:
        cols_to_keep += [col for col in regression_df.columns if col.startswith(prefix)]
    
    # Filter to only keep the columns we want
    regression_df = regression_df[cols_to_keep]
    
    # Final check to ensure all columns (except participantID) are numeric
    for col in regression_df.columns:
        if col != 'participantID':
            regression_df[col] = pd.to_numeric(regression_df[col], errors='coerce')
    
    return regression_df

def run_and_display_regression(data, selected_dv):
    """
    Run regression analysis and display results.
    
    Args:
        data: DataFrame prepared for regression
        selected_dv: The dependent variable being analyzed
    """
    import statsmodels.formula.api as smf
    
    st.markdown(f"### Regression Results for {selected_dv}")
    
    # Display data summary
    st.markdown(f"**Number of observations:** {len(data)}")
    
    # Allow user to select which variables to include
    st.markdown("#### Select Variables for Regression")
    
    # Get available demographic variables (those that have been dummy-encoded)
    demographic_prefixes = ['age_', 'gender_', 'education_', 'occupation_', 'party_']
    available_demographics = []
    
    for prefix in demographic_prefixes:
        demo_cols = [col for col in data.columns if col.startswith(prefix)]
        if demo_cols:
            available_demographics.append(prefix[:-1])  # Remove trailing underscore
    
    # Let user select which demographics to include
    selected_demographics = st.multiselect(
        "Select demographic variables to include:",
        available_demographics,
        default=available_demographics
    )
    
    # Determine treatment variable based on available columns
    treatment_vars = []
    if 'is_treatment' in data.columns:
        treatment_vars.append('is_treatment')
    
    treatment_specific_cols = [col for col in data.columns if col.startswith('treatment_')]
    if treatment_specific_cols:
        treatment_vars.append('specific_treatments')
    
    selected_treatment = st.radio(
        "Select treatment variable:",
        treatment_vars,
        key="selected_treatment_var",
        index=1
    )
    
    # First, create a copy of the data with cleaned column names
    regression_data = data.copy()
    
    # Create a mapping of original column names to clean column names
    column_mapping = {}
    reverse_mapping = {}
    
    for col in regression_data.columns:
        # Replace spaces and special characters with underscores
        clean_col = col.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '').replace(',', '')
        
        # If the column name has changed, add it to the mapping
        if clean_col != col:
            column_mapping[col] = clean_col
            reverse_mapping[clean_col] = col
    
    # Rename the columns in the dataframe
    regression_data = regression_data.rename(columns=column_mapping)
    
    interaction_demographics = st.multiselect(
        "Select demographics for treatment interactions:",
        selected_demographics,  # Only show demographics that are already selected as main effects
        default=[]
    )

    # Now build the formula with the clean column names
    formula_parts = ["dv_value ~ "]

    # Add treatment term if selected
    if selected_treatment == 'is_treatment' and 'is_treatment' in regression_data.columns:
        formula_parts.append('is_treatment')
        
        for demo in interaction_demographics:
            demo_cols = [col for col in regression_data.columns if col.startswith(f"{demo}_")]
            for demo_col in demo_cols:
                formula_parts.append(f" + is_treatment:{demo_col}")
            
    elif selected_treatment == 'specific_treatments':
        # Get all treatment dummy columns
        treatment_cols = [col for col in regression_data.columns if col.startswith('treatment_')]
        
        if treatment_cols:
            formula_parts.append(" + ".join(treatment_cols))
            
            
            for demo in interaction_demographics:
                demo_cols = [col for col in regression_data.columns if col.startswith(f"{demo}_")]
                for treatment_col in treatment_cols:
                    for demo_col in demo_cols:
                        formula_parts.append(f" + {treatment_col}:{demo_col}")
    
    # Add demographic terms
    for demo in selected_demographics:
        # Get columns for this demographic
        demo_cols = [col for col in regression_data.columns if col.startswith(f"{demo}_")]
        
        if demo_cols:
            if len(formula_parts) > 1:  # If we already have other terms
                formula_parts.append(" + ")
            formula_parts.append(" + ".join(demo_cols))

    formula = "".join(formula_parts)

    column_values_mapping_to_remove =['occupation_Other__please_specify_']

    for col in column_values_mapping_to_remove:
        if col in regression_data.columns:
            regression_data = regression_data[regression_data[col] != True]
    

    # Run regression
    try:
        # Print formula for debugging
        st.write("Regression formula:")
        st.code(formula)
        
        # Determine which regression model to use based on selected_dv
        if selected_dv in ['Average Rating', 'informative', 'positivity', 'engagement', 'politeness']:
            # Linear regression for continuous variables
            st.write("Using Linear Regression (OLS)")
            model = smf.ols(formula=formula, data=regression_data).fit()
        elif selected_dv.startswith('∆'):
            # Mixed effects model for delta variables
            st.write("Using Mixed Effects Model for change variables")
            
            # Extract formula parts
            dv_part = formula.split('~')[0].strip()
            exog_part = formula.split('~')[1].strip()
            
            # Add user ID as a random effect if available
            if 'user_id' in regression_data.columns:
                # Create design matrix from formula
                exog = smf.ols(formula=f"0 ~ {exog_part}", data=regression_data).fit().model.exog
                
                model = sm.MixedLM(
                    regression_data[dv_part],
                    exog,
                    groups=regression_data['user_id']
                ).fit()
            else:
                # Fallback to OLS with baseline adjustment if user_id not available
                st.write("User ID not available, using OLS with baseline adjustment")
                model = smf.ols(formula=formula, data=regression_data).fit()
        else:
            # Default to OLS for any other variables
            st.write("Using default Linear Regression (OLS)")
            model = smf.ols(formula=formula, data=regression_data).fit()
        
        st.write(model.summary())
        
        # Display regression results
        st.markdown("#### Regression Output")
        
        # Create a DataFrame for the results
        results_df = pd.DataFrame({
            'Variable': model.params.index,
            'Coefficient': model.params.values,
            'Std Error': model.bse.values,
            'P-value': model.pvalues.values
        })
        
        # Map the clean column names back to the original names for display
        results_df['Variable'] = results_df['Variable'].apply(
            lambda x: reverse_mapping.get(x.split('[')[0], x) if '[' in x else reverse_mapping.get(x, x)
        )
        
        # Add significance stars
        results_df['Significance'] = results_df['P-value'].apply(get_stars)
        
        # Format the values to be more readable
        results_df['Coefficient'] = results_df['Coefficient'].round(4)
        results_df['Std Error'] = results_df['Std Error'].round(4)
        results_df['P-value'] = results_df['P-value'].round(4)
        
        # Display the results table
        st.dataframe(results_df)
        
        # Display model fit statistics
        st.markdown("#### Model Fit")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("R-squared", f"{model.rsquared:.4f}")
            st.metric("F-statistic", f"{model.fvalue:.4f}")
        # Display model statistics
        st.markdown("#### Model Statistics")
        stats_df = pd.DataFrame({
            'Statistic': ['R-squared', 'Adj. R-squared', 'F-statistic', 'Prob (F-statistic)', 'AIC', 'BIC'],
            'Value': [
                round(model.rsquared, 4),
                round(model.rsquared_adj, 4),
                round(model.fvalue, 4),
                round(model.f_pvalue, 4),
                round(model.aic, 4),
                round(model.bic, 4)
            ]
        })
        st.dataframe(stats_df)

    except Exception as e:
        st.error(f"Error running regression: {str(e)}")
        st.markdown("This could be due to perfect multicollinearity or insufficient data. Try selecting different variables.")

# Main dashboard
def main():
    st.title("Social Media Agents Analytics Dashboard")
    
    display_key_metrics()
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Initial Survey", 
        "Exit Survey", 
        "Treatment Effects", 
        "Open-ended Feedback",
        "User Ratings",
        "Demographic Regression"
    ])
    
    with tab1:
        display_initial_survey()
    
    with tab2:
        display_exit_survey()
        # Add barriers contingency table to exit survey tab
        # display_barriers_contingency()
    
    with tab3:
        st.subheader("Treatment Effects")
        st.markdown("Here we show the difference in pre and post survey scores for each question across all groups (control and treatment). For each question, we show the mean difference and the 95% confidence interval. To test for significance, we perform a two-sided t-test.")
        treatment_data = get_treatment_data(df_ate)
        plot_treatment_effects(treatment_data)

    with tab4:
        display_open_ended_feedback()
        
    with tab5:
        display_user_ratings()

    with tab6:
        display_demographic_regression(treatment_data)

if __name__ == "__main__":
    main()

    