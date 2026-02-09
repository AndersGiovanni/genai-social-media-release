import streamlit as st

def create_filters(df, include_participant=True, key_prefix=""):
    """
    Create filter controls for game ID, treatment, and optionally participant.
    
    Args:
        df: DataFrame containing the data
        include_participant: Boolean to determine if participant filter should be included
        key_prefix: String prefix for widget keys to avoid duplicates
    
    Returns:
        tuple: (filtered_df, selected_game_id, selected_treatment, selected_participant)
               selected_participant will be None if include_participant is False
    """
    # Get unique values for filters
    game_ids = ['All'] + sorted(df['gameID'].unique().tolist())
    treatment_names = ['All'] + sorted(df['treatmentName'].unique().tolist())
    
    # Create columns for filters
    if include_participant:
        col1, col2, col3 = st.columns(3)
    else:
        col1, col2 = st.columns(2)
    
    # Add filters to columns with unique keys
    with col1:
        selected_game_id = st.selectbox("Filter by Game ID", game_ids, key=f"{key_prefix}_game_id")
    with col2:
        selected_treatment = st.selectbox("Filter by Treatment", treatment_names, key=f"{key_prefix}_treatment")
    
    # Filter the dataframe
    filtered_df = df
    if selected_game_id != 'All':
        filtered_df = filtered_df[filtered_df['gameID'] == selected_game_id]
    if selected_treatment != 'All':
        filtered_df = filtered_df[filtered_df['treatmentName'] == selected_treatment]
    
    # Add participant filter if requested
    selected_participant = None
    if include_participant:
        with col3:
            participant_ids = filtered_df['participantIdentifier'].tolist()
            selected_participant = st.selectbox("Select a participant", participant_ids, key=f"{key_prefix}_participant")
            if selected_participant:
                filtered_df = filtered_df[filtered_df['participantIdentifier'] == selected_participant]
    
    return filtered_df, selected_game_id, selected_treatment, selected_participant

def filter_data(df, selected_games, selected_actions, selected_comments):
    """Filter dataframe based on selected games and their action/comment columns.
    We want to filter out the unselected action and comment columns for each game.
    This is because some of the rounds got corrupted and we don't want to show them.
    Args:
        df: DataFrame containing the data
        selected_games: List of game IDs to filter by
        selected_actions: List of action columns to filter by
        selected_comments: List of comment columns to filter by
        
    Returns:
        DataFrame: Filtered dataframe
    """
    if not selected_games:
        return df
        
    filtered_df = df[df['id'].isin(selected_games)].copy()
    
    # Drop unselected action/comment columns for each game
    for game_id in selected_games:
        game_mask = filtered_df['id'] == game_id
        
        # Drop unselected action columns
        action_cols = [col for col in filtered_df.columns if col.startswith('actions-')]
        for col in action_cols:
            if f"{game_id}_{col}" not in selected_actions:
                filtered_df.loc[game_mask, col] = None
                
        # Drop unselected comment columns 
        comment_cols = [col for col in filtered_df.columns if col.startswith('comments-')]
        for col in comment_cols:
            if f"{game_id}_{col}" not in selected_comments:
                filtered_df.loc[game_mask, col] = None
                
    return filtered_df