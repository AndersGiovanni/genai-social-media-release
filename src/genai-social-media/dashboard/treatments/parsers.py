from datetime import datetime, timezone
import json
import streamlit as st
import pandas as pd
from Levenshtein import ratio

def get_round_number(timestamp, round_starts):
    """
    Determine which round a timestamp belongs to based on round start times
    Returns 0, 1, or 2
    """
    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    
    round_times = [datetime.fromisoformat(str(t).replace('Z', '+00:00')) 
                  for t in round_starts]
    
    # If before first round, return 0
    if timestamp < round_times[1]:
        return 0
    # If before second round, return 1    
    elif timestamp < round_times[2]:
        return 1
    # Must be round 2
    else:
        return 2


# @st.cache_data
def process_player_rounds(processed_data, df_player_round):
    """Process player round data and merge with engagement data"""
    
    # Map treatment names to their interaction types
    treatment_interactions = {
        'suggestions_5': ['suggestions'],
        'feedback_5': ['feedback'],
        'chat_5': ['chat'],
        'conversation_5': ['conversation_starter'],
        'control': ['suggestions']  # control group only needs suggestions for comparison
    }
    
    for group in processed_data.keys():  # Handle all treatment groups dynamically
        game_data = processed_data[group]
        if not isinstance(game_data, dict):  # Skip if not a dict
            continue
            
        for game_id, game_info in game_data.items():
            # Get rows for this game
            game_rows = df_player_round[df_player_round['gameID'] == game_id].to_dict('records')
            round_starts = game_info['round_starts']
            
            # Get treatment name from game info
            treatment_name = game_info.get('treatmentName', 'control')
            
            # Initialize only the relevant interaction types for this treatment
            relevant_interactions = treatment_interactions.get(treatment_name, [])
            for interaction_type in relevant_interactions:
                game_info[interaction_type] = {0: {}, 1: {}, 2: {}}
            
            for row in game_rows:
                # Handle suggestions if it's relevant for this treatment
                if 'suggestions' in relevant_interactions:
                    suggestions = json.loads(row['suggestions']) if not pd.isna(row['suggestions']) else []
                    selected = json.loads(row['suggestionsSelected']) if not pd.isna(row['suggestionsSelected']) else []
                    
                    if suggestions:
                        timestamp = suggestions[0][2]
                        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        round_num = None
                        for i, start in enumerate(round_starts):
                            if timestamp >= start:
                                round_num = i
                        
                        if round_num is not None:
                            player_id = row['playerID']
                            if player_id not in game_info['suggestions'][round_num]:
                                game_info['suggestions'][round_num][player_id] = {
                                    'suggestions': suggestions,
                                    'selected': selected
                                }
                
                # Handle other interaction types if relevant
                for field in set(relevant_interactions) - {'suggestions'}:
                    data = json.loads(row[field]) if not pd.isna(row[field]) else []
                    if not data:
                        continue
                    
                    # Extract timestamp based on data type
                    timestamp = None
                    if field == 'chat':
                        timestamp = data[0][1]
                    elif field == 'conversation_starter':
                        timestamp = data[0][2]
                    elif field == 'feedback':
                        first_key = list(data.keys())[0]
                        timestamp = data[first_key][0][2]
                    
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    
                    round_num = None
                    for i, start in enumerate(round_starts):
                        if timestamp >= start:
                            round_num = i
                    
                    if round_num is not None:
                        player_id = row['playerID']
                        if player_id not in game_info[field][round_num]:
                            game_info[field][round_num][player_id] = data
    
    return processed_data


def match_selections_to_comments(processed_data, suggestions_data):
    """Match selections from suggestions data to comments in processed data"""
    
    for group in processed_data:
        for game_id, game_data in processed_data[group].items():
            if game_id not in suggestions_data:
                continue
                
            # For each round in the suggestions data
            for round_num, round_data in suggestions_data[game_id].items():
                round_num = int(round_num)
                
                # Skip if no comments for this round
                if round_num >= len(game_data['comments']):
                    continue
                    
                # For each player's generations in this round
                for player_id, player_data in round_data.items():
                    for generation in player_data['generations']:
                        if 'selections' not in generation:
                            continue
                            
                        for selection in generation['selections']:
                            selection_text = selection['text']
                            
                            # Look for matching comment in this round
                            for comment in game_data['comments'][round_num]:
                                comment_text = comment['content']
                                similarity = ratio(selection_text, comment_text)
                                
                                if similarity > 0.80:
                                    # Add generation info to the comment
                                    comment['generation_info'] = {
                                        'timestamp': generation['timestamp'],
                                        'suggestions': generation['suggestions'],
                                        'selection': selection
                                    }
                                    break
    
    return processed_data

