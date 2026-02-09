import pandas as pd
import json
import streamlit as st


# Helper function to check if timestamp is within bucket (with buffer)
def is_in_bucket(timestamp, start_time, buffer_seconds=30):
    """Helper function to check if timestamp is within bucket (with buffer)"""
    buffer = pd.Timedelta(seconds=buffer_seconds)
    start_with_buffer = start_time - buffer
    end_with_buffer = start_time + pd.Timedelta(minutes=10) + buffer
    return start_with_buffer <= timestamp <= end_with_buffer

# Helper function to process items and check bucket validity
def process_items_for_bucket(items, start_time, item_type, end_time=None):
    valid_items = []
    invalid_items = []
    
    for item in items:
        timestamp = pd.to_datetime(item['timestamp'])
        if end_time is None or (start_time <= timestamp <= end_time):
            valid_items.append(item)
        else:
            invalid_items.append({
                'timestamp': timestamp,
                'expected_bucket': start_time,
                'item': item,
                'reason': 'before_start' if timestamp < start_time else 'after_end',
                'time_diff': (timestamp - start_time).total_seconds() / 60  # difference in minutes
            })
    
    return valid_items, invalid_items

def parse_content_order(content_order_str):
    """
    Parse the content order string into a mapping of topic names to their positions.
    
    Args:
        content_order_str (str): JSON string containing content order information
        
    Returns:
        dict: Mapping of topic names to their positions
    """
    if pd.isna(content_order_str):
        return {}
        
    try:
        # Remove outer quotes and replace double quotes
        if isinstance(content_order_str, str):
            if content_order_str.startswith('"') and content_order_str.endswith('"'):
                content_order_str = content_order_str[1:-1]
            content_order_str = content_order_str.replace('""', '"')
        
        content_order = json.loads(content_order_str)
        return {item['name']: item['position'] for item in content_order}
    except Exception as e:
        st.warning(f"Error parsing content order: {str(e)}")
        return {}

def process_topic_order(row):
    """
    Process topic order using content order and create topic mapping.
    
    Args:
        row (pd.Series): DataFrame row containing contentOrder
        
    Returns:
        dict: Mapping of topic names to their position
    """
    # Parse the content order
    if isinstance(row['contentOrder'], str):
        content_order_str = row['contentOrder']
        if content_order_str.startswith('"') and content_order_str.endswith('"'):
            content_order_str = content_order_str[1:-1]
        content_order_str = content_order_str.replace('""', '"')
        
        try:
            content_order = json.loads(content_order_str)
            # Create mapping using the position directly
            return {item['name']: item['position'] for item in content_order}
        except json.JSONDecodeError as e:
            st.warning(f"Error parsing content order: {str(e)}")
            return {}
    
    return {}

# @st.cache_data
def process_engagement_data(df, control_group='baseline_5', combine_treatment=False):
    """
    Process engagement data to organize actions and comments into correct time buckets for each ID,
    comparing control group vs selected treatment(s).
    
    Args:
        df (pd.DataFrame): DataFrame containing multiple rows with actions and comments
        control_group (str): Name of the control group (default: 'baseline_5')
        combine_treatment (bool): If True, all non-control treatment groups will be combined under
                                  a single 'treatment' key. If False, treatment groups are kept separate.
        
    Returns:
        tuple: (
            dict: Processed data organized by group and ID with correct bucket assignments,
            list: Invalid items that couldn't be properly bucketed
        )
    """

    # Initialize processed_data with empty dictionaries
    processed_data = {}
    all_invalid_items = []
    
    # Process each row (game instance) separately
    for _, row in df.iterrows():
        game_id = row['id']
        # Determine group key: use 'control' if it's the control group; otherwise,
        # combine all treatments under 'treatment' if combine_treatment is True, or use the original treatment name.
        if row['treatmentName'] == control_group:
            group_key = 'control'
        else:
            group_key = 'treatment' if combine_treatment else row['treatmentName']
        
        # Initialize group in processed_data if not already present
        if group_key not in processed_data:
            processed_data[group_key] = {}
            
        # Parse content order to get fileId -> position mapping
        content_order = json.loads(row['contentOrder'])
        content_mapping = {
            item['fileId']: {
                'position': item['position'],
                'name': item['name']
            }
            for item in content_order
        }
        
        # Initialize buckets for this game
        correct_bucket_actions = [[] for _ in range(3)]
        correct_bucket_comments = [[] for _ in range(3)]
        game_invalid_items = []
        
        # Process each fileId's actions and comments
        for file_id in range(3):
            position = content_mapping[file_id]['position']
            round_start = row[f'round_{position}_startTime']
            round_end = round_start + pd.Timedelta(minutes=10)
            
            # Get actions and comments for this fileId
            actions = row[f'actions-{file_id}']
            comments = row[f'comments-{file_id}']
            
            # Process actions
            if actions:
                valid_actions, invalid_actions = process_items_for_bucket(
                    actions,
                    round_start,
                    'action',
                    end_time=round_end
                )
                correct_bucket_actions[position] = valid_actions
                game_invalid_items.extend(invalid_actions)
            
            # Process comments
            if comments:
                valid_comments, invalid_comments = process_items_for_bucket(
                    comments,
                    round_start,
                    'comment',
                    end_time=round_end
                )
                correct_bucket_comments[position] = valid_comments
                game_invalid_items.extend(invalid_comments)
        
        # Store processed data for this game
        processed_data[group_key][game_id] = {
            'actions': correct_bucket_actions,
            'comments': correct_bucket_comments,
            'round_starts': [row[f'round_{i}_startTime'] for i in range(3)],
            'topic_order': [item['name'] for item in sorted(content_order, key=lambda x: x['position'])],
            'treatmentName': row['treatmentName']
        }
        
        # Add game ID and treatment info to invalid items
        for item in game_invalid_items:
            item['game_id'] = game_id
            item['treatmentName'] = row['treatmentName']
        all_invalid_items.extend(game_invalid_items)
    
    return processed_data, all_invalid_items

def filter_treatment_users_by_treatment_usage(processed_data: dict):
    with open('data/combining/combined/users_using_treatment.json', 'r') as f:
        users_using_treatments = json.load(f)
    
    # Create a copy of the input data to avoid modifying the original
    filtered_data = processed_data.copy()
    
    # Process each treatment group
    for treatment in filtered_data:
        # Skip baseline_5
        if treatment == 'control':
            continue
            
        valid_users = set(users_using_treatments.get(treatment, []))
        
        # Process each game in the treatment
        for game_id, game_data in filtered_data[treatment].items():
            # Filter actions in place
            for round_idx, round_actions in enumerate(game_data['actions']):
                game_data['actions'][round_idx] = [
                    action for action in round_actions 
                    if action['user_id'] in valid_users
                ]
            
            # Filter comments in place
            for round_idx, round_comments in enumerate(game_data['comments']):
                game_data['comments'][round_idx] = [
                    comment for comment in round_comments 
                    if comment['user_id'] in valid_users
                ]
    
    return filtered_data, users_using_treatments
