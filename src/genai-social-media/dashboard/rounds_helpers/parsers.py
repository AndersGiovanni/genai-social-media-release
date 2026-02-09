import pandas as pd
import json
import streamlit as st

def parse_timestamp(timestamp):
    try:
        # Try parsing as ISO format
        return pd.to_datetime(timestamp, format='%Y-%m-%dT%H:%M:%S.%fZ')
    except ValueError:
        try:
            # Try parsing as milliseconds since epoch
            return pd.to_datetime(int(timestamp), unit='ms')
        except ValueError:
            # If both fail, return NaT (Not a Time)
            return pd.NaT


# Parse JSON strings in action and comment columns with error handling
def parse_json_safely(x):
    if pd.isna(x):
        return []
    try:
        # Skip if it's a timestamp (contains 'T' and 'Z' and no '{')
        if isinstance(x, str) and 'T' in x and 'Z' in x and '{' not in x:
            return []
            
        # If it's already a list or dict, return as is
        if isinstance(x, (list, dict)):
            return x
            
        # Handle the string parsing
        if isinstance(x, str):
            # Remove outer quotes if they exist
            if x.startswith('"') and x.endswith('"'):
                x = x[1:-1]
            
            # Handle potential JSON string escaping issues
            try:
                # First attempt: Try direct JSON loads
                return json.loads(x)
            except json.JSONDecodeError:
                # Second attempt: Handle double-encoded JSON
                try:
                    # Remove any BOM characters
                    x = x.encode('utf-8').decode('utf-8-sig')
                    
                    # Handle cases where the JSON might be double-encoded
                    while x.startswith('"[') and x.endswith(']"'):
                        x = x[1:-1]
                    
                    # Replace problematic escape sequences
                    x = x.replace('\\"', '"')  # Replace escaped quotes
                    x = x.replace('\\\\', '\\')  # Replace double backslashes
                    x = x.replace('\r', '').replace('\n', '')  # Remove newlines
                    
                    # Ensure the string starts with [ and ends with ]
                    if not x.startswith('['):
                        x = '[' + x
                    if not x.endswith(']'):
                        x = x + ']'
                    
                    return json.loads(x)
                except Exception as e:
                    with open('debug_error_details.txt', 'w', encoding='utf-8') as f:
                        f.write(f"Error in second attempt: {str(e)}\n")
                        f.write(f"Input string: {x[:200]}...\n")
                    raise
                
        return []
    except Exception as e:
        st.warning(f"JSON parsing error: {str(e)}\nValue: {x[:100]}...")
        return []
    

def standardize_timestamps(data_list):
    if not isinstance(data_list, list):
        return []
    for item in data_list:
        if isinstance(item, dict) and 'timestamp' in item:
            item['timestamp'] = pd.to_datetime(item['timestamp'], utc=True)
    return data_list