import streamlit as st
import numpy as np

def display_suggestions_metrics(data, display: bool = True):
    # Extract treatment data
    treatment_data = data.get('suggestions_5', {})
    
    # Basic metrics
    num_games = len(treatment_data)
    
    # Count actions and comments (each is a list of lists)
        # Count actions and comments (sum the lengths of inner lists)
    total_actions = sum(len(inner_list) for game in treatment_data.values() 
                       for inner_list in game['actions'])
    total_comments = sum(len(inner_list) for game in treatment_data.values() 
                        for inner_list in game['comments'])
    
    # Suggestions and selections analysis
    total_generations = 0
    total_selections = 0
    selection_types = {'positive': 0, 'neutral': 0, 'negative': 0}
    unique_users = set()
    users_with_suggestions = set()
    users_suggestion_usage = {}
    
    # Collect unique users from actions and comments
    for game in treatment_data.values():
        # Get users from actions
        for action_list in game['actions']:
            for action in action_list:
                if isinstance(action, dict) and 'user_id' in action:
                    unique_users.add(action['user_id'])
        
        # Get users from comments
        for comment_list in game['comments']:
            for comment in comment_list:
                if isinstance(comment, dict) and 'user_id' in comment:
                    unique_users.add(comment['user_id'])
    
    for game in treatment_data.values():
        # Iterate through rounds in suggestions
        for round_num, round_data in game.get('suggestions', {}).items():
            # Iterate through users in each round
            for user_id, user_data in round_data.items():
                unique_users.add(user_id)
                
                # Count generations (each suggestions list contains 3 suggestions)
                if 'suggestions' in user_data:
                    total_generations += len(user_data['suggestions'])
                
                if user_id not in users_suggestion_usage:
                    users_suggestion_usage[user_id] = 0
                users_suggestion_usage[user_id] += len(user_data['suggestions'])
                
                # Initialize selection details storage
                selection_details = {
                    'positive': [],
                    'neutral': [],
                    'negative': []
                }
                
                # Process selections
                if 'selected' in user_data:
                    selections = user_data['selected']
                    total_selections += len(selections)
                    users_with_suggestions.add(user_id)
                    
                    # Match selections with original suggestions
                    for selection in selections:
                        selected_text = selection[0]
                        selected_timestamp = selection[1]
                        found_match = False
                        
                        # Look only at this user's suggestions
                        for suggestion_set in user_data.get('suggestions', []):
                            if found_match:
                                break
                            if isinstance(suggestion_set[1], list):
                                suggestions_list = suggestion_set[1]
                                for idx, reply in enumerate(suggestions_list):
                                    if reply.get('reply') == selected_text:
                                        selection_info = {
                                            'text': selected_text,
                                            'timestamp': selected_timestamp,
                                            'user_id': user_id
                                        }
                                        if idx == 0:
                                            selection_types['positive'] += 1
                                            selection_details['positive'].append(selection_info)
                                        elif idx == 1:
                                            selection_types['neutral'] += 1
                                            selection_details['neutral'].append(selection_info)
                                        elif idx == 2:
                                            selection_types['negative'] += 1
                                            selection_details['negative'].append(selection_info)
                                        found_match = True
                                        break
                
    # Calculate user engagement rate
    user_engagement_rate = len(users_with_suggestions) / len(unique_users) if unique_users else 0
    
    # Add round-specific generation tracking
    generations_per_round = {0: 0, 1: 0, 2: 0}
    
    for game in treatment_data.values():
        # Iterate through rounds in suggestions
        for round_num, round_data in game.get('suggestions', {}).items():
            round_idx = int(round_num)
            # Iterate through users in each round
            for user_id, user_data in round_data.items():
                # Count generations (each suggestions list contains 3 suggestions)
                if 'suggestions' in user_data:
                    generations_per_round[round_idx] += len(user_data['suggestions'])
    
    if display:
        # Display metrics using Streamlit
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Games", num_games)
            st.metric("Total Actions", total_actions)
            st.metric("Total Comments", total_comments)
        with col2:
            st.metric("Total Generations", total_generations)
            st.metric("Total Selections", total_selections)
            st.metric("Fraction of generations selected", f"{total_selections / total_generations:.2%}")
        with col3:
            st.metric("Total Users", len(unique_users))
            st.metric("Users Using Suggestions", len(users_with_suggestions))
            st.metric("Fraction of users using suggestions", f"{user_engagement_rate:.2%}")
            
        # Display selection types
        st.subheader("Selection Types")
        type_col1, type_col2, type_col3 = st.columns(3)
        with type_col1:
            st.metric("🟢 Positive Selections", selection_types['positive'])
        with type_col2:
            st.metric("🟡 Neutral Selections", selection_types['neutral'])
        with type_col3:
            st.metric("🔴 Negative Selections", selection_types['negative'])
        
        # Display generations by round
        st.subheader("Generations by Round")
        gen_col1, gen_col2, gen_col3 = st.columns(3)
        with gen_col1:
            st.metric("Round 1", generations_per_round[0])
        with gen_col2:
            st.metric("Round 2", generations_per_round[1])
        with gen_col3:
            st.metric("Round 3", generations_per_round[2])
    
    # Initialize detailed_data dictionary
    detailed_data = {}  # game_id -> round_id -> user_id -> data
    
    # Process suggestions and selections
    for game_id, game in treatment_data.items():
        detailed_data[game_id] = {}
        for round_id, round_data in game.get('suggestions', {}).items():
            detailed_data[game_id][round_id] = {}
            for user_id, user_data in round_data.items():
                detailed_data[game_id][round_id][user_id] = {
                    'generations': []
                }
                
                # Store generations with their selections
                if 'suggestions' in user_data:
                    for suggestion_set in user_data['suggestions']:
                        if isinstance(suggestion_set[1], list):
                            generation_info = {
                                'timestamp': suggestion_set[2],
                                'suggestions': [
                                    {'type': 'positive', 'text': suggestion_set[1][0]['reply']},
                                    {'type': 'neutral', 'text': suggestion_set[1][1]['reply']},
                                    {'type': 'negative', 'text': suggestion_set[1][2]['reply']}
                                ],
                                'selections': []  # Will store selections for this generation
                            }
                            
                            # Match selections to this generation
                            if 'selected' in user_data:
                                for selection in user_data['selected']:
                                    selected_text = selection[0]
                                    selected_timestamp = selection[1]
                                    
                                    # Check if this selection matches any suggestion in this generation
                                    for idx, reply in enumerate(suggestion_set[1]):
                                        if reply.get('reply') == selected_text:
                                            selection_type = 'positive' if idx == 0 else 'neutral' if idx == 1 else 'negative'
                                            selection_info = {
                                                'text': selected_text,
                                                'timestamp': selected_timestamp,
                                                'type': selection_type
                                            }
                                            generation_info['selections'].append(selection_info)
                            
                            detailed_data[game_id][round_id][user_id]['generations'].append(generation_info)
    
    # Add detailed_data to the returned metrics
    return {**detailed_data, 'generations_per_round': generations_per_round, 'users_suggestion_usage': users_suggestion_usage}

def display_chat_metrics(data, display: bool = True):
    # Extract treatment data
    treatment_data = data.get('chat_5', {})
    
    # Basic metrics
    num_games = len(treatment_data)
    
    # Count actions and comments
    total_actions = sum(len(inner_list) for game in treatment_data.values() 
                       for inner_list in game['actions'])
    total_comments = sum(len(inner_list) for game in treatment_data.values() 
                        for inner_list in game['comments'])
    
    # Initialize user tracking
    unique_users = set()
    users_with_chat = set()
    total_user_prompts = 0
    prompts_per_user = []  # Will store number of prompts for each user
    
    # Add round-specific chat tracking
    chat_usage_per_round = {0: 0, 1: 0, 2: 0}  # Track chat prompts by round
    
    # Collect unique users from actions and comments
    for game in treatment_data.values():
        # Get users from actions
        for action_list in game['actions']:
            for action in action_list:
                if isinstance(action, dict) and 'user_id' in action:
                    unique_users.add(action['user_id'])
        
        # Get users from comments
        for comment_list in game['comments']:
            for comment in comment_list:
                if isinstance(comment, dict) and 'user_id' in comment:
                    unique_users.add(comment['user_id'])
    
    # Process chat data
    chat_prompts = []
    chat_prompts_per_user = {}

    for game in treatment_data.values():
        if 'chat' in game:
            # For each round
            for round_num, round_data in game['chat'].items():
                round_idx = int(round_num)
                # For each user in the round
                for user_id, messages in round_data.items():
                    unique_users.add(user_id)
                    user_prompt_count = 0
                    
                    # Count user messages
                    for message in messages:
                        if message[0].startswith('user:'):
                            chat_prompts.append(message)
                            user_prompt_count += 1
                            total_user_prompts += 1
                            chat_usage_per_round[round_idx] += 1
                    
                    if user_prompt_count > 0:
                        users_with_chat.add(user_id)
                        prompts_per_user.append(user_prompt_count)

                    if user_id not in chat_prompts_per_user:
                        chat_prompts_per_user[user_id] = 0
                    chat_prompts_per_user[user_id] += user_prompt_count

    
    
    # Calculate statistics
    # Only include users who actually used the chat
    active_chat_users = [prompts for prompts in prompts_per_user if prompts > 0]
    avg_prompts_per_user = np.mean(active_chat_users) if active_chat_users else 0
    std_prompts_per_user = np.std(prompts_per_user) if prompts_per_user else 0
    user_engagement_rate = len(users_with_chat) / len(unique_users) if unique_users else 0
    
    if display:
        # Display metrics using Streamlit
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Games", num_games)
            st.metric("Total Actions", total_actions)
            st.metric("Total Comments", total_comments)
        with col2:
            st.metric("Total Users", len(unique_users))
            st.metric("Users Using Chat", len(users_with_chat))
            st.metric("Chat Usage Rate", f"{user_engagement_rate:.2%}")
        with col3:
            st.metric("Total Chat Prompts", total_user_prompts)
            st.metric("Avg Prompts per User", f"{avg_prompts_per_user:.2f}")
            st.metric("Std Prompts per User", f"{std_prompts_per_user:.2f}")
            
        # Display chat usage by round
        st.subheader("Chat Usage by Round")
        round_col1, round_col2, round_col3 = st.columns(3)
        with round_col1:
            st.metric("Round 1", chat_usage_per_round[0])
        with round_col2:
            st.metric("Round 2", chat_usage_per_round[1])
        with round_col3:
            st.metric("Round 3", chat_usage_per_round[2])
    
    return {
        'num_games': num_games,
        'total_actions': total_actions,
        'total_comments': total_comments,
        'total_users': len(unique_users),
        'users_with_chat': len(users_with_chat),
        'total_user_prompts': total_user_prompts,
        'avg_prompts_per_user': avg_prompts_per_user,
        'std_prompts_per_user': std_prompts_per_user,
        'chat_usage_per_round': chat_usage_per_round,
        'chat_prompts': chat_prompts,
        'chat_prompts_per_user': chat_prompts_per_user
    }


def display_feedback_metrics(data, display: bool = True):
    # Extract treatment data
    treatment_data = data.get('feedback_5', {})
    
    # Basic metrics
    num_games = len(treatment_data)
    
    # Count actions and comments
    total_actions = sum(len(inner_list) for game in treatment_data.values() 
                       for inner_list in game['actions'])
    total_comments = sum(len(inner_list) for game in treatment_data.values() 
                        for inner_list in game['comments'])
    
    # Initialize tracking
    unique_users = set()
    users_with_feedback = set()
    total_feedback_requests = 0
    feedback_requests_per_user = []  # Will store number of feedback requests per user
    feedback_per_round = {0: 0, 1: 0, 2: 0}  # Track feedback requests by round
    feedbacks_per_user = {}

    
    # Collect unique users from actions and comments
    for game in treatment_data.values():
        # Get users from actions
        for action_list in game['actions']:
            for action in action_list:
                if isinstance(action, dict) and 'user_id' in action:
                    unique_users.add(action['user_id'])
        
        # Get users from comments
        for comment_list in game['comments']:
            for comment in comment_list:
                if isinstance(comment, dict) and 'user_id' in comment:
                    unique_users.add(comment['user_id'])
    
    # Process feedback data
    for game in treatment_data.values():
        if 'feedback' in game:
            for round_num, round_data in game['feedback'].items():
                for user_id, content_data in round_data.items():
                    unique_users.add(user_id)
                    user_feedback_count = 0
                    
                    for content_id, feedback_list in content_data.items():
                        user_feedback_count += len(feedback_list)
                        total_feedback_requests += len(feedback_list)
                        feedback_per_round[int(round_num)] += len(feedback_list)
                    
                    if user_feedback_count > 0:
                        users_with_feedback.add(user_id)
                        feedback_requests_per_user.append(user_feedback_count)

                    if user_id not in feedbacks_per_user:
                        feedbacks_per_user[user_id] = 0
                    feedbacks_per_user[user_id] += user_feedback_count
    
    # Calculate statistics
    active_feedback_users = [requests for requests in feedback_requests_per_user if requests > 0]
    avg_feedback_per_user = np.mean(active_feedback_users) if active_feedback_users else 0
    avg_feedback_per_round = total_feedback_requests / 3 if total_feedback_requests > 0 else 0  # Average across all 3 rounds
    avg_feedback_per_user_per_round = avg_feedback_per_user / 3  # Since there are 3 rounds
    user_engagement_rate = len(users_with_feedback) / len(unique_users) if unique_users else 0
    
    if display:
        # Display metrics using Streamlit
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Games", num_games)
            st.metric("Total Actions", total_actions)
            st.metric("Total Comments", total_comments)
        with col2:
            st.metric("Total Users", len(unique_users))
            st.metric("Users Using Feedback", len(users_with_feedback))
            st.metric("Feedback Usage Rate", f"{user_engagement_rate:.2%}")
        with col3:
            st.metric("Total Feedback Requests", total_feedback_requests)
            st.metric("Avg Feedback per User", f"{avg_feedback_per_user:.2f}")
            st.metric("Avg Feedback per Round", f"{avg_feedback_per_user_per_round:.2f}")

        # Display feedback requests by round
        st.subheader("Feedback Requests by Round")
        round_col1, round_col2, round_col3 = st.columns(3)
        with round_col1:
            st.metric("Round 1", feedback_per_round[0])
        with round_col2:
            st.metric("Round 2", feedback_per_round[1])
        with round_col3:
            st.metric("Round 3", feedback_per_round[2])
    
    return {
        'num_games': num_games,
        'total_actions': total_actions,
        'total_comments': total_comments,
        'total_users': len(unique_users),
        'users_with_feedback': len(users_with_feedback),
        'user_engagement_rate': user_engagement_rate,
        'total_feedback_requests': total_feedback_requests,
        'avg_feedback_per_user': avg_feedback_per_user,
        'avg_feedback_per_round': avg_feedback_per_round,
        'feedback_per_round': feedback_per_round,
        'feedbacks_per_user': feedbacks_per_user
    }


def calculate_feedback_to_comment_rate(game):
    """Calculate what fraction of feedback requests led to comments per round."""
    feedback_to_comment = {0: {'requests': 0, 'comments_made': 0},
                         1: {'requests': 0, 'comments_made': 0},
                         2: {'requests': 0, 'comments_made': 0}}
    
    if 'feedback' not in game:
        return feedback_to_comment

    # For each round
    for round_num, round_data in game['feedback'].items():
        round_idx = int(round_num)
        
        # Get comments for this round
        round_comments = game['comments'][round_idx] if round_idx < len(game['comments']) else []
        
        # For each user's feedback in this round
        for user_id, content_data in round_data.items():
            # For each parent comment they requested feedback for
            for parent_id, feedback_list in content_data.items():
                if feedback_list:  # If they requested feedback
                    feedback_to_comment[round_idx]['requests'] += 1
                    
                    # Check if user made a comment with this parent_id
                    user_made_comment = any(
                        comment.get('user_id') == user_id and 
                        str(comment.get('parent_content_id')) == str(parent_id)
                        for comment in round_comments
                    )
                    
                    if user_made_comment:
                        feedback_to_comment[round_idx]['comments_made'] += 1
    
    return feedback_to_comment

def display_feedback_metrics_by_topic(data, display: bool = True):
    # Extract treatment data
    treatment_data = data.get('feedback_5', {})
    
    # Initialize per-topic tracking
    topics_data = {
        'oats': {'actions': 0, 'comments': 0, 'unique_users': set(), 
                'users_with_feedback': set(), 'feedback_requests': 0,
                'feedback_comment_pairs': []},  # Add new field for storing pairs
        'cats': {'actions': 0, 'comments': 0, 'unique_users': set(), 
                'users_with_feedback': set(), 'feedback_requests': 0,
                'feedback_comment_pairs': []},
        'politics': {'actions': 0, 'comments': 0, 'unique_users': set(), 
                    'users_with_feedback': set(), 'feedback_requests': 0,
                    'feedback_comment_pairs': []}
    }
    
    # Initialize feedback-to-comment tracking in topics_data
    for topic_data in topics_data.values():
        topic_data['feedback_to_comment'] = {
            'feedback_requests': 0,  # Total number of feedback requests
            'followed_by_comment': 0  # Number of requests that led to comments
        }
    
    # Process each game
    for game in treatment_data.values():
        # Get topic order mapping
        topic_order = game.get('topic_order', [])
        if not topic_order:
            continue
            
        round_to_topic = {i: topic for i, topic in enumerate(topic_order)}
        
        # Process actions and comments by round/topic
        for round_num in range(3):
            topic = round_to_topic[round_num]
            
            # Count actions
            if round_num < len(game['actions']):
                topics_data[topic]['actions'] += len(game['actions'][round_num])
                # Add users from actions
                for action in game['actions'][round_num]:
                    if isinstance(action, dict) and 'user_id' in action:
                        topics_data[topic]['unique_users'].add(action['user_id'])
            
            # Count comments
            if round_num < len(game['comments']):
                topics_data[topic]['comments'] += len(game['comments'][round_num])
                # Add users from comments
                for comment in game['comments'][round_num]:
                    if isinstance(comment, dict) and 'user_id' in comment:
                        topics_data[topic]['unique_users'].add(comment['user_id'])
        
        # Process feedback data and corresponding comments
        if 'feedback' in game:
            for round_num, round_data in game['feedback'].items():
                round_idx = int(round_num)
                topic = round_to_topic[round_idx]
                
                # Get comments for this round
                round_comments = game['comments'][round_idx] if round_idx < len(game['comments']) else []
                
                # Process each user's feedback requests
                for user_id, content_data in round_data.items():
                    topics_data[topic]['unique_users'].add(user_id)
                    feedback_count = sum(len(feedback_list) for feedback_list in content_data.values())
                    
                    if feedback_count > 0:
                        topics_data[topic]['users_with_feedback'].add(user_id)
                        topics_data[topic]['feedback_requests'] += feedback_count
                    
                    # Track feedback-to-comment conversion
                    for parent_id, feedback_list in content_data.items():
                        if feedback_list:  # If they requested feedback
                            topics_data[topic]['feedback_to_comment']['feedback_requests'] += 1
                            
                            # Check if user made a comment with this parent_id
                            user_made_comment = any(
                                comment.get('user_id') == user_id and 
                                str(comment.get('parent_content_id')) == str(parent_id)
                                for comment in round_comments
                            )
                            
                            if user_made_comment:
                                topics_data[topic]['feedback_to_comment']['followed_by_comment'] += 1
                    
                    # Track feedback-to-comment pairs
                    for parent_id, feedback_list in content_data.items():
                        if feedback_list:  # If they requested feedback
                            try:
                                # Get the original content they requested feedback on
                                original_content = feedback_list[0][0]  # First element is the original text
                                
                                # Find the corresponding comment they made (if any)
                                final_comment = None
                                for comment in round_comments:
                                    if (comment.get('user_id') == user_id and 
                                        str(comment.get('parent_content_id')) == str(parent_id)):
                                        final_comment = comment.get('content')  # Changed from 'text' to 'content'
                                        break
                                
                                if final_comment:  # Only store if they made a comment
                                    topics_data[topic]['feedback_comment_pairs'].append(
                                        (original_content, final_comment)
                                    )
                            except Exception as e:
                                print(f"Error processing feedback: {e}")
                                continue
    
    if display:
        for topic in topics_data:
            st.subheader(f"Topic: {topic.capitalize()}")
            
            # Calculate metrics for this topic
            total_users = len(topics_data[topic]['unique_users'])
            users_with_feedback = len(topics_data[topic]['users_with_feedback'])
            feedback_requests = topics_data[topic]['feedback_requests']
            avg_per_user = feedback_requests / users_with_feedback if users_with_feedback > 0 else 0
            usage_rate = users_with_feedback / total_users if total_users > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Actions", topics_data[topic]['actions'])
                st.metric("Total Comments", topics_data[topic]['comments'])
            with col2:
                st.metric("Total Users", total_users)
                st.metric("Users Using Feedback", users_with_feedback)
                st.metric("Feedback Usage Rate", f"{usage_rate:.2%}")
            with col3:
                st.metric("Total Feedback Requests", feedback_requests)
                st.metric("Avg Feedback per User", f"{avg_per_user:.2f}")

            # Add feedback-to-comment rate metric
            feedback_requests = topics_data[topic]['feedback_to_comment']['feedback_requests']
            comments_made = topics_data[topic]['feedback_to_comment']['followed_by_comment']
            comment_rate = comments_made / feedback_requests if feedback_requests > 0 else 0
            
            with col2:  # Add to existing column
                st.metric(
                    "Feedback-to-Comment Rate", 
                    f"{comment_rate:.2%}",
                    delta=f"{comments_made} comments made from {feedback_requests} feedback requests",
                    delta_color="off",
                    help=f"The number of feedback requests is smaller than 'Total Feedback Requests' because we here don't account for multiple feedback requests per comment. This is a 'how often did a feedback lead to a comment' metric."
                )

            st.markdown("---")
        
    return topics_data


def display_conversation_metrics(data, display: bool = True):
    # Extract treatment data
    treatment_data = data.get('conversation_5', {})
    
    # Basic metrics
    num_games = len(treatment_data)
    
    # Count actions and comments
    total_actions = sum(len(inner_list) for game in treatment_data.values() 
                       for inner_list in game['actions'])
    total_comments = sum(len(inner_list) for game in treatment_data.values() 
                        for inner_list in game['comments'])
    
    # Initialize tracking
    unique_users = set()
    users_with_conversation = set()
    total_conversation_uses = 0
    conversation_uses_per_user = {}
    
    # Initialize round tracking
    starters_per_round = {0: 0, 1: 0, 2: 0}  # Track usage by round
    
    # Collect unique users from actions and comments
    for game in treatment_data.values():
        # Get users from actions
        for action_list in game['actions']:
            for action in action_list:
                if isinstance(action, dict) and 'user_id' in action:
                    unique_users.add(action['user_id'])
        
        # Get users from comments
        for comment_list in game['comments']:
            for comment in comment_list:
                if isinstance(comment, dict) and 'user_id' in comment:
                    unique_users.add(comment['user_id'])
    
    # Process conversation starter data
    for game in treatment_data.values():
        if 'conversation_starter' in game:
            # For each round
            for round_num, round_data in game['conversation_starter'].items():
                round_idx = int(round_num)
                # For each user in the round
                for user_id, usage_list in round_data.items():
                    unique_users.add(user_id)
                    if usage_list:  # If user has any usage
                        users_with_conversation.add(user_id)
                        total_conversation_uses += len(usage_list)
                        starters_per_round[round_idx] += len(usage_list)
                    
                    if user_id not in conversation_uses_per_user:
                        conversation_uses_per_user[user_id] = 0
                    conversation_uses_per_user[user_id] += len(usage_list)
    
    # Calculate statistics
    user_engagement_rate = len(users_with_conversation) / len(unique_users) if unique_users else 0
    avg_usage_per_user_per_round = (total_conversation_uses / len(users_with_conversation) / 3) if users_with_conversation else 0
    
    if display:
        # Display metrics using Streamlit
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Games", num_games)
            st.metric("Total Actions", total_actions)
            st.metric("Total Comments", total_comments)
        with col2:
            st.metric("Total Users", len(unique_users))
            st.metric("Users Using Conversation Starter", len(users_with_conversation))
            st.metric("Usage Rate", f"{user_engagement_rate:.2%}")
        with col3:
            st.metric("Total Conversation Uses", total_conversation_uses)
            st.metric("Avg Uses per User per Round", f"{avg_usage_per_user_per_round:.2f}")
            
        # Display conversation starter usage by round
        st.subheader("Conversation Starter Usage by Round")
        round_col1, round_col2, round_col3 = st.columns(3)
        with round_col1:
            st.metric("Round 1", starters_per_round[0])
        with round_col2:
            st.metric("Round 2", starters_per_round[1])
        with round_col3:
            st.metric("Round 3", starters_per_round[2])
    
    return {
        'num_games': num_games,
        'total_actions': total_actions,
        'total_comments': total_comments,
        'total_users': len(unique_users),
        'users_with_conversation': len(users_with_conversation),
        'total_conversation_uses': total_conversation_uses,
        'avg_usage_per_user_per_round': avg_usage_per_user_per_round,
        'user_engagement_rate': user_engagement_rate,
        'starters_per_round': starters_per_round,
        'conversation_uses_per_user': conversation_uses_per_user
    }