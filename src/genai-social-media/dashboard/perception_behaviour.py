import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from players import load_data
from rounds_game import load_data as load_data_rounds, process_game_data
from engagement_helpers.parsers import process_engagement_data
from treatments.parsers import process_player_rounds
from treatments.treatment_info import (
    display_chat_metrics, 
    display_suggestions_metrics, 
    display_feedback_metrics, 
    display_conversation_metrics
)

#########################################################################
#### Making figure 3. freq of AI usage vs. user ratings of others ####
#########################################################################

df, df_ate = load_data()

# Storing the ratings of comments for all users in each treatment
comments_ratings = {
    "baseline_5": {},
    "chat_5": {},
    "suggestions_5": {},
    "conversation_5": {},
    "feedback_5": {}
}

# Storing all the ratings received in each treatment
received_user_ratings = {
    "baseline_5": {},
    "chat_5": {},
    "suggestions_5": {},
    "conversation_5": {},
    "feedback_5": {}
}

for _, row in df.iterrows():
    if isinstance(row['exitSurvey'], dict) and 'userRating' in row['exitSurvey']:
        user_id = row['id']
        ratings = row['exitSurvey']['userRating'].get('commentRatings', {})
        user_ratings = row['exitSurvey']['userRating'].get('participantRatings', {})
        treatment_name = row['treatmentName']
        rating_values = []
        for comment_id, rating_data in ratings.items():
            if 'valueComment' in rating_data:
                try:
                    rating = float(rating_data['valueComment'])
                    rating_values.append(rating)
                except (ValueError, TypeError):
                    continue
        mean_rating = np.mean(rating_values)
        if not np.isnan(mean_rating):
            comments_ratings[treatment_name][user_id] = float(mean_rating)

        for user_id, ratings in user_ratings.items():
            if user_id in received_user_ratings[treatment_name]:
                received_user_ratings[treatment_name][user_id]['engagement'].append(int(ratings['engagement']))
                received_user_ratings[treatment_name][user_id]['politeness'].append(int(ratings['politeness']))
                received_user_ratings[treatment_name][user_id]['positivity'].append(int(ratings['positivity']))
            else:
                received_user_ratings[treatment_name][user_id] = {
                    'engagement': [int(ratings['engagement'])],
                    'politeness': [int(ratings['politeness'])],
                    'positivity': [int(ratings['positivity'])]
                }

# Iterate through all treatments and users in received_user_ratings and calculate the mean of the ratings
for treatment_name, users in received_user_ratings.items():
    for user_id, ratings in users.items():
        received_user_ratings[treatment_name][user_id]['engagement'] = float(np.mean(ratings['engagement']))
        received_user_ratings[treatment_name][user_id]['politeness'] = float(np.mean(ratings['politeness']))
        received_user_ratings[treatment_name][user_id]['positivity'] = float(np.mean(ratings['positivity']))



##################################
#### Getting freq of AI usage ####
##################################

df, df_round, df_batch = load_data_rounds()
# st.write(df)
df_player_round = pd.read_csv('data/combining/combined/playerRound.csv')
df, action_columns, comment_columns = process_game_data(df, df_round, df_batch)

# Process data for both control and treatment groups
processed_data, invalid_items = process_engagement_data(
    df,
    control_group='baseline_5'
)

player_round_data = process_player_rounds(processed_data, df_player_round)

chat_metrics = display_chat_metrics(player_round_data)['chat_prompts_per_user']

suggestions_metrics = display_suggestions_metrics(player_round_data)['users_suggestion_usage']

feedback_metrics = display_feedback_metrics(player_round_data)['feedbacks_per_user']

conversation_metrics = display_conversation_metrics(player_round_data)['conversation_uses_per_user']

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Prepare data for plotting by matching user IDs
chat_users = []
suggestions_users = []
feedback_users = []
conversation_users = []
chat_ai_freq = []
suggestions_ai_freq = []
feedback_ai_freq = []
conversation_ai_freq = []
avg_comment_ratings_chat = []
avg_engagement_ratings_chat = []
avg_comment_ratings_suggestions = []
avg_engagement_ratings_suggestions = []
avg_comment_ratings_feedback = []
avg_engagement_ratings_feedback = []
avg_comment_ratings_conversation = []
avg_engagement_ratings_conversation = []

# Types of engagement to measure: engagement, politeness, positivity
USER_RATING_METRIC = 'positivity'
# Process chat metrics
for user_id, ai_usage in chat_metrics.items():
    if user_id in comments_ratings['chat_5'] and user_id in received_user_ratings['chat_5']:
        if ai_usage > 30:  # Keep existing outlier filter
            continue
        chat_users.append(user_id)
        chat_ai_freq.append(ai_usage)
        avg_comment_ratings_chat.append(comments_ratings['chat_5'][user_id])
        avg_engagement_ratings_chat.append(received_user_ratings['chat_5'][user_id][USER_RATING_METRIC])

# Process suggestions metrics
for user_id, ai_usage in suggestions_metrics.items():
    if user_id in comments_ratings['suggestions_5'] and user_id in received_user_ratings['suggestions_5']:
        if ai_usage > 60:  # Keep existing outlier filter
            continue
        suggestions_users.append(user_id)
        suggestions_ai_freq.append(ai_usage)
        avg_comment_ratings_suggestions.append(comments_ratings['suggestions_5'][user_id])
        avg_engagement_ratings_suggestions.append(received_user_ratings['suggestions_5'][user_id][USER_RATING_METRIC])

# Process feedback metrics
for user_id, ai_usage in feedback_metrics.items():
    if user_id in comments_ratings['feedback_5'] and user_id in received_user_ratings['feedback_5']:
        feedback_users.append(user_id)
        feedback_ai_freq.append(ai_usage)
        avg_comment_ratings_feedback.append(comments_ratings['feedback_5'][user_id])
        avg_engagement_ratings_feedback.append(received_user_ratings['feedback_5'][user_id][USER_RATING_METRIC])

# Process conversation metrics
for user_id, ai_usage in conversation_metrics.items():
    if user_id in comments_ratings['conversation_5'] and user_id in received_user_ratings['conversation_5']:
        conversation_users.append(user_id)
        conversation_ai_freq.append(ai_usage)
        avg_comment_ratings_conversation.append(comments_ratings['conversation_5'][user_id])
        avg_engagement_ratings_conversation.append(received_user_ratings['conversation_5'][user_id][USER_RATING_METRIC])

# Create first figure with three subplots for different rating types
fig1, (ax_eng, ax_pol, ax_pos) = plt.subplots(3, 1, figsize=(12, 15))
rating_types = ['engagement', 'politeness', 'positivity']
axes = [ax_eng, ax_pol, ax_pos]
titles = [' Engagement Ratings', 
         'Politeness Ratings',
         'Positivity Ratings']

# Set style and colors for better readability
plt.style.use('bmh')
# sns.color_palette("colorblind")[:4]
colors = {
    'Chat': '#0173B2',        # blue
    'Suggestions': '#D55E00',  # red
    'Feedback': '#029E73',     # green
    'Conversation': '#DE8F05',  # orange
}

# Add a bold main title
fig1.suptitle('Impact of AI Usage on Received User Ratings', 
              fontsize=16, 
              y=0.95, 
              weight='bold')

# Plot each rating type
for idx, (ax, rating_type, title) in enumerate(zip(axes, rating_types, titles)):
    for data_x, treatment, label, color in [
        (chat_ai_freq, 'chat_5', 'Chat', colors['Chat']),
        (conversation_ai_freq, 'conversation_5', 'Conversation', colors['Conversation']),
        (feedback_ai_freq, 'feedback_5', 'Feedback', colors['Feedback']),
        (suggestions_ai_freq, 'suggestions_5', 'Suggestions', colors['Suggestions']),
    ]:
        # Get ratings for this specific metric
        ratings = []
        for i, freq in enumerate(data_x):
            user_id = locals()[f"{label.lower()}_users"][i]
            if user_id in received_user_ratings[treatment]:
                ratings.append(received_user_ratings[treatment][user_id][rating_type])
        
        if len(data_x) > 0 and len(ratings) > 0:
            # Add jitter to x values
            jitter = np.random.normal(0, 0.1, len(data_x))
            jittered_x = np.array(data_x) + jitter
            
            # Plot scatter points with high transparency
            ax.scatter(jittered_x, ratings, 
                      alpha=0.15, 
                      s=15,
                      color=color, 
                      label=None)
            
            # Plot regression line with enhanced visibility
            sns.regplot(x=data_x, y=ratings,
                       scatter=False,
                       line_kws={
                           'color': color,
                           'lw': 2,
                           'markersize': 4,
                           'markevery': 5
                       },
                       ci=95,
                       label=f'{label}' if idx == 0 else None,  # Only add label for first subplot
                       ax=ax)
    
    # Enhance grid and labels
    ax.grid(True, which='major', linestyle='--', alpha=0.3)
    ax.grid(True, which='minor', linestyle=':', alpha=0.15)
    
    # Only show x-label for bottom subplot
    if idx == 2:  # Bottom subplot
        ax.set_xlabel('AI Usage Frequency', fontsize=12, weight='bold')
    else:
        ax.set_xlabel('')
    
    ax.set_ylabel(f'{rating_type.capitalize()} Rating Received', fontsize=10, weight='bold')
    ax.set_title(title, fontsize=13, pad=10, weight='bold')
    
    # Expand y-axis slightly
    ax.set_ylim(0.5, 5.5)
    
    # Add horizontal reference lines at each rating level
    for y in range(1, 6):
        ax.axhline(y=y, color='gray', linestyle='--', alpha=0.2)
    
    # Set x-axis limits consistently
    ax.set_xlim(-1, max(max(chat_ai_freq), max(suggestions_ai_freq),
                       max(feedback_ai_freq), max(conversation_ai_freq)) + 1)
    
    # Reduce number of ticks
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))
    
    # Enhance tick labels
    ax.tick_params(axis='both', labelsize=10)

# Create a single legend outside the plots
legend = fig1.legend(loc='center right',
                    bbox_to_anchor=(0.98, 0.5),
                    fontsize=12,
                    title='AI Features',
                    title_fontsize=12,
                    framealpha=0.9,
                    edgecolor='white')

# Add a light box behind the legend
legend.get_frame().set_facecolor('white')
legend.get_frame().set_alpha(0.8)

# Adjust spacing between subplots
plt.subplots_adjust(hspace=0.3, right=0.85)  # Make room for legend

plt.show()
