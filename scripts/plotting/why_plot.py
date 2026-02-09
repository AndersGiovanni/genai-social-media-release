import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import Levenshtein
from collections import defaultdict
import matplotlib.colors
import matplotlib.gridspec as gridspec

#### Define variables ####

FEEDBACK_ACCEPTANCE_RATE, FEEDBACK_USERS_N = 74.81, 98
SUGGESTIONS_ACCEPTANCE_RATE, SUGGESTIONS_USERS_N = 64.34, 83
CONVERSATION_ACCEPTANCE_RATE, CONVERSATION_USERS_N = 71.65, 91
CHAT_ACCEPTANCE_RATE, CHAT_USERS_N = 94.40, 118


SHOW_SUGGESTIONS_PLOT = True
SHOW_CHAT_PLOT = True
SHOW_FEEDBACK_PLOT = True
SHOW_CONVERSATION_PLOT = True

#################################
####### Suggestions #############
#################################


if SHOW_SUGGESTIONS_PLOT:
    # Set Nature-specific plotting parameters
    plt.rcParams.update({
        'font.size': 8,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'figure.titlesize': 10
    })

    # Define Nature-compatible dimensions (convert mm to inches)
    mm_to_inch = 1/25.4
    fig_width = 180 * mm_to_inch  # 180mm is Nature's full page width
    fig_height = 100 * mm_to_inch   # Increased height to allow more margin at top

    suggestions_metrics = {
    'total_generations': 1245,
    'total_selections': 1197,
    'Agree_selections': 582,
    'neutral_selections': 359,
    'Disagree_selections': 256,
    'topics': {
        'Cats': {
            'Agree_selections': 168,
            'neutral_selections': 130,
            'Disagree_selections': 90,
            'usage_rate': 57.14,
            'total_generations': 394,
            'total_selections': 388,
            'users_n': 72,
        },  
        'Oats': {
            'Agree_selections': 216,
            'neutral_selections': 117,
            'Disagree_selections': 90,
            'usage_rate': 56.35,
            'users_n': 71,
            'total_generations': 450,
            'total_selections': 423,
        },
        'Politics': {
            'Agree_selections': 198,
            'neutral_selections': 112,
            'Disagree_selections': 76,
            'usage_rate': 60,
            'users_n': 75,
            'total_generations': 401,
            'total_selections': 386,
        }
        }
    }

    # Create figure with two subplots with appropriate spacing for Nature
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height), 
                                  gridspec_kw={'wspace': 0.25, 'width_ratios': [0.6, 1.2]})
    
    # Add subplot labels (a, b) with Nature formatting
    ax1.text(-0.18, 1.15, 'h', transform=ax1.transAxes, 
            fontsize=14, fontweight='bold')
    ax2.text(0.0, 1.15, 'i', transform=ax2.transAxes, 
            fontsize=14, fontweight='bold')

    # First subplot (original plot)
    total = suggestions_metrics['total_selections']
    selections = {
        'Agree': suggestions_metrics['Agree_selections'],
        'Neutral': suggestions_metrics['neutral_selections'],
        'Disagree': suggestions_metrics['Disagree_selections']
    }
    proportions = {k: (v/total)*100 for k, v in selections.items()}

    # Nature-friendly colors (colorblind-safe)
    colors = ['#A7D89E', '#BBD3EE', '#F2C0BF']  # Green, Blue, Red
    
    # Plot first subplot
    bars1 = ax1.bar(proportions.keys(), proportions.values(), color=colors, width=0.6)

    # Add labels for first subplot with clean formatting
    for bar, (label, value) in zip(bars1, selections.items()):
        percentage = proportions[label]
        count = selections[label]
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{percentage:.1f}%',
                ha='center', va='bottom', fontsize=7)

    ax1.set_title('Selection Distribution Overall', pad=10, fontsize=9.5)
    ax1.set_ylim(0, 65)  # Increased upper limit to allow more space at top
    ax1.set_ylabel('Percentage')
    
    # Remove top and right spines for cleaner look
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Add light grid lines
    ax1.yaxis.grid(True, linestyle='--', alpha=0.3)

    # Second subplot (topics)
    topics = suggestions_metrics['topics']
    x_positions = np.arange(len(topics))
    width = 0.25

    # Calculate proportions for each topic
    topic_data = {}
    for i, (topic, data) in enumerate(topics.items()):
        total = data['total_selections']
        topic_data[topic] = {
            'pos_prop': (data['Agree_selections'] / total) * 100,
            'neu_prop': (data['neutral_selections'] / total) * 100,
            'neg_prop': (data['Disagree_selections'] / total) * 100,
            'pos_count': data['Agree_selections'],
            'neu_count': data['neutral_selections'],
            'neg_count': data['Disagree_selections'],
            'usage_rate': data['usage_rate'],
            'users_n': data['users_n']
        }

    # Create grouped bars
    bars2_list = []
    categories = [('pos_prop', colors[0]), ('neu_prop', colors[1]), ('neg_prop', colors[2])]
    
    for i, category in enumerate(categories):
        prop_key, color = category
        props = [topic_data[topic][prop_key] for topic in topics.keys()]
        count_key = prop_key.replace('_prop', '_count')
        counts = [topic_data[topic][count_key] for topic in topics.keys()]
        
        bars2 = ax2.bar(x_positions + i*width, props, width, color=color)
        bars2_list.append(bars2[0])
        
        # Add labels with consistent formatting
        for j, (prop, count) in enumerate(zip(props, counts)):
            ax2.text(x_positions[j] + i*width, prop + 0.5,
                    f'{prop:.1f}%',
                    ha='center', va='bottom', fontsize=6.5)

    # Customize second subplot with Nature-style formatting
    ax2.set_title('Selection Distribution by Topic', pad=10, fontsize=9.5)
    ax2.set_ylabel('Percentage')
    ax2.set_xticks(x_positions + width)
    ax2.set_xticklabels(topics.keys())
    
    # Remove top and right spines
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Add light grid lines
    ax2.yaxis.grid(True, linestyle='--', alpha=0.3)

    # Add suggestions acceptance rate text with better formatting
    usage_text = f'Users using suggestions at\nleast once: {SUGGESTIONS_ACCEPTANCE_RATE:.1f}% (n={SUGGESTIONS_USERS_N})'
    ax1.text(1, 0.90, usage_text,
            transform=ax1.transAxes, ha='right', va='top', fontsize=7,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

    # Add usage rates for each topic at the top with better positioning
    # Position text in a row at the top of the plot (outside the bars area)
    for i, topic in enumerate(topics.keys()):
        usage_rate = topic_data[topic]['usage_rate']
        users_n = topic_data[topic]['users_n']
        
        # Position text above each topic with better spacing
        ax2.text(x_positions[i] + width, 61,
                f'{topic.title()} usage:\n{usage_rate:.0f}% (n={users_n})', 
                ha='center', va='bottom', fontsize=6.5,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    # Add significance markers
    def add_significance_marker(ax, x1, x2, y, h, text="*"):
        # Draw the horizontal line
        ax.hlines(y=y, xmin=x1, xmax=x2, color='black', linewidth=0.7)
        # Draw the vertical lines
        ax.vlines(x=[x1, x2], ymin=y, ymax=y-h, color='black', linewidth=0.7)
        # Add the significance marker
        ax.text((x1 + x2) * 0.5, y*1.01, text, ha='center', va='bottom', fontsize=10)

    # Add significance markers between pairs with appropriate positioning
    # Cats vs Oats
    add_significance_marker(ax2, 
                          x1=x_positions[0] + 0.8,  # Center of Cats
                          x2=x_positions[1] + 0.7,  # Center of Oats
                          y=56,  # Raised height to accommodate usage rate text
                          h=0.8,   # Height of vertical lines
                          text="†")  # Single star for p < 0.05
    
    # Cats vs Politics
    add_significance_marker(ax2, 
                          x1=x_positions[1] + 0.8,  # Center of Oats
                          x2=x_positions[2] + 0.7,  # Center of Politics
                          y=56,  # Raised for consistency
                          h=0.8,
                          text="†")

    # Set y-axis limit to accommodate all elements including usage rates
    ax2.set_ylim(0, 65)  # Increased upper limit to fit the usage rate text and significance markers

    # Add legend with Nature-appropriate formatting
    legend = fig.legend(bars2_list, ['Agree', 'Neutral', 'Disagree'],
                       loc='lower center', bbox_to_anchor=(0.5, -0.08),
                       ncol=3, frameon=False, fontsize=7, handletextpad=0.5)

    # Ensure proper layout with sufficient margins
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjusted top margin 
    
    # Create figures directory if it doesn't exist
    import os
    os.makedirs('figures', exist_ok=True)
    
    # Save to file in various formats for Nature submission
    plt.savefig('figures/suggestions.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/suggestions.png', format='png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/suggestions.svg', format='svg', bbox_inches='tight')
    plt.savefig('figures/suggestions.eps', format='eps', bbox_inches='tight')



#################################
####### Chat Usage Patterns #####
#################################


def adjust_lightness(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    """
    import colorsys
    try:
        c = matplotlib.colors.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*matplotlib.colors.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


if SHOW_CHAT_PLOT:
    # Set Nature-specific plotting parameters
    plt.rcParams.update({
        'font.size': 8,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.titlesize': 12
    })

    # Define Nature-compatible dimensions (convert mm to inches)
    mm_to_inch = 1/25.4
    fig_width = 180 * mm_to_inch  # 180mm is Nature's full page width
    fig_height = 100 * mm_to_inch  # Reduced height for horizontal layout

    # Load the data
    with open('data/plotting/chat_usage_categories_o3-mini-2025-01-31.json', 'r') as f:
        usage_patterns = json.load(f)

    # Calculate overall predictions (including all categories for percentage base)
    all_predictions = []
    for topic in usage_patterns.values():
        all_predictions.extend([msg['prediction'] for msg in topic])

    # Get unique predictions and their counts (including all categories)
    prediction_counts_full = pd.Series(all_predictions).value_counts()
    total_messages = len(all_predictions)  # Total includes all categories
    
    # Define categories to exclude from display only
    categories_to_exclude = {
        'conspiracy_and_speculative_thinking',
        'sentiment_and_context_analysis'
    }

    # Filter out excluded categories for display only
    prediction_counts = prediction_counts_full[~prediction_counts_full.index.isin(categories_to_exclude)]
    prediction_props = (prediction_counts / total_messages * 100).round(1)  # Use full total

    # Sort predictions to put 'other' last and store the order
    other_mask = prediction_counts.index != 'other'
    prediction_counts = pd.concat([
        prediction_counts[other_mask].sort_values(ascending=True), 
        prediction_counts[~other_mask]
    ])
    prediction_props = prediction_props[prediction_counts.index]

    # Store this order for use in second plot
    ordered_pred_types = prediction_counts.index.tolist()

    # Define shorter label mappings
    label_mapping = {
        'casual_and_conversational_queries': 'Casual Queries',
        'fact_checking_and_scientific_validation': 'Fact Checking',
        'debate_and_argumentation_support': 'Argumentation',
        'engagement_and_social_interaction_assistance': 'Engagement',
        'instructional_and_how_to_requests': 'How-to Requests',
        'political_philosophical_and_ethical_discussions': 'Political Discussions',
        'sentiment_and_context_analysis': 'Sentiment Analysis',
        'conspiracy_and_speculative_thinking': 'Conspiracy',
        'other': 'Other'
    }

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height), 
                                  gridspec_kw={'wspace': 0.25, 'width_ratios': [1, 1.2]})

    # Add subplot labels (a, b) with Nature formatting and proper spacing
    ax1.text(-0.12, 1.15, 'a', transform=ax1.transAxes, 
            fontsize=14, fontweight='bold')
    ax2.text(-0.12, 1.15, 'b', transform=ax2.transAxes, 
            fontsize=14, fontweight='bold')

    # Use Nature-friendly colors (colorblind-safe)
    main_bar_color = '#88CCEE'  # Light blue
    
    # Use consistent y-positions for both plots
    y_positions = np.arange(len(ordered_pred_types))
    
    # Use consistent bar height that matches the grouped bars in the second plot
    consistent_bar_height = 0.25  # Match the bar_width used in second plot
    
    # Create main bars with consistent height and aligned positions
    bars1 = ax1.barh(y_positions, prediction_props, 
                    color=main_bar_color, alpha=0.7, height=consistent_bar_height*3)

    # Add percentage labels only (no counts)
    for i, (pred, prop) in enumerate(prediction_props.items()):
        if prop > 0:
            ax1.text(prop + 0.3, i, f'{prop:.1f}%', 
                    va='center', ha='left', fontsize=6.5)
        # Use shorter labels
        short_label = label_mapping.get(pred, pred.replace('_', ' ').title())
        ax1.text(-0.3, i, short_label, 
                va='center', ha='right', fontsize=7)

    # Customize first plot
    ax1.set_xlim(0, 45)
    ax1.set_yticks([])
    ax1.set_title('Prompt Categories Overall', fontsize=9.5, pad=10)
    ax1.set_xlabel('Percentage')

    # Second plot: Distribution by topic (grouped bars)
    topic_users_n = {
        'cats': 85.25,
        'cats_users_n': 104,
        'oats': 83.87,
        'oats_users_n': 104,
        'politics': 76.61,
        'politics_users_n': 95
    }

    # Nature-friendly color palette (colorblind-safe)
    topic_colors = {
        'cats': '#0072B2',    # Blue
        'oats': '#E69F00',    # Orange
        'politics': '#009E73'  # Green
    }

    # Use the same bar width and y-positions as the first plot for perfect alignment
    bar_width = 0.25
    
    for j, (topic, color) in enumerate(topic_colors.items()):
        bar_positions = y_positions + (j - 1) * bar_width
        
        topic_data = []
        topic_counts = []
        
        # Get total messages for this topic (including excluded categories)
        total_topic_messages = len(usage_patterns[topic])
        
        for pred in ordered_pred_types:
            count = len([m for m in usage_patterns[topic] if m['prediction'] == pred])
            # Use full total including excluded categories for percentage calculation
            percentage = (count / total_topic_messages * 100) if total_topic_messages > 0 else 0
            topic_data.append(percentage)
            topic_counts.append(count)
        
        bars = ax2.barh(bar_positions, topic_data, bar_width, 
                        label=topic.title(), color=color, alpha=0.9)
        
        # Add percentage labels only (no counts)
        for i, width in enumerate(topic_data):
            if width > 0:
                ax2.text(width + 0.3, bar_positions[i], 
                        f'{width:.1f}%',
                        va='center', ha='left',
                        fontsize=6.5)

    # Use shorter labels for y-axis (only on left plot, remove from right plot)
    ax2.set_yticks([])  # Remove y-axis labels from second plot
    
    # Set consistent y-axis limits for both plots
    y_min = -0.5
    y_max = len(ordered_pred_types) - 0.5
    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)
    
    # Add titles with proper spacing
    ax2.set_title('Prompt Categories by Topic', fontsize=9.5, pad=10)
    ax2.set_xlabel('Percentage')

    # Set same x-axis limits for both plots
    ax1.set_xlim(0, 47)
    ax2.set_xlim(0, 47)

    # Remove all spines for a cleaner look
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Add light grid lines (vertical for x-axis)
    ax1.grid(axis='x', linestyle='--', alpha=0.3)
    ax2.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Add horizontal dashed lines to help with alignment
    for i in range(len(ordered_pred_types)):
        # Add subtle horizontal lines at each category position
        ax1.axhline(y=i, color='gray', linestyle='--', alpha=0.2, linewidth=0.5)
        ax2.axhline(y=i, color='gray', linestyle='--', alpha=0.2, linewidth=0.5)

    # Add chat acceptance rate to first subplot
    ax1.text(0.95, 0.25, f'Users using chat at\nleast once: {CHAT_ACCEPTANCE_RATE:.1f}% (n={CHAT_USERS_N})',
            transform=ax1.transAxes, ha='right', va='top',
            fontsize=7, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

    # Add usage rates for each topic in second subplot
    for i, topic in enumerate(['politics', 'oats', 'cats']):
        usage_rate = topic_users_n[topic]
        users_n = topic_users_n[f'{topic}_users_n']
        ax2.text(0.95, 0.07 + i*0.05,  
                f'{topic.title()} usage: {usage_rate:.0f}% (n={users_n})', 
                transform=ax2.transAxes,
                ha='right', va='bottom',
                fontsize=6.5, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

    # Add legend
    legend = ax2.legend(bbox_to_anchor=(0.5, -0.15), 
                        loc='upper center', 
                        ncol=3,
                        borderaxespad=0.,
                        frameon=False,
                        fontsize=8)

    # Adjust the layout
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    # Create figures directory if it doesn't exist
    import os
    os.makedirs('figures', exist_ok=True)
    
    # Save to file in various formats for Nature submission
    plt.savefig('figures/chat.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/chat.png', format='png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/chat.svg', format='svg', bbox_inches='tight')




#################################
####### Feedback ###############
#################################

if SHOW_FEEDBACK_PLOT:
    # Set Nature-specific plotting parameters
    plt.rcParams.update({
        'font.size': 8,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'figure.titlesize': 10
    })

    # Define Nature-compatible dimensions (convert mm to inches)
    mm_to_inch = 1/25.4
    fig_width = 180 * mm_to_inch  # 180mm is Nature's full page width
    fig_height = 100 * mm_to_inch  # Reduced height for horizontal layout

    # Create figure with three subplots with Nature-optimized proportions
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.3], wspace=0.3)  # Adjusted width ratios and spacing
    ax1 = plt.subplot(gs[0])  # Now Feedback Change Categories
    ax2 = plt.subplot(gs[1])  # Now Feedback Changes by Topic
    ax3 = plt.subplot(gs[2])  # Now Jaccard Similarity Distribution
    
    # Add subplot labels (a, b, c) with Nature formatting
    ax1.text(-0.12, 1.15, 'e', transform=ax1.transAxes, 
            fontsize=14, fontweight='bold')
    ax2.text(-0.12, 1.15, 'f', transform=ax2.transAxes, 
            fontsize=14, fontweight='bold')
    ax3.text(-0.12, 1.15, 'g', transform=ax3.transAxes, 
            fontsize=14 , fontweight='bold')
    
    # Load feedback data
    with open('data/plotting/feedback_change_categories_o3-mini-2025-01-31.json', 'r') as f:
        feedback_patterns = json.load(f)

    with open('data/plotting/feedback_comment_pairs.json', 'r') as f:
        feedback_comment_pairs = json.load(f)

    # Calculate overall feedback predictions (including all categories for percentage base)
    all_feedback_predictions = []
    for topic in feedback_patterns.values():
        all_feedback_predictions.extend([msg['prediction'] for msg in topic])

    # Get unique predictions and their counts (including all categories)
    feedback_counts_full = pd.Series(all_feedback_predictions).value_counts()
    total_feedback = len(all_feedback_predictions)  # Total includes all categories
    
    # Define categories to exclude from display only
    categories_to_exclude = {
        'other'
    }

    # Filter out excluded categories for display only
    feedback_counts = feedback_counts_full[~feedback_counts_full.index.isin(categories_to_exclude)]
    feedback_props = (feedback_counts / total_feedback * 100).round(1)  # Use full total

    # Sort predictions to put remaining categories in ascending order
    feedback_counts = feedback_counts.sort_values(ascending=True)
    feedback_props = feedback_props[feedback_counts.index]
    ordered_feedback_types = feedback_counts.index.tolist()

    # Configuration for Jaccard similarity plot
    METRIC_TYPE = 'jaccard'  # Using Jaccard similarity as metric
    USE_LOG_SCALE = False

    topics_usage_rate = {
        'oats': 49.62,
        'oats_users_n': 65,
        'cats': 53.85,
        'cats_users_n': 70,
        'politics': 57.94,
        'politics_users_n': 73
    }

    # Nature-friendly color palette (colorblind-safe)
    topic_colors = {
        'cats': '#0072B2',    # Blue
        'oats': '#E69F00',    # Orange
        'politics': '#009E73'  # Green
    }

    # Helper functions
    def word_level_levenshtein(s1, s2):
        """Calculate Levenshtein distance between two strings at word level"""
        words1 = s1.split()
        words2 = s2.split()
        return Levenshtein.distance(' '.join(words1), ' '.join(words2))
    
    def calculate_jaccard_similarity(s1, s2):
        """Calculate Jaccard similarity between two strings at word level"""
        words1 = set(s1.split())
        words2 = set(s2.split())
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return intersection / union if union > 0 else 0
    
    def get_relative_metrics(before, after):
        """Calculate relative metrics for a pair of texts"""
        # Word-level Levenshtein
        lev_dist = word_level_levenshtein(before, after)
        rel_lev = lev_dist / len(before.split()) if before.split() else 0
        
        # Signed length difference
        len_diff = (len(after.split()) - len(before.split())) / len(before.split()) if before.split() else 0
        
        # Jaccard similarity
        jaccard = calculate_jaccard_similarity(before, after)
        
        return {
            'levenshtein': rel_lev,
            'length_diff': len_diff,
            'jaccard': jaccard
        }
    
    # Organize data by topic
    topic_data = defaultdict(list)
    all_metrics = []
    
    # Calculate metrics for each pair
    for topic, pairs in feedback_comment_pairs.items():
        for before, after in pairs:
            if before == after:
                continue
            metrics = get_relative_metrics(before, after)
            metric_value = metrics[METRIC_TYPE]
            
            # For log scale, ensure values are positive and non-zero
            if USE_LOG_SCALE and METRIC_TYPE == 'length_diff':
                metric_value = abs(metric_value) + 1e-10
            
            topic_data[topic].append(metric_value)
            all_metrics.append(metric_value)
    
    # First plot: Overall distribution of feedback changes
    bars1 = ax1.barh(range(len(feedback_counts)), feedback_props, 
                    color='#88CCEE', alpha=0.7, height=0.6)  # Increased bar height

    # Define label mapping for feedback categories
    feedback_label_mapping = {
        'no_change': 'No Change',
        'Structural Changes': 'Structural Changes', 
        'Factual & Informational Updates': 'Informational Updates',
        'Argumentation & Logic': 'Argumentation',
        'Lexical & Vocabulary Changes': 'Lexical Changes',
        'Persuasion & Engagement Enhancements': 'Engagment Enhancements',
        'Stylistic Adjustments': 'Stylistic Adjustments',
        'other': 'Other',
        'Sentiment & Emotional Adjustments': 'Sentiment Adjustments'
    }

    # Add labels for counts and percentages
    for i, (pred, prop) in enumerate(feedback_props.items()):
        count = feedback_counts[pred]
        # Add labels for non-zero bars with consistent styling
        if prop > 0:
            ax1.text(prop + 0.3, i, f'{prop:.1f}%', 
                    va='center', ha='left', fontsize=6)
        # Use mapped category names
        short_label = feedback_label_mapping.get(pred, pred.replace('_', ' ').title())
        ax1.text(-0.3, i, short_label,
                va='center', ha='right', fontsize=7)

    # Customize first plot
    ax1.set_yticks([])
    ax1.set_title('Change Categories Overall', pad=10, fontsize=9.5)
    ax1.set_xlabel('Percentage', labelpad=5)
    
    # Remove all spines
    for spine in ax1.spines.values():
        spine.set_visible(False)
    
    # Add grid lines
    ax1.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Add feedback acceptance rate to first plot
    feedback_text = f'Users using feedback at\nleast once: {FEEDBACK_ACCEPTANCE_RATE:.1f}% (n={FEEDBACK_USERS_N})'
    ax1.text(1.1, 0.15, feedback_text,
            transform=ax1.transAxes, ha='right', va='top',
            fontsize=7, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

    # Second plot: Distribution by topic
    bar_width = 0.25  # Match the bar width from chat/conversation plots
    y_positions = np.arange(len(ordered_feedback_types))

    for j, (topic, color) in enumerate(topic_colors.items()):
        bar_positions = y_positions + (j - 1) * bar_width
        
        topic_data_bars = []
        topic_counts = []
        
        # Get total messages for this topic (including excluded categories)
        total_topic_messages = len(feedback_patterns[topic])
        
        for pred in ordered_feedback_types:
            count = len([m for m in feedback_patterns[topic] if m['prediction'] == pred])
            # Use full total including excluded categories for percentage calculation
            percentage = (count / total_topic_messages * 100) if total_topic_messages > 0 else 0
            topic_data_bars.append(percentage)
            topic_counts.append(count)
        
        bars = ax2.barh(bar_positions, topic_data_bars, bar_width, 
                       label=topic.title(), color=color, alpha=0.9)
        
        # Add percentage labels only (no counts)
        for i, width in enumerate(topic_data_bars):
            if width > 0:
                ax2.text(width + 0.3, bar_positions[i], 
                        f'{width:.1f}%',
                        va='center', ha='left',
                        fontsize=6.5)

    # Customize second plot
    ax2.set_yticks([])
    ax2.set_title('Changes by Topic', pad=10, fontsize=9.5)
    ax2.set_xlabel('Percentage', labelpad=5)
    
    # Remove all spines
    for spine in ax2.spines.values():
        spine.set_visible(False)
    
    # Add grid lines
    ax2.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Add horizontal dashed lines to help with alignment
    for i in range(len(ordered_feedback_types)):
        ax1.axhline(y=i, color='gray', linestyle='--', alpha=0.2, linewidth=0.5)
        ax2.axhline(y=i, color='gray', linestyle='--', alpha=0.2, linewidth=0.5)
    
    # Set consistent x-axis limits for first two plots
    ax1.set_xlim(0, 45)
    ax2.set_xlim(0, 45)
    
    # Set consistent y-axis limits for both plots
    y_min = -0.5
    y_max = len(ordered_feedback_types) - 0.5
    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)

    # Plot KDE for each topic with appropriate colors
    for topic, metrics in topic_data.items():
        sns.kdeplot(data=metrics, ax=ax3, color=topic_colors[topic], alpha=0.8, 
                   linewidth=1.5, label=topic.title())
    
    # Third plot: Jaccard Similarity Distribution (previously first plot)
    # Plot KDE for all data with thicker line
    sns.kdeplot(data=all_metrics, ax=ax3, color='black', linewidth=2, 
                label='All Topics', linestyle='--')
    
    
    # Customize third plot
    metric_name = 'Jaccard Similarity'
    ax3.set_title(f'{metric_name}', pad=10, fontsize=9.5)
    
    # Set axis labels
    ax3.set_xlabel(metric_name, labelpad=5)
    ax3.set_ylabel('Density', labelpad=5)
    
    # Set x-axis limits to exclude unnecessary negative values
    ax3.set_xlim(left=0, right=1)  # Only show positive values
    
    # Remove all spines for a cleaner look
    for spine in ax3.spines.values():
        spine.set_visible(False)
    
    # Add light grid lines
    ax3.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Create more compact legend with Nature styling 
    ax3.legend(
        loc='upper center',
        bbox_to_anchor=(0.0, -0.18),
        ncol=4,
        frameon=False,
        fontsize=7,
        handletextpad=0.5
    )
    
    # Add usage rates for each topic positioned at the top right
    for i, topic in enumerate(topic_colors.keys()):
        usage_rate = topics_usage_rate[topic]
        users_n = topics_usage_rate[f'{topic}_users_n']
        
        # Create colored text box
        ax3.text(0.75, 0.25 - i*0.05,  
                f'{topic.title()} usage: {usage_rate:.0f}% (n={users_n})', 
                ha='right', va='top', fontsize=6.5,
                transform=ax3.transAxes)

    # Add subtle tick marks
    for ax in [ax1, ax2, ax3]:
        ax.tick_params(axis='both', which='both', direction='out', length=2, pad=2)
    
    # Ensure proper layout and save with Nature requirements
    plt.tight_layout(rect=[0, 0.1, 1, 0.98])  # Leave space for legends
    
    # Create figures directory if it doesn't exist
    import os
    os.makedirs('figures', exist_ok=True)
    
    # Save to file in various formats for Nature submission
    plt.savefig('figures/feedback.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/feedback.png', format='png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/feedback.svg', format='svg', bbox_inches='tight')
    plt.savefig('figures/feedback.eps', format='eps', bbox_inches='tight')




#################################
####### Conversation ###########
#################################


if  SHOW_CONVERSATION_PLOT:
    # Set Nature-specific plotting parameters
    plt.rcParams.update({
        'font.size': 8,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'figure.titlesize': 10
    })

    # Define Nature-compatible dimensions (convert mm to inches)
    mm_to_inch = 1/25.4
    fig_width = 180 * mm_to_inch  # 180mm is Nature's full page width
    fig_height = 100 * mm_to_inch  # Consistent height with chat plot

    with open('data/plotting/conversation_starter_predictions_o3-mini-2025-01-31.json', 'r') as f:
        conversation_starter_predictions = json.load(f)

    usage_rates = {
        'oats': 53.66,
        'oats_users_n': 66,
        'cats': 52,
        'cats_users_n': 65,
        'politics': 53.17,
        'politics_users_n': 67
    }
    
    # Calculate overall predictions (including all categories for percentage base)
    all_predictions = []
    for topic, messages in conversation_starter_predictions.items():
        for msg in messages:
            all_predictions.extend(msg['predictions_mapped'])

    # Get unique predictions and their counts (including all categories)
    prediction_counts_full = pd.Series(all_predictions).value_counts()
    total_messages = len(all_predictions)  # Total includes all categories

    # Define categories to exclude from display only (too small)
    categories_to_exclude = {
        'Research and Science Discussions',
        'Personal Experiences and Anecdotes', 
        'Animal Behavior and Intelligence',
        'Practical Advice and Suggestions'
    }

    # Filter out excluded categories for display only
    prediction_counts = prediction_counts_full[~prediction_counts_full.index.isin(categories_to_exclude)]
    prediction_props = (prediction_counts / total_messages * 100).round(1)  # Use full total

    # Sort predictions to put 'other' last and sort the rest by frequency
    other_mask = prediction_counts.index != 'other'
    prediction_counts = pd.concat([
        prediction_counts[other_mask].sort_values(ascending=True), 
        prediction_counts[~other_mask]
    ])
    prediction_props = prediction_props[prediction_counts.index]
    ordered_pred_types = prediction_counts.index.tolist()

    # Define shorter label mappings for conversation starters
    label_mapping = {
        'Questions, Discussions, and Exploration': 'Questions',
        'Sharing and Expressing Thoughts': 'Sharing Thoughts',
        'Humor and Lightheartedness': 'Humor',
        'Challenges and Debates': 'Debating',
        'Comparisons and Contrasts': 'Comparisons',
        'Philosophical Reflections and Speculations': 'Reflections',
        'other': 'Other',
        'Research and Science Discussions': 'Research',
        'Personal Experiences and Anecdotes': 'Personal Experiences',
        'Animal Behavior and Intelligence': 'Animal Behavior',
        'Practical Advice and Suggestions': 'Practical Advice'
    }

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height), 
                                  gridspec_kw={'wspace': 0.25, 'width_ratios': [1, 1.2]})


    # Add subplot labels (c, d) with Nature formatting and proper spacing
    ax1.text(-0.12, 1.15, 'c', transform=ax1.transAxes, 
            fontsize=14, fontweight='bold')
    ax2.text(-0.12, 1.15, 'd', transform=ax2.transAxes, 
            fontsize=14, fontweight='bold')

    # Use Nature-friendly colors (colorblind-safe)
    main_bar_color = '#88CCEE'  # Light blue
    
    # Use consistent y-positions for both plots
    y_positions = np.arange(len(ordered_pred_types))
    
    # Use consistent bar height that matches the grouped bars in the second plot
    consistent_bar_height = 0.25  # Match the bar_width used in second plot
    
    # Create main bars with consistent height and aligned positions
    bars1 = ax1.barh(y_positions, prediction_props, 
                    color=main_bar_color, alpha=0.7, height=consistent_bar_height*3)

    # Add percentage labels only (no counts)
    for i, (pred, prop) in enumerate(prediction_props.items()):
        if prop > 0:
            ax1.text(prop + 0.2, i, f'{prop:.1f}%', 
                    va='center', ha='left', fontsize=6)
        # Use shorter labels
        short_label = label_mapping.get(pred, pred.replace('_', ' ').title())
        ax1.text(-0.3, i, short_label, 
                va='center', ha='right', fontsize=7)

    # Customize first plot
    ax1.set_xlim(0, 40)
    ax1.set_yticks([])
    ax1.set_title('Conversation Starters Overall', fontsize=9.5, pad=10)
    ax1.set_xlabel('Percentage')

    # Second plot: Distribution by topic
    # Nature-friendly color palette (colorblind-safe)
    topic_colors = {
        'cats': '#0072B2',    # Blue
        'oats': '#E69F00',    # Orange
        'politics': '#009E73'  # Green
    }

    # Use the same bar width and y-positions as the first plot for perfect alignment
    bar_width = 0.25
    
    for j, (topic, color) in enumerate(topic_colors.items()):
        topic_predictions = []
        for msg in conversation_starter_predictions[topic]:
            topic_predictions.extend(msg['predictions_mapped'])
        
        # Get total topic predictions (including excluded categories)
        total_topic_predictions = len(topic_predictions)
        
        # Filter out excluded categories from topic predictions for counting displayed categories
        topic_counts = pd.Series(topic_predictions).value_counts()
        topic_props = pd.Series(0, index=ordered_pred_types)
        
        # Calculate percentages using full total (including excluded categories)
        for pred in ordered_pred_types:
            count = topic_counts.get(pred, 0)
            percentage = (count / total_topic_predictions * 100) if total_topic_predictions > 0 else 0
            topic_props[pred] = percentage
        
        # Position bars with perfect alignment
        bar_positions = y_positions + (j - 1) * bar_width
        bars = ax2.barh(bar_positions, topic_props, bar_width, 
                      label=topic.title(), color=color, alpha=0.9)
        
        # Add percentage labels only (no counts)
        for i, prop in enumerate(topic_props):
            if prop > 0:
                ax2.text(prop + 0.2, bar_positions[i], 
                        f'{prop:.1f}%',
                        va='center', ha='left',
                        fontsize=6.5)

    # Use shorter labels for y-axis (only on left plot, remove from right plot)
    ax2.set_yticks([])  # Remove y-axis labels from second plot
    
    # Set consistent y-axis limits for both plots
    y_min = -0.5
    y_max = len(ordered_pred_types) - 0.5
    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)
    
    # Add titles with proper spacing
    ax2.set_title('Conversation Starters by Topic', fontsize=9.5, pad=10)
    ax2.set_xlabel('Percentage')

    # Set same x-axis limits for both plots
    ax1.set_xlim(0, 40)
    ax2.set_xlim(0, 40)

    # Remove all spines for a cleaner look
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Add light grid lines (vertical for x-axis)
    ax1.grid(axis='x', linestyle='--', alpha=0.3)
    ax2.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Add horizontal dashed lines to help with alignment
    for i in range(len(ordered_pred_types)):
        # Add subtle horizontal lines at each category position
        ax1.axhline(y=i, color='gray', linestyle='--', alpha=0.2, linewidth=0.5)
        ax2.axhline(y=i, color='gray', linestyle='--', alpha=0.2, linewidth=0.5)

    # Add conversation acceptance rate to first subplot
    ax1.text(0.95, 0.25, f'Users using conversation starter\nat least once: {CONVERSATION_ACCEPTANCE_RATE:.1f}% (n={CONVERSATION_USERS_N})',
            transform=ax1.transAxes, ha='right', va='top',
            fontsize=7, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

    # Add usage rates for each topic in second subplot
    for i, topic in enumerate(['politics', 'oats', 'cats']):
        usage_rate = usage_rates[topic]
        users_n = usage_rates[f'{topic}_users_n']
        ax2.text(0.95, 0.07 + i*0.05,  
                f'{topic.title()} usage: {usage_rate:.0f}% (n={users_n})', 
                transform=ax2.transAxes,
                ha='right', va='bottom',
                fontsize=6.5, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

    # Add legend
    legend = ax2.legend(bbox_to_anchor=(0.5, -0.15), 
                        loc='upper center', 
                        ncol=3,
                        borderaxespad=0.,
                        frameon=False,
                        fontsize=8)

    # Adjust the layout
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    # Create figures directory if it doesn't exist
    import os
    os.makedirs('figures', exist_ok=True)
    
    # Save to file in various formats for Nature submission
    plt.savefig('figures/conversation.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/conversation.png', format='png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/conversation.svg', format='svg', bbox_inches='tight')
    plt.savefig('figures/conversation.eps', format='eps', bbox_inches='tight')


    