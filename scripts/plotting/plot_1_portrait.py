import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.gridspec as gridspec

# Perceived Producer Value
data_0_0 = {"aiSuggestions": {
    "x_vals": [
      1,
      2,
      3,
      4,
      5
    ],
    "y_vals": [
      0.26956521739130435,
      0.6666666666666666,
      0.2833333333333333,
      0.10833333333333334,
      0.5555555555555556
    ],
    "err": [
      0.1608695652173913,
      0.21544715447154475,
      0.2,
      0.17083333333333334,
      0.20096153846153691
    ],
    "significance": [
      "ns",
      "***",
      "ns",
      "ns",
      "**"
    ],
    "n_values": [
      115,
      123,
      120,
      120,
      117
    ],
    "treatment_labels": [
      "Control",
      "Chat",
      "Conversation",
      "Feedback",
      "Suggestions"
    ],
    "question_label": "AI suggestions can make it more likely for me to participate in online discussions."
  }}

# Measurable Consumer Value
data_1_1 = {
  "commentRatings": {
    "x_vals": [
      1,
      2,
      3,
      4,
      5
    ],
    "y_vals": [
      3.4054600606673406,
      3.7406311637080867,
      3.474530831099196,
      3.38663967611336,
      3.630359212050985
    ],
    "err": [
      0.08291203235591516,
      0.0769230769230771,
      0.08579088471849872,
      0.07894736842105265,
      0.07995365005793742
    ],
    "significance": [
      "***",
      "*",
      "***",
      "***",
      ""
    ],
    "n_values": [
      989,
      1014,
      746,
      988,
      863
    ],
    "treatment_labels": [
      "Chat",
      "Suggestions",
      "Feedback",
      "Conversation",
      "Control"
    ],
    "question_label": "Comment Ratings by Treatment"
  }
}

# Perceived Consumer Value
data_0_1 = {
    "informative": {
    "x_vals": [
      1,
      2,
      3,
      4,
      5
    ],
    "y_vals": [
      3.8548387096774195,
      4.0625,
      4.076271186440678,
      4.066666666666666,
      3.7903225806451615
    ],
    "err": [
      0.17741935483870952,
      0.1328125,
      0.14406779661016955,
      0.16666666666666696,
      0.17741935483870952
    ],
    "significance": [
      "*",
      "",
      "",
      "",
      "**"
    ],
    "n_values": [
      124,
      128,
      118,
      120,
      124
    ],
    "treatment_labels": [
      "Chat",
      "Feedback",
      "Control",
      "Suggestions",
      "Conversation"
    ],
    "question_label": "I found the comments from other participants to be informative and high-quality",
    "scale_type": "default"
  }
}

# Comment lengths
data_1_0 = {
  "comment_words": {
    "x_vals": [
      0,
      1,
      2,
      3,
      4
    ],
    "y_vals": [
      18.764615384615386,
      28.58695652173913,
      20.645641389085753,
      26.606511627906976,
      27.754317548746517
    ],
    "err": [
      1.0484615384615381,
      1.296195652173914,
      1.0086995038979474,
      1.7619534883720966,
      0.760055710306407
    ],
    "significance": [
      "",
      "***",
      "***",
      "***",
      "***"
    ],
    "n_values": [
      1300,
      1334,
      1411,
      1075,
      1795
    ],
    "treatment_labels": [
      "Control",
      "Chat",
      "Conversation",
      "Feedback",
      "Suggestions"
    ],
    "question_label": "Comment words analysis"
  }
}

# Entropy
data_2_0 = {
  "shannon_entropy": {
    "x_vals": [
      0,
      1,
      2,
      3,
      4
    ],
    "y_vals": [
      0.8878936632058528,
      0.9148302937366369,
      0.9210215017773887,
      0.8820040042916945,
      0.8878553600215716
    ],
    "err": [
      0.02508019352115698,
      0.015394711520484994,
      0.018338960721557385,
      0.02140352354994901,
      0.020368851083854445
    ],
    "significance": [
      "ns",
      "*",
      "**",
      "ns",
      "ns"
    ],
    "n_values": [
      81,
      79,
      84,
      81,
      81
    ],
    "treatment_labels": [
      "Control",
      "Chat",
      "Conversation",
      "Feedback",
      "Suggestions"
    ],
    "question_label": "Distribution of Normalized Shannon Entropy"
  }
}

# Likes distribution
data_2_1 = {
  "likes_Angry": {
    "x_vals": [
      0,
      1,
      2,
      3,
      4
    ],
    "y_vals": [
      0.009523809523809525,
      0.010810810810810811,
      0.005571030640668524,
      0.00825536598789213,
      0.003694581280788177
    ],
    "err": [
      0.005146520146520129,
      0.004864864864864864,
      0.003899721448467967,
      0.004402861860209136,
      0.0028735632183908046
    ],
    "significance": [
      "ns",
      "ns",
      "ns",
      "ns",
      "**"
    ],
    "n_values": [
      1918,
      1850,
      1795,
      1817,
      2436
    ],
    "treatment_labels": [
      "Control",
      "Chat",
      "Conversation",
      "Feedback",
      "Suggestions"
    ],
    "question_label": "Distribution of Angry reactions"
  },
  "likes_Dislike": {
    "x_vals": [
      0,
      1,
      2,
      3,
      4
    ],
    "y_vals": [
      0.031837160751565764,
      0.0681081081081081,
      0.06128133704735376,
      0.048431480462300495,
      0.05377668308702791
    ],
    "err": [
      0.008350730688935278,
      0.011891891891891895,
      0.010041782729805,
      0.010456796917996697,
      0.00903119868637111
    ],
    "significance": [
      "ns",
      "***",
      "***",
      "**",
      "***"
    ],
    "n_values": [
      1918,
      1850,
      1795,
      1817,
      2436
    ],
    "treatment_labels": [
      "Control",
      "Chat",
      "Conversation",
      "Feedback",
      "Suggestions"
    ],
    "question_label": "Distribution of Dislike reactions"
  },
  "likes_Funny": {
    "x_vals": [
      0,
      1,
      2,
      3,
      4
    ],
    "y_vals": [
      0.10454783063251437,
      0.08378378378378379,
      0.07632311977715878,
      0.05723720418271877,
      0.04967159277504105
    ],
    "err": [
      0.013604286461055923,
      0.013513513513513514,
      0.013927576601671307,
      0.011557512383048973,
      0.009441707717569782
    ],
    "significance": [
      "ns",
      "**",
      "***",
      "***",
      "***"
    ],
    "n_values": [
      1918,
      1850,
      1795,
      1817,
      2436
    ],
    "treatment_labels": [
      "Control",
      "Chat",
      "Conversation",
      "Feedback",
      "Suggestions"
    ],
    "question_label": "Distribution of Funny reactions"
  },
  "likes_Like": {
    "x_vals": [
      0,
      1,
      2,
      3,
      4
    ],
    "y_vals": [
      0.3761755485893417,
      0.3783783783783784,
      0.3626740947075209,
      0.3869014859658778,
      0.3485221674876847
    ],
    "err": [
      0.020376175548589337,
      0.021094594594594562,
      0.02395543175487469,
      0.021463951568519546,
      0.01889367816091958
    ],
    "significance": [
      "ns",
      "ns",
      "ns",
      "ns",
      "*"
    ],
    "n_values": [
      1918,
      1850,
      1795,
      1817,
      2436
    ],
    "treatment_labels": [
      "Control",
      "Chat",
      "Conversation",
      "Feedback",
      "Suggestions"
    ],
    "question_label": "Distribution of Like reactions"
  },
  "likes_Love": {
    "x_vals": [
      0,
      1,
      2,
      3,
      4
    ],
    "y_vals": [
      0.4410844629822732,
      0.40540540540540543,
      0.45403899721448465,
      0.4369840396257567,
      0.5123152709359606
    ],
    "err": [
      0.02085505735140769,
      0.022162162162162158,
      0.023398328690807824,
      0.024766097963676403,
      0.021346469622331665
    ],
    "significance": [
      "ns",
      "**",
      "ns",
      "ns",
      "***"
    ],
    "n_values": [
      1918,
      1850,
      1795,
      1817,
      2436
    ],
    "treatment_labels": [
      "Control",
      "Chat",
      "Conversation",
      "Feedback",
      "Suggestions"
    ],
    "question_label": "Distribution of Love reactions"
  },
  "likes_Sad": {
    "x_vals": [
      0,
      1,
      2,
      3,
      4
    ],
    "y_vals": [
      0.007885304659498209,
      0.01027027027027027,
      0.008913649025069638,
      0.015960374243258118,
      0.003284072249589491
    ],
    "err": [
      0.005017921146953404,
      0.005405405405405406,
      0.004456824512534818,
      0.006053935057787563,
      0.0024630541871921183
    ],
    "significance": [
      "ns",
      "ns",
      "ns",
      "*",
      "*"
    ],
    "n_values": [
      1918,
      1850,
      1795,
      1817,
      2436
    ],
    "treatment_labels": [
      "Control",
      "Chat",
      "Conversation",
      "Feedback",
      "Suggestions"
    ],
    "question_label": "Distribution of Sad reactions"
  },
  "likes_Wow": {
    "x_vals": [
      0,
      1,
      2,
      3,
      4
    ],
    "y_vals": [
      0.03506017791732077,
      0.043243243243243246,
      0.03119777158774373,
      0.04623004953219593,
      0.028735632183908046
    ],
    "err": [
      0.00784929356357928,
      0.009729729729729727,
      0.00891364902506964,
      0.010456796917996697,
      0.007389162561576353
    ],
    "significance": [
      "ns",
      "ns",
      "ns",
      "*",
      "ns"
    ],
    "n_values": [
      1918,
      1850,
      1795,
      1817,
      2436
    ],
    "treatment_labels": [
      "Control",
      "Chat",
      "Conversation",
      "Feedback",
      "Suggestions"
    ],
    "question_label": "Distribution of Wow reactions"
  }
}

# Use a colorblind-friendly palette
treatment_colors = ['#808080'] + sns.color_palette("colorblind")[:4]  # Gray for control, colorblind-friendly for treatments
treatment_labels = ["Control", "Chat", "Conversation", "Feedback", "Suggestions"]

# Create figure with better spacing
fig = plt.figure(figsize=(12, 18))  # Changed from (18, 12) to (12, 18) for portrait
gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1], hspace=0.4, wspace=0.3)  # Changed from (2, 3) to (3, 2)
axes = [[plt.subplot(gs[i, j]) for j in range(2)] for i in range(3)]  # Updated to match new grid
axes = np.array(axes)

# Clean style settings
plt.style.use('seaborn-v0_8-white')
for ax in axes.flat:
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Add row headers with more space
fig.text(0.2, 0.98, 'Producer Value', ha='left', va='center', fontsize=16, fontweight='bold')
fig.text(0.65, 0.98, 'Consumer Value', ha='left', va='center', fontsize=16, fontweight='bold')

# Add subplot labels (A-F) column by column
for col in range(2):  # For each column
    for row in range(3):  # For each row
        idx = col * 3 + row  # Calculate index for A-F labeling
        ax = axes[row, col]
        label = f"({chr(65+idx)})"  # A, B, C, etc.
        ax.text(-0.1, 1.1, label, transform=ax.transAxes, 
                fontsize=14, fontweight='bold')

# Standardize y-axis ranges for similar metrics
def set_common_ylim(ax1, ax2):
    if ax1.get_ylabel() == ax2.get_ylabel():
        ymin = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
        ymax = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
        ax1.set_ylim(ymin, ymax)
        ax2.set_ylim(ymin, ymax)

# Update axis labels to be more concise
axes[0, 0].set_ylabel("Participation likelihood\n∆(pre-post)")
axes[0, 1].set_ylabel("Rating (1-5)")
axes[1, 0].set_ylabel("Number of words")
axes[1, 1].set_ylabel("Shannon Entropy")
axes[2, 0].set_ylabel("Rating (1-5)")
axes[2, 1].set_ylabel("Distribution of different reaction types across treatments")

# Adjust subplot titles to be more concise
for ax in axes.flat:
    if ax.texts:  # If there's a title
        current_title = ax.texts[0].get_text()
        # Simplify the titles (examples)
        if "Distribution of Normalized Shannon Entropy" in current_title:
            ax.texts[0].set_text("Information diversity across comments")
        # Add more title simplifications as needed

# Move legend outside and below the plots
legend_handles = [plt.Line2D([0], [0], marker='o', color=color, linestyle='None', 
                            markersize=8, label=label)
                 for color, label in zip(treatment_colors, treatment_labels)]

fig.legend(handles=legend_handles,
          labels=treatment_labels,
          loc='center',
          bbox_to_anchor=(0.5, 0.02),
          ncol=5,
          frameon=False,
          fontsize=14)

# Standardize error bar appearance
error_bar_props = dict(capsize=4, capthick=1, elinewidth=1, alpha=0.8)

# Update subplot title font sizes
title_fontsize = 12
label_fontsize = 12

###############################################################################
# Top row (A, B): Producer Value metrics
###############################################################################

# Position A: Perceived Producer Value (stays the same)
###############################################################################
axes[0, 0].text(0.5, 1.08,  # Moved subtitle higher
    "Q: AI suggestions can make it more likely for\nme to participate in online discussions",
    ha='center', va='bottom', transform=axes[0, 0].transAxes,
    fontsize=14, style='italic', wrap=True)

# Use data from data_0_0
x_vals_tc = np.array(data_0_0["aiSuggestions"]["x_vals"])
y_vals_tc = np.array(data_0_0["aiSuggestions"]["y_vals"])
err_tc = np.array(data_0_0["aiSuggestions"]["err"])
significance_tc = data_0_0["aiSuggestions"]["significance"]

# Plot each treatment point with its own color
for i, (x, y, err, sig) in enumerate(zip(x_vals_tc, y_vals_tc, err_tc, significance_tc)):
    axes[0, 0].errorbar(
        x, y, yerr=err,
        fmt='o', capsize=4, color=treatment_colors[i],
        ecolor=treatment_colors[i]
    )
    if sig != "ns":
        axes[0, 0].text(x, y + 0.1, sig, ha='center', va='bottom', fontsize=12)

ax = axes[0, 0]
ax.set_xticks([])
ax.set_ylabel("∆(pre-post)")

###############################################################################
# Position B: Comment Length
###############################################################################
axes[1, 0].text(0.5, 1.10,
    "Average comment length in words",
    ha='center', va='bottom', transform=axes[1, 0].transAxes,
    fontsize=14, style='italic', wrap=True)

# Create mapping from data labels to desired order
label_mapping = {
    "Control": 0,
    "Chat": 1,
    "Conversation": 2,
    "Feedback": 3,
    "Suggestions": 4
}

# Reorder the data
ordered_indices = [label_mapping[label.split('_')[0].capitalize()] for label in data_1_0["comment_words"]["treatment_labels"]]
y_vals_tc = np.array([y for _, y in sorted(zip(ordered_indices, data_1_0["comment_words"]["y_vals"]))])
err_tc = np.array([e for _, e in sorted(zip(ordered_indices, data_1_0["comment_words"]["err"]))])
significance_tc = [s for _, s in sorted(zip(ordered_indices, data_1_0["comment_words"]["significance"]))]

# Use sequential x values (1 through 5)
x_vals_tc = np.array([1, 2, 3, 4, 5])

# Plot each treatment point with its own color
for i, (x, y, err, sig) in enumerate(zip(x_vals_tc, y_vals_tc, err_tc, significance_tc)):
    axes[1, 0].errorbar(
        x, y, yerr=err,
        fmt='o', capsize=4, color=treatment_colors[i],
        ecolor=treatment_colors[i]
    )
    if sig != "ns":  # Only add significance markers if they exist and aren't "ns"
        axes[1, 0].text(x, y + 1, sig, ha='center', va='bottom', fontsize=12)

ax = axes[1, 0]
ax.set_xticks([])
ax.set_ylabel("Number of words")

###############################################################################
# Middle row (C, D)
###############################################################################

# Position C: Shannon Entropy
axes[2, 0].text(0.5, 1.10,
    "Diversity of Contributions",
    ha='center', va='bottom', transform=axes[2, 0].transAxes,
    fontsize=14, style='italic', wrap=True)

# Create mapping from data labels to desired order
label_mapping = {
    "Control": 0,
    "Chat": 1,
    "Conversation": 2,
    "Feedback": 3,
    "Suggestions": 4
}

# Reorder the data
ordered_indices = [label_mapping[label] for label in data_2_0["shannon_entropy"]["treatment_labels"]]
y_vals_tc = np.array([y for _, y in sorted(zip(ordered_indices, data_2_0["shannon_entropy"]["y_vals"]))])
err_tc = np.array([e for _, e in sorted(zip(ordered_indices, data_2_0["shannon_entropy"]["err"]))])
significance_tc = [s for _, s in sorted(zip(ordered_indices, data_2_0["shannon_entropy"]["significance"]))]

# Use sequential x values (1 through 5)
x_vals_tc = np.array([1, 2, 3, 4, 5])

# Plot each treatment point with its own color
for i, (x, y, err, sig) in enumerate(zip(x_vals_tc, y_vals_tc, err_tc, significance_tc)):
    axes[2, 0].errorbar(
        x, y, yerr=err,
        fmt='o', capsize=4, color=treatment_colors[i],
        ecolor=treatment_colors[i]
    )
    if sig != "ns":  # Only add significance markers if they exist and aren't "ns"
        axes[2, 0].text(x, y + 0.02, sig, ha='center', va='bottom', fontsize=12)

ax = axes[2, 0]
ax.set_xticks([])
ax.set_ylabel("Shannon Entropy")

# Position D: Perceived Consumer Value
axes[0, 1].text(0.5, 1.08,
    "Q: I found the comments from other participants to be\ninformative and high-quality.",
    ha='center', va='bottom', transform=axes[0, 1].transAxes,
    fontsize=14, style='italic', wrap=True)

# Create mapping from data labels to desired order
label_mapping = {
    "Control": 0,
    "Chat": 1,
    "Conversation": 2,
    "Feedback": 3,
    "Suggestions": 4
}

# Reorder the data
ordered_indices = [label_mapping[label] for label in data_0_1["informative"]["treatment_labels"]]
y_vals_tc = np.array([y for _, y in sorted(zip(ordered_indices, data_0_1["informative"]["y_vals"]))])
err_tc = np.array([e for _, e in sorted(zip(ordered_indices, data_0_1["informative"]["err"]))])
significance_tc = [s for _, s in sorted(zip(ordered_indices, data_0_1["informative"]["significance"]))]

# Use sequential x values (1 through 5)
x_vals_tc = np.array([1, 2, 3, 4, 5])

# Plot each treatment point with its own color
for i, (x, y, err, sig) in enumerate(zip(x_vals_tc, y_vals_tc, err_tc, significance_tc)):
    axes[0, 1].errorbar(
        x, y, yerr=err,
        fmt='o', capsize=4, color=treatment_colors[i],
        ecolor=treatment_colors[i]
    )
    if sig:  # Only add significance markers if they exist
        axes[0, 1].text(x, y + 0.1, sig, ha='center', va='bottom', fontsize=12)

ax = axes[0, 1]
ax.set_xticks([])
ax.set_ylabel("Rating (1-5)")

###############################################################################
# Bottom row (E, F)
###############################################################################

# Position E: Reaction Types
axes[1, 1].text(0.5, 1.10,
    "Distribution of different reaction\ntypes across treatments",
    ha='center', va='bottom', transform=axes[1, 1].transAxes,
    fontsize=14, style='italic', wrap=True)

reaction_types = ['likes_Like', 'likes_Love', 'likes_Funny', 'likes_Dislike']
reaction_labels = ['Like', 'Love', 'Funny', 'Dislike']
treatment_labels = ["Control", "Chat", "Conversation", "Feedback", "Suggestions"]

# Find the global y-axis limits
y_max = max([max(data_2_1[rt]["y_vals"]) * 100 for rt in reaction_types]) + 5
y_min = 0

# Calculate x positions for each group
group_width = 1.2
group_spacing = 2
x_offsets = np.arange(len(reaction_types)) * (5 + group_spacing)

# For legend creation
legend_handles = []

for idx, (reaction_type, reaction_label) in enumerate(zip(reaction_types, reaction_labels)):
    # Reorder the data
    ordered_indices = [label_mapping[label] for label in data_2_1[reaction_type]["treatment_labels"]]
    y_vals_tc = np.array([y * 100 for _, y in sorted(zip(ordered_indices, data_2_1[reaction_type]["y_vals"]))])
    err_tc = np.array([e * 100 for _, e in sorted(zip(ordered_indices, data_2_1[reaction_type]["err"]))])
    significance_tc = [s for _, s in sorted(zip(ordered_indices, data_2_1[reaction_type]["significance"]))]
    
    # Calculate x positions for this group
    x_vals_tc = np.arange(5) + x_offsets[idx]
    
    # Plot points with different colors for each treatment
    for i in range(5):
        h = axes[1, 1].errorbar(
            x_vals_tc[i], y_vals_tc[i], yerr=err_tc[i],
            fmt='o', capsize=4, color=treatment_colors[i],
            elinewidth=1, markersize=6
        )
        if idx == 0:  # Only add to legend handles once
            legend_handles.append(h)
    
    # Add significance markers
    for x, y, sig in zip(x_vals_tc, y_vals_tc, significance_tc):
        if sig and sig != "ns":
            axes[1, 1].text(x, y + 2, sig, ha='center', va='bottom', fontsize=8)
    
    # Add reaction type label
    center_x = np.mean(x_vals_tc)
    axes[1, 1].text(center_x, y_max + 2, reaction_label,
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

# Set axis properties
axes[1, 1].set_ylim(y_min, y_max + 7)
axes[1, 1].set_xlim(-0.5, x_offsets[-1] + 5.5)
axes[1, 1].set_xticks([])
axes[1, 1].set_ylabel("Proportion of reactions")
axes[1, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}%'))

# Position F: Comment Ratings
axes[2, 1].text(0.5, 1.10,
    "Ratings of comments received to own comments",
    ha='center', va='bottom', transform=axes[2, 1].transAxes,
    fontsize=14, style='italic', wrap=True)

# Create mapping from data labels to desired order
label_mapping = {
    "Control": 0,    # Control
    "Chat": 1,        # Chat
    "Conversation": 2, # Conversation
    "Feedback": 3,    # Feedback
    "Suggestions": 4   # Suggestions
}

# Reorder the data
ordered_indices = [label_mapping[label] for label in data_1_1["commentRatings"]["treatment_labels"]]
y_vals_tc = np.array([y for _, y in sorted(zip(ordered_indices, data_1_1["commentRatings"]["y_vals"]))])
err_tc = np.array([e for _, e in sorted(zip(ordered_indices, data_1_1["commentRatings"]["err"]))])
significance_tc = [s for _, s in sorted(zip(ordered_indices, data_1_1["commentRatings"]["significance"]))]

# Use sequential x values (1 through 5)
x_vals_tc = np.array([1, 2, 3, 4, 5])

# Plot each treatment point with its own color
for i, (x, y, err, sig) in enumerate(zip(x_vals_tc, y_vals_tc, err_tc, significance_tc)):
    axes[2, 1].errorbar(
        x, y, yerr=err,
        fmt='o', capsize=4, color=treatment_colors[i],
        ecolor=treatment_colors[i]
    )
    if sig:  # Only add significance markers if they exist
        axes[2, 1].text(x, y + 0.1, sig, ha='center', va='bottom', fontsize=12)

ax = axes[2, 1]
ax.set_xticks([])
ax.set_ylabel("Rating (1-5)")

# Remove any existing legends from subplots
for ax in axes.flat:
    if ax.get_legend() is not None:
        ax.get_legend().remove()

# Create a single legend for the entire figure
legend_handles = [plt.Line2D([0], [0], marker='o', color=color, linestyle='None', 
                            markersize=12, label=label)
                 for color, label in zip(treatment_colors, treatment_labels)]

fig.legend(handles=legend_handles,
          labels=treatment_labels,
          loc='center',
          bbox_to_anchor=(0.5, 0.02),
          ncol=5,
          frameon=False,
          fontsize=14)

# Adjust layout to make room for the legend
plt.tight_layout(rect=[0.05, 0.08, 1, 0.93])  # Adjusted left margin for row headers


# Update all text elements to use larger fonts
for ax in axes.flat:
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.yaxis.label.set_size(label_fontsize)
    
    # Make subplot titles larger
    title_obj = ax.texts[0] if ax.texts else None
    if title_obj:
        title_obj.set_fontsize(title_fontsize)

plt.show()