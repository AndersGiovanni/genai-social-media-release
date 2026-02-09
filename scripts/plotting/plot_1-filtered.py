import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.gridspec as gridspec

# Font sizes
label_fontsize = 12
title_fontsize = 10

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
      0.7043478260869566,
      0.4482758620689655,
      0.12631578947368421,
      0.6835443037974683
    ],
    "err": [
      0.16521739130434782,
      0.21739130434782608,
      0.22413793103448273,
      0.2,
      0.24050632911392408
    ],
    "significance": [
      "ns",
      "***",
      "ns",
      "ns",
      "***"
    ],
    "n_values": [
      115,
      115,
      87,
      95,
      79
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
      3.39587852494577,
      3.712707182320442,
      3.5,
      3.3696236559139785,
      3.630359212050985
    ],
    "err": [
      0.0856832971800432,
      0.09668508287292843,
      0.09764309764309775,
      0.09139784946236551,
      0.07995365005793742
    ],
    "significance": [
      "***",
      "",
      "**",
      "***",
      ""
    ],
    "n_values": [
      922,
      724,
      594,
      744,
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
      3.8189655172413794,
      4.076271186440678,
      3.975,
      4.11340206185567,
      3.7752808988764044
    ],
    "err": [
      0.18103448275862055,
      0.14406779661016955,
      0.2250000000000001,
      0.14432989690721687,
      0.21348314606741603
    ],
    "significance": [
      "**",
      "",
      "",
      "",
      "**"
    ],
    "n_values": [
      116,
      118,
      80,
      97,
      89
    ],
    "treatment_labels": [
      "Chat",
      "Control",
      "Suggestions",
      "Feedback",
      "Conversation"
    ],
    "question_label": "I found the comments from other participants to be informative and high-quality.",
    "scale_type": "default"
  },
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
      28.79822437449556,
      19.239653512993264,
      29.1307966706302,
      29.994452149791954
    ],
    "err": [
      1.034057692307691,
      1.5423930589184813,
      0.9419634263715082,
      2.2071046373365064,
      0.6941920943134541
    ],
    "significance": [
      "",
      "***",
      "",
      "***",
      "***"
    ],
    "n_values": [
      1300,
      1239,
      1039,
      841,
      1442
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
      0.010662177328843996,
      0.00683371298405467,
      0.007716049382716049,
      0.005583126550868486
    ],
    "err": [
      0.005128205128205127,
      0.004489337822671156,
      0.004574791192103248,
      0.004629629629629629,
      0.003722084367245657
    ],
    "significance": [
      "ns",
      "ns",
      "ns",
      "ns",
      "ns"
    ],
    "n_values": [
      1918,
      1782,
      1317,
      1296,
      1612
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
      0.07014590347923681,
      0.060744115413819286,
      0.038580246913580245,
      0.06451612903225806
    ],
    "err": [
      0.007828810020876827,
      0.011784511784511786,
      0.012908124525436604,
      0.011574074074074077,
      0.0130272952853598
    ],
    "significance": [
      "ns",
      "***",
      "***",
      "ns",
      "***"
    ],
    "n_values": [
      1918,
      1782,
      1317,
      1296,
      1612
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
      0.08585858585858586,
      0.0774487471526196,
      0.06481481481481481,
      0.05707196029776675
    ],
    "err": [
      0.014649764767381071,
      0.012920875420875402,
      0.014426727410782075,
      0.013908179012345673,
      0.011786600496277916
    ],
    "significance": [
      "ns",
      "*",
      "***",
      "***",
      "***"
    ],
    "n_values": [
      1918,
      1782,
      1317,
      1296,
      1612
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
      0.3793490460157127,
      0.38496583143507973,
      0.4104938271604938,
      0.37655086848635233
    ],
    "err": [
      0.02298850574712641,
      0.02244668911335579,
      0.02505694760820043,
      0.029320987654321007,
      0.02358870967741933
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
      1782,
      1317,
      1296,
      1612
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
      0.3995510662177329,
      0.4267274107820805,
      0.4104938271604938,
      0.45905707196029777
    ],
    "err": [
      0.023474973931178256,
      0.021899551066217715,
      0.027334851936218707,
      0.027006172839506182,
      0.024193548387096753
    ],
    "significance": [
      "ns",
      "**",
      "ns",
      "*",
      "ns"
    ],
    "n_values": [
      1918,
      1782,
      1317,
      1296,
      1612
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
      0.010662177328843996,
      0.009111617312072893,
      0.015432098765432098,
      0.0037220843672456576
    ],
    "err": [
      0.005017921146953404,
      0.005050505050505052,
      0.005315110098709188,
      0.007716049382716049,
      0.003101736972704715
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
      1782,
      1317,
      1296,
      1612
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
      0.04377104377104377,
      0.03416856492027335,
      0.05246913580246913,
      0.033498759305210915
    ],
    "err": [
      0.008895866038723187,
      0.009539842873176205,
      0.00911161731207289,
      0.012345679012345678,
      0.008700372208436717
    ],
    "significance": [
      "ns",
      "ns",
      "ns",
      "**",
      "ns"
    ],
    "n_values": [
      1918,
      1782,
      1317,
      1296,
      1612
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
fig = plt.figure(figsize=(18, 12))  # Adjusted figure size for new layout
gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], hspace=0.4, wspace=0.3)
axes = [[plt.subplot(gs[i, j]) for j in range(3)] for i in range(2)]
axes = np.array(axes)

# Clean style settings
plt.style.use('seaborn-v0_8-white')  # Cleaner base style
for ax in axes.flat:
    ax.grid(True, linestyle=':', alpha=0.5)  # Softer gridlines
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Add row headers with more space
fig.text(0.02, 0.75, 'Producer Value', ha='left', va='center', fontsize=16, fontweight='bold', rotation=90)
fig.text(0.02, 0.25, 'Consumer Value', ha='left', va='center', fontsize=16, fontweight='bold', rotation=90)

###############################################################################
# Position A: Perceived Producer Value (stays the same)
###############################################################################
axes[0, 0].text(0.5, 1.08,
    "Q: AI suggestions can make it more likely for\nme to participate in online discussions",
    ha='center', va='bottom', transform=axes[0, 0].transAxes,
    fontsize=10, style='italic', wrap=True)

x_vals_tc = np.array(data_0_0["aiSuggestions"]["x_vals"])
y_vals_tc = np.array(data_0_0["aiSuggestions"]["y_vals"])
err_tc = np.array(data_0_0["aiSuggestions"]["err"])
significance_tc = data_0_0["aiSuggestions"]["significance"]

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
axes[0, 1].text(0.5, 1.10,
    "Average comment length in words",
    ha='center', va='bottom', transform=axes[0, 1].transAxes,
    fontsize=10, style='italic', wrap=True)

label_mapping = {
    "Control": 0,
    "Chat": 1,
    "Conversation": 2,
    "Feedback": 3,
    "Suggestions": 4
}

ordered_indices = [label_mapping[label.split('_')[0].capitalize()] for label in data_1_0["comment_words"]["treatment_labels"]]
y_vals_tc = np.array([y for _, y in sorted(zip(ordered_indices, data_1_0["comment_words"]["y_vals"]))])
err_tc = np.array([e for _, e in sorted(zip(ordered_indices, data_1_0["comment_words"]["err"]))])
significance_tc = [s for _, s in sorted(zip(ordered_indices, data_1_0["comment_words"]["significance"]))]

x_vals_tc = np.array([1, 2, 3, 4, 5])

for i, (x, y, err, sig) in enumerate(zip(x_vals_tc, y_vals_tc, err_tc, significance_tc)):
    axes[0, 1].errorbar(
        x, y, yerr=err,
        fmt='o', capsize=4, color=treatment_colors[i],
        ecolor=treatment_colors[i]
    )
    if sig != "ns":
        axes[0, 1].text(x, y + 1, sig, ha='center', va='bottom', fontsize=12)

ax = axes[0, 1]
ax.set_xticks([])
ax.set_ylabel("Number of words")

###############################################################################
# Position C: Shannon Entropy
###############################################################################
axes[0, 2].text(0.5, 1.10,
    "Diversity of Contributions",
    ha='center', va='bottom', transform=axes[0, 2].transAxes,
    fontsize=10, style='italic', wrap=True)

ordered_indices = [label_mapping[label] for label in data_2_0["shannon_entropy"]["treatment_labels"]]
y_vals_tc = np.array([y for _, y in sorted(zip(ordered_indices, data_2_0["shannon_entropy"]["y_vals"]))])
err_tc = np.array([e for _, e in sorted(zip(ordered_indices, data_2_0["shannon_entropy"]["err"]))])
significance_tc = [s for _, s in sorted(zip(ordered_indices, data_2_0["shannon_entropy"]["significance"]))]

x_vals_tc = np.array([1, 2, 3, 4, 5])

for i, (x, y, err, sig) in enumerate(zip(x_vals_tc, y_vals_tc, err_tc, significance_tc)):
    axes[0, 2].errorbar(
        x, y, yerr=err,
        fmt='o', capsize=4, color=treatment_colors[i],
        ecolor=treatment_colors[i]
    )
    if sig != "ns":
        axes[0, 2].text(x, y + 0.02, sig, ha='center', va='bottom', fontsize=12)

ax = axes[0, 2]
ax.set_xticks([])
ax.set_ylabel("Shannon Entropy")

###############################################################################
# Position D: Perceived Consumer Value
###############################################################################
axes[1, 0].text(0.5, 1.08,
    "Q: I found the comments from other participants to be\ninformative and high-quality.",
    ha='center', va='bottom', transform=axes[1, 0].transAxes,
    fontsize=10, style='italic', wrap=True)

ordered_indices = [label_mapping[label] for label in data_0_1["informative"]["treatment_labels"]]
y_vals_tc = np.array([y for _, y in sorted(zip(ordered_indices, data_0_1["informative"]["y_vals"]))])
err_tc = np.array([e for _, e in sorted(zip(ordered_indices, data_0_1["informative"]["err"]))])
significance_tc = [s for _, s in sorted(zip(ordered_indices, data_0_1["informative"]["significance"]))]

x_vals_tc = np.array([1, 2, 3, 4, 5])

for i, (x, y, err, sig) in enumerate(zip(x_vals_tc, y_vals_tc, err_tc, significance_tc)):
    axes[1, 0].errorbar(
        x, y, yerr=err,
        fmt='o', capsize=4, color=treatment_colors[i],
        ecolor=treatment_colors[i]
    )
    if sig:
        axes[1, 0].text(x, y + 0.1, sig, ha='center', va='bottom', fontsize=12)

ax = axes[1, 0]
ax.set_xticks([])
ax.set_ylabel("Rating (1-5)")

###############################################################################
# Position E: Reaction Types
###############################################################################
axes[1, 1].text(0.5, 1.10,
    "Distribution of different reaction types across treatments",
    ha='center', va='bottom', transform=axes[1, 1].transAxes,
    fontsize=10, style='italic', wrap=True)

reaction_types = ['likes_Like', 'likes_Love', 'likes_Funny', 'likes_Dislike']
reaction_labels = ['Like', 'Love', 'Funny', 'Dislike']

y_max = max([max(data_2_1[rt]["y_vals"]) * 100 for rt in reaction_types]) + 5
y_min = 0

group_width = 1.2
group_spacing = 2
x_offsets = np.arange(len(reaction_types)) * (5 + group_spacing)

legend_handles = []

for idx, (reaction_type, reaction_label) in enumerate(zip(reaction_types, reaction_labels)):
    ordered_indices = [label_mapping[label] for label in data_2_1[reaction_type]["treatment_labels"]]
    y_vals_tc = np.array([y * 100 for _, y in sorted(zip(ordered_indices, data_2_1[reaction_type]["y_vals"]))])
    err_tc = np.array([e * 100 for _, e in sorted(zip(ordered_indices, data_2_1[reaction_type]["err"]))])
    significance_tc = [s for _, s in sorted(zip(ordered_indices, data_2_1[reaction_type]["significance"]))]
    
    x_vals_tc = np.arange(5) + x_offsets[idx]
    
    for i in range(5):
        h = axes[1, 1].errorbar(
            x_vals_tc[i], y_vals_tc[i], yerr=err_tc[i],
            fmt='o', capsize=4, color=treatment_colors[i],
            elinewidth=1, markersize=6
        )
        if idx == 0:
            legend_handles.append(h)
    
    for x, y, sig in zip(x_vals_tc, y_vals_tc, significance_tc):
        if sig and sig != "ns":
            axes[1, 1].text(x, y + 2, sig, ha='center', va='bottom', fontsize=8)
    
    center_x = np.mean(x_vals_tc)
    axes[1, 1].text(center_x, y_max + 2, reaction_label,
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

axes[1, 1].set_ylim(y_min, y_max + 7)
axes[1, 1].set_xlim(-0.5, x_offsets[-1] + 5.5)
axes[1, 1].set_xticks([])
axes[1, 1].set_ylabel("Proportion of reactions")
axes[1, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}%'))

###############################################################################
# Position F: Comment Ratings
###############################################################################
axes[1, 2].text(0.5, 1.10,
    "Ratings of comments received to own comments",
    ha='center', va='bottom', transform=axes[1, 2].transAxes,
    fontsize=10, style='italic', wrap=True)

ordered_indices = [label_mapping[label] for label in data_1_1["commentRatings"]["treatment_labels"]]
y_vals_tc = np.array([y for _, y in sorted(zip(ordered_indices, data_1_1["commentRatings"]["y_vals"]))])
err_tc = np.array([e for _, e in sorted(zip(ordered_indices, data_1_1["commentRatings"]["err"]))])
significance_tc = [s for _, s in sorted(zip(ordered_indices, data_1_1["commentRatings"]["significance"]))]

x_vals_tc = np.array([1, 2, 3, 4, 5])

for i, (x, y, err, sig) in enumerate(zip(x_vals_tc, y_vals_tc, err_tc, significance_tc)):
    axes[1, 2].errorbar(
        x, y, yerr=err,
        fmt='o', capsize=4, color=treatment_colors[i],
        ecolor=treatment_colors[i]
    )
    if sig:
        axes[1, 2].text(x, y + 0.1, sig, ha='center', va='bottom', fontsize=12)

ax = axes[1, 2]
ax.set_xticks([])
ax.set_ylabel("Rating (1-5)")

# Remove any existing legends from subplots
for ax in axes.flat:
    if ax.get_legend() is not None:
        ax.get_legend().remove()

# Create a single legend for the entire figure
legend_handles = [plt.Line2D([0], [0], marker='o', color=color, linestyle='None', 
                            markersize=8, label=label)
                 for color, label in zip(treatment_colors, treatment_labels)]

fig.legend(handles=legend_handles,
          labels=treatment_labels,
          loc='center',
          bbox_to_anchor=(0.5, 0.02),
          ncol=5,
          frameon=False,
          borderaxespad=0.)

# Adjust layout to make room for the legend
plt.tight_layout(rect=[0.05, 0.08, 1, 0.93])

# Update all text elements to use larger fonts
for ax in axes.flat:
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.yaxis.label.set_size(label_fontsize)
    
    title_obj = ax.texts[0] if ax.texts else None
    if title_obj:
        title_obj.set_fontsize(title_fontsize)

plt.show()