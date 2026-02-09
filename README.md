# The Impact of Generative AI on Social Media: An Experimental Study

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Overview

This repository contains the analysis code and publication figures for the paper *"The Impact of Generative AI on Social Media: An Experimental Study"*.

The study investigates how generative AI integrations affect user behavior, engagement, and content quality on social media platforms through a controlled experiment with four AI treatment conditions: comment suggestions, open chat, AI feedback, and AI-generated conversation starters.

## Repository Structure

```
genai-social-media-release/
├── src/                                    # Main application source code
│   └── genai-social-media/
│       ├── config.py                       # Path configuration
│       └── dashboard/                      # Streamlit analytics dashboard
│           ├── runner.py                   # Dashboard entry point
│           ├── engagement_analytics.py     # Engagement metrics (control vs. treatment)
│           ├── players.py                  # Participant analysis
│           ├── players-paper.py            # Publication-specific player analysis
│           ├── topics.py                   # Topic-level analysis (cats, oats, politics)
│           ├── rounds_game.py              # Per-round timeline analysis
│           ├── suggestions.py              # Treatment 1: AI suggestions analysis
│           ├── chat.py                     # Treatment 2: Open chat analysis
│           ├── feedback.py                 # Treatment 3: AI feedback analysis
│           ├── conversation.py             # Treatment 4: Conversation starters analysis
│           ├── filters.py                  # Shared data filtering utilities
│           ├── perception_behaviour.py     # Survey perception analysis
│           ├── engagement_helpers/         # Engagement visualization and parsing
│           ├── players_helpers/            # Average treatment effect estimation
│           ├── rounds_helpers/             # Round-level visualization
│           ├── topics_helpers/             # Topic metrics and visuals
│           └── treatments/                 # Treatment-specific parsers and visuals
├── scripts/                                # Standalone analysis scripts
│   ├── plotting/                           # Publication figure generation
│   │   ├── plot_1.py                       # Main figure (landscape)
│   │   ├── plot_1_portrait.py              # Main figure (portrait variant)
│   │   ├── plot_1-filtered.py              # Filtered subset figure
│   │   └── why_plot.py                     # Treatment usage motivation plot
│   └── classification/                     # LLM-based content classification
│       ├── chat_classify.py                # Classify open chat messages
│       ├── conversation_classify.py        # Classify conversation starters
│       ├── feedback_classify.py            # Classify feedback comments
│       └── categories/                     # Classification taxonomy definitions
│           ├── chat_topics_description.json
│           ├── chat_subcategories_description.json
│           ├── feedback_categories_description.json
│           └── conversation_starter_categories_descriptions.json
├── figures/                                # Publication figures (PDF, PNG, SVG, EPS)
│   ├── fig-1-plots/                        # Main paper figures
│   ├── initial-survey/                     # Pre-experiment survey figures
│   ├── exit-survey/                        # Post-experiment survey figures
│   ├── treatment-effects/                  # Treatment effect plots
│   └── user-ratings/                       # Participant rating figures
├── .streamlit/config.toml                  # Streamlit dashboard theme
├── pyproject.toml                          # Python project configuration (Poetry)
├── poetry.lock                             # Locked dependency versions
├── requirements.txt                        # pip-compatible dependencies
└── LICENSE                                 # Apache License 2.0
```

## Prerequisites

- Python 3.11+
- [Poetry](https://python-poetry.org/) (recommended) or pip

## Installation

### Using Poetry (recommended)

```bash
git clone https://github.com/AndersGiovanni/genai-social-media-release.git
cd genai-social-media-release
poetry install
```

### Using pip

```bash
git clone https://github.com/AndersGiovanni/genai-social-media-release.git
cd genai-social-media-release
pip install -r requirements.txt
```

## Usage

### Interactive Analytics Dashboard

The Streamlit dashboard provides interactive exploration of the experimental results. It requires the experimental data files (see [Data Availability](#data-availability)).

```bash
cd src/genai-social-media/dashboard
streamlit run runner.py
```

The dashboard includes tabs for:
- **Engagement Analytics** &mdash; control vs. treatment group comparisons, Shannon entropy, sentiment, toxicity
- **Players** &mdash; participant demographics, survey responses, treatment effects
- **Topics** &mdash; topic-specific engagement metrics across discussion topics
- **Rounds** &mdash; per-round timeline visualization
- **Treatments 1-4** &mdash; detailed analysis of each AI treatment condition

### Generating Publication Figures

```bash
python scripts/plotting/plot_1.py
```

### Running LLM-Based Content Classification

The classification scripts use OpenAI's API to categorize user-generated content. An API key is required via the `OPENAI_API_KEY` environment variable.

```bash
export OPENAI_API_KEY="your-key-here"
python scripts/classification/chat_classify.py
```

## Data Availability

The experimental data containing participant responses are available from the corresponding author upon reasonable request. The data are not included in this repository to protect participant privacy.

## Citation

If you use this code in your research, please cite both the paper and this software:


### Software

```bibtex
@software{moller2025genai_social_media,
  author    = {Møller, Anders Giovanni},
  title     = {Analysis Code for "The Impact of Generative AI on Social Media"},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.18537633},
  url       = {https://doi.org/10.5281/zenodo.18537633}
}
```

## License

This project is licensed under the Apache License 2.0 &mdash; see the [LICENSE](LICENSE) file for details.
