import json
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm
import concurrent.futures
from functools import partial

with open('data/plotting/conversation_starter_data.json', 'r') as f:
    conversation_starter_data = json.load(f)

with open('data/plotting/conversation_starter_categories_descriptions.json', 'r') as f:
    categories_descriptions = json.load(f)

# Create a mapping from title to category
title_to_category = {}
for category in categories_descriptions['categories']:
    category_title = category['title']
    for title in category['titles']:
        title_to_category[title] = category_title


CLASSIFY_CONVERSATION_STARTERS = True

if CLASSIFY_CONVERSATION_STARTERS:
    class ConversationStarter(BaseModel):
        conversation_starters: list[str]

    PROMPT = """You are a helpful assistant that identifies which conversation starters influenced a social media comment.

Conversation starters:
{categories}

For the comment below, list any conversation starters that appear to have directly influenced or inspired the comment. If the comment doesn't seem influenced by any of the listed conversation starters, classify it as "other".

Comment: {comment}

Conversation starters that directly influenced this comment:
"""

client = OpenAI()
model = "o3-mini-2025-01-31"    

def process_message(sample):
    comment = sample['comment_content']
    categories = sample['treatment_data'][0][1]
    categories_string = "\n".join([f"- {key}: {value}" for key, value in categories.items()])
    sample_prompt = PROMPT.format(comment=comment, categories=categories_string)
    response = client.beta.chat.completions.parse(
        model=model,
        messages=[{"role": "user", "content": sample_prompt}],
        response_format=ConversationStarter
    )
    
    # Map each prediction to its category from title_to_category
    mapped_categories = []
    for starter in response.choices[0].message.parsed.conversation_starters:
        category = title_to_category.get(starter, "other")
        mapped_categories.append(category)
    
    # Return the original sample dict with additional fields
    result = sample.copy()  # Create a copy to avoid modifying the original
    result["predictions"] = response.choices[0].message.parsed.conversation_starters
    result["predictions_mapped"] = mapped_categories
    
    return result

predictions = {
    "cats": [],
    "oats": [],
    "politics": []
}

# Process each topic's messages in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
    for topic in tqdm(conversation_starter_data, desc="Topics"):
        # Create a partial function with the prompt and model already set
        process_func = partial(process_message)
        
        # Process messages in parallel and collect results
        future_results = list(tqdm(
            executor.map(process_func, conversation_starter_data[topic]),
            total=len(conversation_starter_data[topic]),
            desc=f"Messages for {topic}"
        ))
        
        predictions[topic].extend(future_results)