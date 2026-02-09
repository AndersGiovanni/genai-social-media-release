import json
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm
import concurrent.futures
from functools import partial

client = OpenAI()

with open('data/plotting/feedback_categories_description.json', 'r') as f:
    change_categories = json.load(f)['categories']

feedback_comment_pairs = json.load(open('data/plotting/feedback_comment_pairs.json', 'r'))

model = "o3-mini-2025-01-31"

class ChangeCategory(BaseModel):
    change_category: str


PROMPT = """You are a helpful assistant that classifies changes between two versions of a text into one of the following change categories. Each change category contains a brief description of the category.

Change Categories:

{change_categories}

If none of the change categories apply, classify the change as "other".

Original Text: {original_text}

Revised Text: {revised_text}

Change Pattern:
"""

# Add all the usage patterns to the prompt
change_categories_prompt = "\n".join([f"- {category}: {description['description']}" for category, description in change_categories.items()])
PROMPT = PROMPT.format(change_categories=change_categories_prompt, original_text="{original_text}", revised_text="{revised_text}")

def process_feedback_pair(pair, prompt, model):
    original_text, revised_text = pair
    
    if original_text == revised_text:
        prediction = "no_change"
    else:
        response = client.beta.chat.completions.parse(
            model=model,
            messages=[{"role": "user", "content": prompt.format(original_text=original_text, revised_text=revised_text)}],
            response_format=ChangeCategory
        )
        prediction = response.choices[0].message.parsed.change_category
    
    return {
        "original_text": original_text,
        "revised_text": revised_text,
        "prediction": prediction
    }

predictions = {
    "cats": [],
    "oats": [],
    "politics": []
}

# Process feedback pairs in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
    for topic in tqdm(feedback_comment_pairs, desc="Topics"):
        # Create a partial function with the prompt and model already set
        process_func = partial(
            process_feedback_pair,
            prompt=PROMPT,
            model=model
        )
        
        # Process pairs in parallel and collect results
        future_results = list(tqdm(
            executor.map(process_func, feedback_comment_pairs[topic]),
            total=len(feedback_comment_pairs[topic]),
            desc=f"Processing {topic}"
        ))
        
        predictions[topic].extend(future_results)

with open(f'data/plotting/feedback_change_categories_{model}.json', 'w') as f:
    json.dump(predictions, f)
