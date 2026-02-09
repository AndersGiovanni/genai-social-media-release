import json
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm
import concurrent.futures
from functools import partial

client = OpenAI()

with open('data/plotting/chat_topics_description.json', 'r') as f:
    usage_categories = json.load(f)['usage_categories']

topic_predictions = json.load(open('data/plotting/chat_usage_categories.json', 'r'))

topic_messages = json.load(open('data/topic_messages.json', 'r'))
subcategory_descriptions = json.load(open('data/plotting/chat_subcategories_description.json', 'r'))

model = "o3-mini-2025-01-31"

CLASSIFY_CATEGORIES = True
CLASSIFY_SUBCATEGORIES = True

if CLASSIFY_CATEGORIES: 

    class UsagePattern(BaseModel):
        usage_pattern: str


    PROMPT = """You are a helpful assistant that classifies chat messages into one of the following message categories. Each message category contains a brief description of the category.

Message Categories:

{usage_categories}

If none of the message categories apply, classify the message as "other".

Message: {message}

Category:
    """

    # Add all the usage patterns to the prompt
    usage_categories_prompt = "\n".join([f"- {category}: {description['description']}" for category, description in usage_categories.items()])
    PROMPT = PROMPT.format(usage_categories=usage_categories_prompt, message="{message}")

    def process_message(message, prompt, model):
        response = client.beta.chat.completions.parse(
            model=model,
            messages=[{"role": "user", "content": prompt.format(message=message)}],
            response_format=UsagePattern
        )
        return {
            "message": message,
            "prediction": response.choices[0].message.parsed.usage_pattern
        }

    predictions = {
        "cats": [],
        "oats": [],
        "politics": []
    }

    # Process each topic's messages in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for topic in tqdm(topic_messages, desc="Topics"):
            # Create a partial function with the prompt and model already set
            process_func = partial(process_message, prompt=PROMPT, model=model)
            
            # Process messages in parallel and collect results
            future_results = list(tqdm(
                executor.map(process_func, topic_messages[topic]),
                total=len(topic_messages[topic]),
                desc=f"Messages for {topic}"
            ))
            
            predictions[topic].extend(future_results)

    with open(f'data/plotting/chat_usage_categories_{model}.json', 'w') as f:
        json.dump(predictions, f)

if CLASSIFY_SUBCATEGORIES:

    class Subcategory(BaseModel):
        subcategory: str

    PROMPT_SUBCATEGORIES = """You are a helpful assistant that classifies chat messages into one of the following message subcategories. Each subcategory contains a brief description of the subcategory.

Overall category: {usage_category}
Category description: {category_description}

Sub categories:
{subcategories}

If none of the subcategories apply, classify the message as "other".

Message: {message}

Subcategory:
    """

    def process_subcategory(sample, subcategory_descriptions, usage_categories, model):
        message = sample['message']
        category = sample['prediction']
        if category not in subcategory_descriptions['usage_categories']:
            return None
            
        SAMPLE_PROMPT = PROMPT_SUBCATEGORIES.format(
            message=message,
            usage_category=category,
            category_description=usage_categories[category]['description'],
            subcategories="\n".join([f"- {subcategory}: {description['description']}" 
                                   for subcategory, description in subcategory_descriptions['usage_categories'][category]['sub_categories'].items()])
        )
        
        response = client.beta.chat.completions.parse(
            model=model,
            messages=[{"role": "user", "content": SAMPLE_PROMPT}],
            response_format=Subcategory
        )
        
        return {
            "message": message,
            "category": category,
            "subcategory": response.choices[0].message.parsed.subcategory
        }

    predictions = {
        "cats": [],
        "oats": [],
        "politics": []
    }

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for topic in tqdm(topic_predictions, desc="Topics"):
            # Create partial function with all the constant parameters
            process_func = partial(
                process_subcategory,
                subcategory_descriptions=subcategory_descriptions,
                usage_categories=usage_categories,
                model=model
            )
            
            # Process messages in parallel and collect results
            future_results = list(tqdm(
                executor.map(process_func, topic_predictions[topic]),
                total=len(topic_predictions[topic]),
                desc=f"Subcategories for {topic}"
            ))
            
            # Filter out None results and add valid predictions
            predictions[topic].extend([result for result in future_results if result is not None])

    with open(f'data/plotting/chat_subcategories_{model}.json', 'w') as f:
        json.dump(predictions, f)
