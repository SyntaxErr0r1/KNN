"""
Tool for creating SumeCzech summarization dataset using LLM API from OpenAI.
"""
import sys
import pandas as pd    

from urllib.parse import urlparse
from tqdm import tqdm
from datetime import datetime

from openai import OpenAI
from dotenv import load_dotenv
import os

print(os.getcwd())


load_dotenv("./.env")

print(os.getenv('TEST_VAR'))

print(f"OpenAI API key: {os.getenv('OPENAI_API_KEY')}")
client = OpenAI()



def print_usage():
    print("Usage: python dataset_tool.py <path-to-dataset> <output-path> [num-articles]")
    sys.exit(1)

if len(sys.argv) < 3:
    print_usage()

# get the first argument from the command line
dataset_path = sys.argv[1]

# get the second argument from the command line
output_path = sys.argv[2]

# get the number of articles to summarize
num_articles = int(sys.argv[3]) if len(sys.argv) > 3 else 1


print(f"Loading dataset from {dataset_path}...")
dataset = pd.read_json(path_or_buf=dataset_path, lines=True)


"""
Function to get the summary of an article using the LLM API from OpenAI.
"""
def get_summary(article):

    system_message = "Jsi pracovník, který má za úkol sumarizovat text. Když ti někdo pošle článek, tvou odpovědí bude 5 vět shrnujícich podstatné informace."
    
    response = client.chat.completions.create(
        model = "gpt-3.5-turbo-0125",
        temperature = 0.8,
        max_tokens = 4096,
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": article}
    ])

    return response.choices[0].message.content

if num_articles > len(dataset) or num_articles == 0:
    print("Note: Summarizing all articles in the dataset.")
    num_articles = len(dataset)


new_dataset = []
for i in tqdm(range(num_articles), desc="Summarizing articles"):


    article = dataset.iloc[i]
    article_text = article['text']
    # print(f"Summarizing article, text: {article_text[:50]}... (length: {len(article_text)} chars, md5: {article['md5']})")

    summary = get_summary(article_text)
    # print(f"Summary for article {i+1}:\n{summary}\n\n")

    new_dataset += [{
        'md5': article['md5'],
        'url': article['url'],
        'headline': article['headline'],
        'summary': summary,
        # 'text': article_text
    }]

print(f"Saving the summarized dataset to {output_path}")

new_dataset_pd = pd.DataFrame(new_dataset)

new_dataset_pd.to_json(output_path, orient='records', lines=True)