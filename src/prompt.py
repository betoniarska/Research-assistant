from openai import OpenAI
from dotenv import load_dotenv
import os

# Load variables from .env
load_dotenv()

# init OpenAI client
client = OpenAI()

# Read the API key
key = os.getenv("OPENAI_API_KEY")

import os
os.environ["OPENAI_API_KEY"] = key

def build_prompt(query, results):
    context = "\n\n".join([
        f"""[Paper: {r['paper_title']}]
    [Section: {r['section']}]
    [Similarity: {r['score']:.3f}]

    {r['text']}"""
        for r in results
    ])

    return f"""
    You are an expert research assistant.

    Answer ONLY from the provided context.

    The retrieved passages include:
    - the source paper,
    - the section title,
    - a retrieval similarity score.

    If multiple papers discuss the same concept,
    clearly distinguish which paper each statement comes from.

    If the answer cannot be determined from the context, say so.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """



def ask_llm(query, results):

    prompt = build_prompt(query, results)

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content, results