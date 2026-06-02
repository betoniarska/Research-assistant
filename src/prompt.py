from openai import OpenAI
from src.vector_index import search
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
        f"[Section: {r['section']}]\n{r['text']}"
        for r in results
    ])

    return f"""
    You are an expert research assistant.

    Use ONLY the provided context to answer.

    - Be precise
    - Cite sections when possible
    - If unsure, say you don't know

    Context:
    {context}

    Question:
    {query}

    Answer (with reasoning):
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