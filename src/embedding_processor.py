import openai
import os
import tiktoken
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def get_embedding(text, model="text-embedding-ada-002"):
    """
    Calls OpenAI to get an embedding for the given text.
    """
    response = openai.Embedding.create(
        input=text,
        model=model
    )
    embedding = response["data"][0]["embedding"]
    return embedding

def compute_embedding_cost(token_count, cost_per_100k=0.001):
    """
    Computes an estimated cost for embedding a chunk based on token count.
    """
    cost = (token_count / 1000000.0) * cost_per_100k
    return cost

def count_tokens(text, model="cl100k_base"):
    """
    Counts the number of tokens in a given text using the specified tokenizer.
    """
    tokenizer = tiktoken.get_encoding(model)
    tokens = tokenizer.encode(text)
    return len(tokens)