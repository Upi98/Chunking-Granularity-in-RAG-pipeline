import os
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI, APIError, RateLimitError 
from typing import List, Optional 

# Load environment variables (like OPENAI_API_KEY) from a .env file
load_dotenv()

# --- Client Initialization (New v1.0.0+ way) ---
# The client automatically looks for the OPENAI_API_KEY environment variable.
# It's generally better to initialize it once here rather than inside the function.
try:
    client = OpenAI()
    print("DEBUG: OpenAI client initialized successfully.")
    # You can optionally add a quick test call here if needed, like listing models,
    # but ensure it doesn't run every time the module is simply imported.
    # Example: models = client.models.list()
except APIError as e:
    print(f"ERROR: OpenAI API Error during initialization: {e}")
    client = None
except Exception as e:
    print(f"ERROR: Failed to initialize OpenAI client: {e}. Ensure OPENAI_API_KEY is set correctly.")
    client = None

# Define the standard embedding model (optional, makes it easier to change)
DEFAULT_EMBEDDING_MODEL = "text-embedding-ada-002"
# Define the expected dimension for the default model (ada-002 uses 1536)
DEFAULT_EMBEDDING_DIMENSION = 1536

def get_embedding(text: str, model: str = DEFAULT_EMBEDDING_MODEL) -> Optional[List[float]]:
    """
    Calls OpenAI API to get an embedding for the given text using v1.0.0+ syntax.
    Returns the embedding vector or None if an error occurs or text is empty.
    """
    if client is None:
        print("ERROR: OpenAI client is not initialized. Cannot get embedding.")
        return None

    # Handle empty or whitespace-only strings, as OpenAI API might error or return poor results
    if not text or not text.strip():
        print(f"Warning: Attempted to embed empty or whitespace string. Returning zero vector of dimension {DEFAULT_EMBEDDING_DIMENSION}.")
        # Return a zero vector of the expected dimension for your model
        return [0.0] * DEFAULT_EMBEDDING_DIMENSION

    # OpenAI API might have limits on input length, though embeddings handle reasonably long text.
    # Consider adding token counting and chunking here if you expect very long individual 'text' inputs,
    # although the main script should already be passing reasonably sized chunks.

    try:
        # Replace consecutive newlines/whitespace which can impact embedding quality
        processed_text = ' '.join(text.split())

        # --- Call the API using the new syntax ---
        response = client.embeddings.create(
            input=[processed_text],  # Input must be a list of strings
            model=model
        )

        # --- Extract the embedding from the response object ---
        # Accessing response.data (list) -> first item -> .embedding attribute
        embedding = response.data[0].embedding
        # print(f"DEBUG: Successfully received embedding, dimension: {len(embedding)}") # Optional debug print
        return embedding

    except RateLimitError as e:
        print(f"ERROR: OpenAI API rate limit exceeded: {e}. Consider adding retries with backoff.")
        # Implement retry logic here if needed
        return None
    except APIError as e:
        print(f"ERROR: OpenAI API returned an API Error: {e}")
        return None
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during embedding generation: {e}")
        # Optional: Log the full traceback
        # import traceback
        # traceback.print_exc()
        return None

def compute_embedding_cost(num_tokens: int, model: str = DEFAULT_EMBEDDING_MODEL) -> float:
    """
    Computes an estimated cost for embedding based on token count and model.
    NOTE: Verify pricing from OpenAI documentation as it can change.
    """
    cost = 0.0
    # Pricing varies by model. Add pricing for models you use.
    # Example for text-embedding-ada-002 (Verify current pricing!)
    if model == "text-embedding-ada-002":
        cost_per_1m_tokens = 0.10 # As of early 2024, $0.0001 / 1K tokens
        cost = (num_tokens / 1000000) * cost_per_1m_tokens
    # Add other models if needed
    # elif model == "some-other-model":
    #     cost_per_1k_tokens = X.XXXX
    #     cost = (num_tokens / 1000) * cost_per_1k_tokens
    else:
        print(f"Warning: Embedding cost calculation not defined for model '{model}'. Returning 0.")

    return cost

def count_tokens(text: str, model_encoding: str = "cl100k_base") -> int:
    """
    Counts the number of tokens in a given text using the specified tiktoken encoding.
    """
    if not text:
        return 0
    try:
        tokenizer = tiktoken.get_encoding(model_encoding)
        tokens = tokenizer.encode(text)
        return len(tokens)
    except Exception as e:
        print(f"Error during token counting: {e}")
        # Fallback: estimate tokens based on characters (crude)
        return len(text) // 4