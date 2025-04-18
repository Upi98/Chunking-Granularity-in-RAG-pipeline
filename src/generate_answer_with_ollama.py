import requests
import json

# Define the Ollama endpoint (default)
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
# Define the model you want to use (must match the name Ollama uses)
LOCAL_MODEL_NAME = "mistral" # Or "mistral:7b" if you need to be specific

def generate_answer_with_llama(question: str, context: str) -> str:
    """
    Generates an answer using a local LLM hosted by Ollama.
    """
    # --- Construct the Prompt ---
    # This is where you combine your retrieved context and question.
    # Adjust the formatting/instructions as needed for best results.
    prompt = f"""Use the following context to answer the question. Answer based only on the context provided.

Context:
{context}

Question: {question}

Answer:"""

    # --- Prepare the Request Payload ---
    payload = {
        "model": LOCAL_MODEL_NAME,
        "prompt": prompt,
        "stream": False  # Set to False to get the full response at once
        # Optional: Add other parameters Ollama supports, like temperature, top_k, etc.
        # "options": {
        #     "temperature": 0.7
        # }
    }

    # --- Make the API Call ---
    try:
        response = requests.post(OLLAMA_ENDPOINT, json=payload)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        # --- Process the Response ---
        response_data = response.json()
        generated_text = response_data.get("response", "").strip() # Extract the generated text
        
        # You might want to add post-processing here, e.g., removing repetitive phrases
        # or ensuring it didn't just repeat the question.

        return generated_text

    except requests.exceptions.RequestException as e:
        print(f"ERROR: Ollama API request failed: {e}")
        return "ERROR: Failed to get response from Ollama."
    except json.JSONDecodeError:
        print(f"ERROR: Could not decode JSON response from Ollama: {response.text}")
        return "ERROR: Invalid response format from Ollama."
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during Ollama interaction: {e}")
        return "ERROR: Unexpected error during generation."