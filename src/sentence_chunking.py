# src/sentence_chunking.py
import nltk
import tiktoken

nltk.download('punkt', quiet=True)

def sentence_aware_chunking(text, max_tokens=512):
    """
    Splits text into chunks by grouping sentences without breaking boundaries.
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # Check if adding the sentence exceeds the max token limit.
        if len(tokenizer.encode(current_chunk + " " + sentence)) > max_tokens:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence  # start a new chunk with this sentence
        else:
            current_chunk += " " + sentence
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks
