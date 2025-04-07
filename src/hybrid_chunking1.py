# src/hybrid_chunking.py
import tiktoken
from sentence_chunking import sentence_aware_chunking

def hybrid_element_semantic_chunking(text, max_tokens=1024):
    """
    Combines document structure with semantic coherence.
    Splits text by elements (e.g., paragraphs) and further splits large elements.
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")
    # Split text into elements based on double newlines.
    elements = [element.strip() for element in text.split("\n\n") if element.strip()]
    chunks = []
    
    for element in elements:
        if len(tokenizer.encode(element)) > max_tokens:
            # Further split the element using sentence-aware chunking.
            sub_chunks = sentence_aware_chunking(element, max_tokens)
            chunks.extend(sub_chunks)
        else:
            chunks.append(element)
    return chunks
