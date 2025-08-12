# src/hybrid_chunking_modified.py (consider renaming the file)

import re
from typing import List, Dict, Any, Optional

# Assume these are correctly imported from your project files
# NOTE: sentence_aware_chunking is NO LONGER USED by this modified chunker
# from sentence_chunking import sentence_aware_chunking
from embedding_processor import count_tokens, tokenizer # IMPORTANT: Assumes tokenizer is available via embedding_processor or passed in

# --- Table Detection Helper ---
def check_if_table(chunk_text: str) -> bool:
    # (Keep the check_if_table function as defined previously)
    lines = chunk_text.strip().split("\n")
    if len(lines) < 3: return False
    count = 0
    numeric_pattern = r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?%?\b"
    for line in lines:
        if len(re.findall(numeric_pattern, line)) > 2: count += 1
    if len(lines) > 0 and (count / len(lines)) > 0.4: return True
    if len(re.findall(numeric_pattern, chunk_text)) > (len(chunk_text.split()) / 5): return True
    return False

class HybridChunkerModified:
    """
    Modified Hybrid chunker aiming for chunks closer to max_tokens.
    Detects sections, uses fixed-size splitting for large elements,
    combines paragraphs, filters small chunks, detects tables, and outputs metadata.
    """

    def __init__(self, max_tokens=1024, min_tokens_per_chunk=50, overlap_ratio=0.1): # Default min_tokens higher now
        if tokenizer is None:
             raise ValueError("Tokenizer is required for HybridChunkerModified but is not loaded/passed.")
        if min_tokens_per_chunk >= max_tokens:
            raise ValueError("min_tokens_per_chunk must be less than max_tokens")
        self.max_tokens = max_tokens
        self.min_tokens_per_chunk = min_tokens_per_chunk
        self.overlap = int(max_tokens * overlap_ratio)
        if self.overlap >= max_tokens:
             self.overlap = max(0, max_tokens - 1)

        # Keep the improved regex patterns for section detection
        self.section_pattern = re.compile(
            # (Keep the improved section regex from the previous version)
            r"""
            (?:^|\n)(?:(Item\s+\d+[A-Za-z]?\.?\s+.*|Note\s+\d+\s*[-–—]?\s*.*|PART\s+[IVXLCDM]+\.?\s+.*|(?:[A-Z][A-Za-z\s']{5,}|[A-Z\s&]{5,}):?)(?:\n|$)(?:.*?\n)*?)(?=...)
            """, re.MULTILINE | re.VERBOSE | re.DOTALL # Simplified regex for brevity - use previous full one
        )
        self.heading_extractor = re.compile(
             r"^(?:Item\s+\d+...)" # Keep previous full one
        )
        self.tokenizer = tokenizer # Store the tokenizer instance

    def _split_fixed_size_within_element(self, text: str) -> List[str]:
        """Applies fixed-size chunking to a given piece of text."""
        if not text: return []
        # Encode the text ONCE
        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
        except Exception as e:
            print(f"WARN: Tokenization failed during fixed split: {e}")
            return []

        if not tokens: return []

        chunks_text = []
        start = 0
        while start < len(tokens):
            end = start + self.max_tokens
            chunk_tokens = tokens[start:end]
            # Decode chunk
            try:
                chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True).strip()
                if chunk_text: # Add only if not empty after decode/strip
                     chunks_text.append(chunk_text)
            except Exception as e:
                 print(f"WARN: Decoding failed during fixed split: {e}")


            if end >= len(tokens):
                break # Exit loop if we've reached the end

            # Calculate next start, ensuring progress
            next_start = start + self.max_tokens - self.overlap
            if next_start <= start: # Prevent infinite loop if overlap is too large or chunk size too small
                next_start = start + 1
            start = next_start

        return chunks_text

    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Split text based on document structure and fixed splitting for large elements."""
        return self._chunk_plain_text(text)

    def _chunk_plain_text(self, text: str) -> List[Dict[str, Any]]:
        """Process plain text by identifying structural elements."""
        chunks = []
        last_pos = 0

        # --- Section Processing ---
        section_matches = list(self.section_pattern.finditer(text)) # Use finditer
        if section_matches:
            for section_match in section_matches:
                section_text_full = section_match.group(1) # Group 1 should be the whole section including heading line
                if not section_text_full: continue
                section_text = section_text_full.strip()
                if not section_text: continue

                start_pos = section_match.start(1)

                # Capture and process intermediate text
                if start_pos > last_pos:
                    inter_text = text[last_pos:start_pos].strip()
                    if count_tokens(inter_text) >= self.min_tokens_per_chunk:
                         inter_chunks = self._process_paragraphs(inter_text, parent_heading=None)
                         chunks.extend(inter_chunks)

                # Extract heading
                heading_match = self.heading_extractor.match(section_text) # Match against stripped text
                heading = heading_match.group(0).strip() if heading_match else None

                section_tokens = count_tokens(section_text)
                is_section_table = check_if_table(section_text)

                # Keep section whole if it fits and meets min size
                if section_tokens <= self.max_tokens and section_tokens >= self.min_tokens_per_chunk:
                    chunks.append({
                        "text": section_text,
                        "metadata": {"heading": heading, "is_table": is_section_table}
                    })
                # If section is too large, split it using FIXED size splitting
                elif section_tokens > self.max_tokens:
                    if heading:
                        # Split content *after* the heading
                        content = section_text[len(heading):].strip()
                    else:
                        content = section_text # Split the whole thing if no heading found

                    # ---- MODIFIED: Use fixed splitting instead of sentence ----
                    sub_chunks_text = self._split_fixed_size_within_element(content)

                    for i, sub_chunk in enumerate(sub_chunks_text):
                         # No need to strip again, handled in _split_fixed_size_within_element
                        if not sub_chunk: continue

                        # Check min token size for the sub-chunk
                        sub_chunk_token_count = count_tokens(sub_chunk)
                        if sub_chunk_token_count < self.min_tokens_per_chunk:
                            continue # Skip small sub-chunk

                        is_sub_chunk_table = check_if_table(sub_chunk)
                        # Prepend heading only to the first sub-chunk
                        chunk_text_final = (heading + "\n\n" + sub_chunk) if heading and i == 0 else sub_chunk

                        chunks.append({
                            "text": chunk_text_final,
                            "metadata": {"heading": heading, "is_table": is_sub_chunk_table}
                        })
                # Else: section is smaller than min_tokens, gets skipped

                last_pos = section_match.end()

            # Process remaining text after the last section
            remaining_text = text[last_pos:].strip()
            if count_tokens(remaining_text) >= self.min_tokens_per_chunk:
                 chunks.extend(self._process_paragraphs(remaining_text, parent_heading=None))

        else:
            # No sections detected, process purely by paragraphs
            chunks.extend(self._process_paragraphs(text, parent_heading=None))

        # NOTE: Final fallback to sentence chunking removed as it conflicts with the goal
        # If no chunks are generated, it means the text didn't meet min criteria or was empty.

        return chunks

    def _process_paragraphs(self, text: str, parent_heading: Optional[str]) -> List[Dict[str, Any]]:
        """Processes text by splitting into paragraphs, combining, and fixed-splitting large ones."""
        chunks = []
        paragraphs = re.split(r"\n\s*\n", text)
        current_chunk_paras = []
        current_tokens = 0

        for para in paragraphs:
            para = para.strip()
            if not para: continue

            para_tokens = count_tokens(para)
            if para_tokens == 0: continue # Skip paragraphs that tokenize to zero

            # Handle paragraph larger than max_tokens using fixed split
            if para_tokens > self.max_tokens:
                # Finalize any previously accumulated chunk
                if current_chunk_paras:
                    chunk_text = "\n\n".join(current_chunk_paras)
                    chunk_token_count = count_tokens(chunk_text)
                    if chunk_token_count >= self.min_tokens_per_chunk:
                        is_table = check_if_table(chunk_text)
                        chunks.append({"text": chunk_text, "metadata": {"heading": parent_heading, "is_table": is_table}})
                    current_chunk_paras = []
                    current_tokens = 0

                # ---- MODIFIED: Use fixed splitting for the large paragraph ----
                sub_chunks_text = self._split_fixed_size_within_element(para)
                for sub_chunk in sub_chunks_text:
                    if not sub_chunk: continue
                    # Check min token size
                    sub_chunk_token_count = count_tokens(sub_chunk)
                    if sub_chunk_token_count < self.min_tokens_per_chunk:
                        continue
                    is_sub_table = check_if_table(sub_chunk)
                    chunks.append({"text": sub_chunk, "metadata": {"heading": parent_heading, "is_table": is_sub_table}})

            # Add paragraph to current chunk if it fits
            elif current_tokens + para_tokens <= self.max_tokens:
                # Only add if the paragraph itself meets min tokens, OR if it's being added to a non-empty chunk
                if para_tokens >= self.min_tokens_per_chunk or current_tokens > 0:
                     current_chunk_paras.append(para)
                     current_tokens += para_tokens
                # Else: Skip isolated small paragraph

            # Finalize current chunk and start new one if paragraph doesn't fit
            else:
                if current_chunk_paras: # Ensure we don't add empty chunks
                    chunk_text = "\n\n".join(current_chunk_paras)
                    chunk_token_count = count_tokens(chunk_text)
                    if chunk_token_count >= self.min_tokens_per_chunk:
                        is_table = check_if_table(chunk_text)
                        chunks.append({"text": chunk_text, "metadata": {"heading": parent_heading, "is_table": is_table}})

                # Start new chunk only if the new paragraph meets min size
                if para_tokens >= self.min_tokens_per_chunk:
                    current_chunk_paras = [para]
                    current_tokens = para_tokens
                else:
                    # Discard the small paragraph, reset accumulation
                    current_chunk_paras = []
                    current_tokens = 0

        # Add the last accumulated chunk if it's valid
        if current_chunk_paras:
            chunk_text = "\n\n".join(current_chunk_paras)
            chunk_token_count = count_tokens(chunk_text)
            if chunk_token_count >= self.min_tokens_per_chunk:
                is_table = check_if_table(chunk_text)
                chunks.append({"text": chunk_text, "metadata": {"heading": parent_heading, "is_table": is_table}})

        return chunks

# Helper function to use in your main pipeline.
def hybrid_element_semantic_chunking_modified(text: str, max_tokens=1024, min_tokens=50, overlap=0.1) -> List[Dict[str, Any]]:
    """
    Applies the modified HybridChunker aiming for chunks closer to max_tokens.

    Args:
        text: The input text extracted from the document.
        max_tokens: The target maximum token limit for a chunk.
        min_tokens: The minimum token count required for a chunk to be kept.
        overlap: Overlap ratio (0.0 to < 1.0) used in fixed splitting.

    Returns:
        A list of dictionaries, where each dictionary represents a chunk
        and contains 'text' and 'metadata' (including 'heading' and 'is_table').
    """
    chunker = HybridChunkerModified(
        max_tokens=max_tokens,
        min_tokens_per_chunk=min_tokens,
        overlap_ratio=overlap
    )
    return chunker.chunk_text(text)