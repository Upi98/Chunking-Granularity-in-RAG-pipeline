import re
from typing import List, Dict, Any, Optional

# Assume these are correctly imported from your project files
from sentence_chunking import sentence_aware_chunking
from embedding_processor import count_tokens

# --- Table Detection Helper (similar to previous discussion) ---
def check_if_table(chunk_text: str) -> bool:
    """
    Simple heuristic: if more than half the lines contain multiple numeric values, treat as a table.
    Adjust threshold or logic as needed.
    """
    lines = chunk_text.strip().split("\n")
    if len(lines) < 3: # Require at least 3 lines
        return False
    count = 0
    numeric_pattern = r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?%?\b" # Allow % sign
    for line in lines:
        # Look for lines with multiple distinct numbers
        if len(re.findall(numeric_pattern, line)) > 2:
            count += 1
    # Consider it a table if a significant portion of lines are numeric rows
    # Or if the chunk is very dense with numbers overall
    if len(lines) > 0 and (count / len(lines)) > 0.4: # 40% threshold for lines
         return True
    # Fallback check for overall numeric density if line check fails
    if len(re.findall(numeric_pattern, chunk_text)) > (len(chunk_text.split()) / 5): # More than 20% of words are numbers
        return True
    return False

class HybridChunker:
    """
    Hybrid chunker adjusted for SEC filings (e.g., 10-Q).
    Detects sections, handles large sections with sentence chunking,
    detects potential tables, and outputs chunks with metadata.
    """

    def __init__(self, max_tokens=1024):
        self.max_tokens = max_tokens
        # Regex focused on SEC report structures (Item, Note, common titles)
        # Includes positive lookahead to handle splitting correctly
        # Prioritizes Item/Note, then Title Case/All Caps lines
        self.section_pattern = re.compile(
            r"""
            (?:^|\n)                                # Start of line or after newline
            (                                       # Start capturing group for the whole section
              (?:                                   # Non-capturing group for the heading
                (?:Item\s+\d+[A-Za-z]?\.?\s+.*)     # Matches "Item 1.", "Item 1A." etc.
                |
                (?:Note\s+\d+\s*[-–—]?\s*.*)        # Matches "Note 1 -", "Note 2" etc.
                |
                (?:PART\s+[IVXLCDM]+\.?\s+.*)       # Matches "PART I." etc.
                |
                (?:(?:[A-Z][A-Za-z\s']{5,}|[A-Z\s&]{5,}):?) # Matches Title Case: or ALL CAPS: headings (min 5 chars)
              )
              (?:\n|$)                              # Heading must end with newline or end of string
              (?:.*?\n)*?                           # Non-greedy match for section content lines
            )
            (?=                                     # Positive lookahead for the next heading or end of text
              (?:^|\n)(?:(?:Item\s+\d+|Note\s+\d+|PART\s+[IVXLCDM]+|(?:[A-Z][A-Za-z\s']{5,}|[A-Z\s&]{5,}):?)(?:\n|$))
              |
              \Z
            )
            """,
            re.MULTILINE | re.VERBOSE
        )
        # Simpler regex to extract the heading line itself from a matched section
        self.heading_extractor = re.compile(
             r"^(?:Item\s+\d+[A-Za-z]?\.?\s+.*|Note\s+\d+\s*[-–—]?\s*.*|PART\s+[IVXLCDM]+\.?\s+.*|(?:[A-Z][A-Za-z\s']{5,}|[A-Z\s&]{5,}):?)"
        )


    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Split text based on document structure and semantic coherence."""
        # For simplicity, treating extracted PDF text as plain text.
        # Consider adding pre-processing here to remove headers/footers/TOC if needed.
        return self._chunk_plain_text(text)

    def _chunk_plain_text(self, text: str) -> List[Dict[str, Any]]:
        """Process plain text by identifying structural elements."""
        sections = self.section_pattern.findall(text)
        chunks = []
        last_pos = 0

        if sections:
            for section_match in self.section_pattern.finditer(text):
                section_text = section_match.group(1).strip()
                start_pos = section_match.start(1)

                # Capture text between sections if any significant gap exists
                if start_pos > last_pos:
                    inter_text = text[last_pos:start_pos].strip()
                    if len(inter_text.split()) > 10: # Only add if reasonably substantial
                         # Treat intermediate text using paragraph logic (simplified here)
                         inter_chunks = self._process_paragraphs(inter_text, parent_heading=None)
                         chunks.extend(inter_chunks)

                # Extract heading
                heading_match = self.heading_extractor.match(section_text)
                heading = heading_match.group(0).strip() if heading_match else None

                section_tokens = count_tokens(section_text)
                is_section_table = check_if_table(section_text)

                if section_tokens <= self.max_tokens:
                    chunks.append({
                        "text": section_text,
                        "metadata": {"heading": heading, "is_table": is_section_table}
                    })
                else:
                    # Section too large, apply sentence chunking
                    if heading:
                        content = section_text[len(heading):].strip()
                    else:
                        # Should ideally not happen if regex finds section, but fallback
                        content = section_text

                    sub_chunks_text = sentence_aware_chunking(content, max_tokens=self.max_tokens)

                    for i, sub_chunk in enumerate(sub_chunks_text):
                        sub_chunk = sub_chunk.strip()
                        if not sub_chunk: continue
                        is_sub_chunk_table = check_if_table(sub_chunk)
                        # Prepend heading only to the first sub-chunk of the section
                        chunk_text = (heading + "\n\n" + sub_chunk) if heading and i == 0 else sub_chunk
                        chunks.append({
                            "text": chunk_text,
                            "metadata": {"heading": heading, "is_table": is_sub_chunk_table}
                        })
                last_pos = section_match.end()

            # Process any remaining text after the last section
            remaining_text = text[last_pos:].strip()
            if len(remaining_text.split()) > 10:
                 chunks.extend(self._process_paragraphs(remaining_text, parent_heading=None))

        else:
            # No sections detected, process purely by paragraphs
            chunks.extend(self._process_paragraphs(text, parent_heading=None))

        # Final fallback if absolutely no chunks were generated
        if not chunks:
             fallback_chunks = sentence_aware_chunking(text, max_tokens=self.max_tokens)
             for fb_chunk in fallback_chunks:
                 fb_chunk = fb_chunk.strip()
                 if fb_chunk:
                     is_fb_table = check_if_table(fb_chunk)
                     chunks.append({"text": fb_chunk, "metadata": {"heading": None, "is_table": is_fb_table}})

        return chunks

    def _process_paragraphs(self, text: str, parent_heading: Optional[str]) -> List[Dict[str, Any]]:
        """Processes text by splitting into paragraphs and combining/splitting them."""
        chunks = []
        paragraphs = re.split(r"\n\s*\n", text) # Split by blank lines
        current_chunk_paras = []
        current_tokens = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_tokens = count_tokens(para)

            # Handle paragraph larger than max_tokens
            if para_tokens > self.max_tokens:
                # Add any accumulated chunk before processing the large paragraph
                if current_chunk_paras:
                    chunk_text = "\n\n".join(current_chunk_paras)
                    is_table = check_if_table(chunk_text)
                    chunks.append({"text": chunk_text, "metadata": {"heading": parent_heading, "is_table": is_table}})
                    current_chunk_paras = []
                    current_tokens = 0

                # Sentence-chunk the large paragraph
                sub_chunks_text = sentence_aware_chunking(para, max_tokens=self.max_tokens)
                for sub_chunk in sub_chunks_text:
                    sub_chunk = sub_chunk.strip()
                    if sub_chunk:
                        is_sub_table = check_if_table(sub_chunk)
                        chunks.append({"text": sub_chunk, "metadata": {"heading": parent_heading, "is_table": is_sub_table}})

            # Add paragraph to current chunk if it fits
            elif current_tokens + para_tokens <= self.max_tokens:
                current_chunk_paras.append(para)
                current_tokens += para_tokens

            # Finalize current chunk and start a new one if paragraph doesn't fit
            else:
                chunk_text = "\n\n".join(current_chunk_paras)
                is_table = check_if_table(chunk_text)
                chunks.append({"text": chunk_text, "metadata": {"heading": parent_heading, "is_table": is_table}})
                current_chunk_paras = [para]
                current_tokens = para_tokens

        # Add the last accumulated chunk
        if current_chunk_paras:
            chunk_text = "\n\n".join(current_chunk_paras)
            is_table = check_if_table(chunk_text)
            chunks.append({"text": chunk_text, "metadata": {"heading": parent_heading, "is_table": is_table}})

        return chunks


# Helper function to use in your main pipeline.
def hybrid_element_semantic_chunking(text: str, max_tokens=1024) -> List[Dict[str, Any]]:
    """
    Applies the HybridChunker tailored for SEC filings.

    Args:
        text: The input text extracted from the document.
        max_tokens: The maximum token limit for a chunk.

    Returns:
        A list of dictionaries, where each dictionary represents a chunk
        and contains 'text' and 'metadata' (including 'heading' and 'is_table').
    """
    chunker = HybridChunker(max_tokens=max_tokens)
    return chunker.chunk_text(text)