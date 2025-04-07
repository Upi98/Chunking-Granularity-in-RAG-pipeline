import re
from sentence_chunking import sentence_aware_chunking 
from embedding_processor import count_tokens 

class HybridChunker:
    """Hybrid chunker that respects document structure and uses semantic chunking for large elements."""
    
    def __init__(self, max_tokens=1024):
        self.max_tokens = max_tokens
    
    def chunk_text(self, text):
        """Split text based on document structure and semantic coherence."""
        # Since processing only PDF files, always treat the text as plain text.
        return self._chunk_plain_text(text)
    
    def _chunk_plain_text(self, text):
        """Process plain text by identifying structural elements."""
        # Try to detect sections via headings using a regex pattern.
        section_pattern = r"(?:^|\n)((?:[A-Z][A-Za-z\s]+:|#{1,6}\s+[^\n]+)(?:\n|.)*?)(?=(?:[A-Z][A-Za-z\s]+:|\n#{1,6}\s+[^\n]+|\Z))"
        sections = re.findall(section_pattern, text, re.MULTILINE)
        chunks = []
        if sections:
            for section in sections:
                section_tokens = count_tokens(section)
                if section_tokens <= self.max_tokens:
                    chunks.append(section.strip())
                else:
                    heading_match = re.match(r"((?:[A-Z][A-Za-z\s]+:|#{1,6}\s+[^\n]+))", section)
                    if heading_match:
                        heading = heading_match.group(1)
                        content = section[len(heading):].strip()
                        sub_chunks = sentence_aware_chunking(content, max_tokens=self.max_tokens)
                        if sub_chunks:
                            chunks.append(heading + "\n" + sub_chunks[0])
                            chunks.extend(sub_chunks[1:])
                        else:
                            chunks.append(heading)
                    else:
                        chunks.extend(sentence_aware_chunking(section, max_tokens=self.max_tokens))
        else:
            paragraphs = re.split(r"\n\s*\n", text)
            current_chunk = []
            current_tokens = 0
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                para_tokens = count_tokens(para)
                if para_tokens > self.max_tokens:
                    if current_chunk:
                        chunks.append("\n\n".join(current_chunk))
                        current_chunk = []
                        current_tokens = 0
                    sub_chunks = sentence_aware_chunking(para, max_tokens=self.max_tokens)
                    chunks.extend(sub_chunks)
                elif current_tokens + para_tokens <= self.max_tokens:
                    current_chunk.append(para)
                    current_tokens += para_tokens
                else:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = [para]
                    current_tokens = para_tokens
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
        if not chunks:
            return sentence_aware_chunking(text, max_tokens=self.max_tokens)
        return chunks

# Helper function to use in your main pipeline.
def hybrid_element_semantic_chunking(text, max_tokens=1024):
    chunker = HybridChunker(max_tokens)
    return chunker.chunk_text(text)