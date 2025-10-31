import re

def split_into_chunks(text: str, max_chars: int = 800):
    if not text:
        return []
    text = re.sub(r'\\s+', ' ', text).strip()
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i+max_chars]
        if i + max_chars < len(text):
            last_period = chunk.rfind('. ')
            if last_period > 100:
                cut = last_period + 1
                chunk = chunk[:cut]
        chunks.append(chunk.strip())
        i += len(chunk)
    return chunks
