from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import settings

def chunk_text(text: str) -> list[str]:
    """Split text into 300-500 token chunks (approximated)."""
    # Assuming 1 token ~ 4 characters
    char_chunk_size = settings.CHUNK_SIZE * 4
    char_overlap = settings.CHUNK_OVERLAP * 4
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=char_chunk_size,
        chunk_overlap=char_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks
