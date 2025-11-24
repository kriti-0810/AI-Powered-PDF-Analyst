from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_text(extracted_pages, chunk_size=800, chunk_overlap=100):
    """
    extracted_pages = list of dicts from pdf_loader
    Returns a list of chunks with metadata
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = []

    for page in extracted_pages:
        page_text = page["text"]

        # Split the text into smaller chunks
        split_texts = text_splitter.split_text(page_text)

        # Create chunk objects with metadata
        for chunk in split_texts:
            chunks.append({
                "filename": page["filename"],
                "page_number": page["page_number"],
                "content": chunk
            })

    return chunks
