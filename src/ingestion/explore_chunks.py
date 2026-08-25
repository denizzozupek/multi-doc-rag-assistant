# 1. Ham PDF'i yükle
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, OVERLAP_SIZE, PDF_PATH


loader = PyMuPDFLoader(PDF_PATH)
raw_docs = loader.load()

# 2. Splitter ile parçala
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, 
    chunk_overlap=OVERLAP_SIZE
)
chunks = splitter.split_documents(raw_docs)

lengths = [len(c.page_content) for c in chunks]

# İlk chunk ve metadata incelemesi
print("--- CHUNK 0 ---\n")
print(chunks[0].page_content)
print("Metadata:\n", chunks[0].metadata)

print(chunks[0])

# Örtüşme (Overlap) kontrolü
print("\n--- CHUNK 1 (İlk 150 Karakter) ---")
print(chunks[1].page_content[:150])