import os
import glob # Import glob for finding files
import chromadb
# LangChain imports
from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    # Keep specific loaders needed
    PyPDFLoader,
    TextLoader,
    JSONLoader,
)
# Keep DirectoryLoader import commented or removed if no longer needed
# from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Import tqdm for progress bars if desired (optional)
from tqdm import tqdm

# --- Configuration --- (Keep EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY, KB_DIR, CHUNK_SIZE, CHUNK_OVERLAP)
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
PERSIST_DIRECTORY = "chroma_db"
KB_DIR = "knowledge_bases"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# --- JSON Loader specific configuration (metadata extraction) --- (Keep metadata_func)
def metadata_func(record: dict, metadata: dict) -> dict:
    # ... (keep the existing metadata_func implementation) ...
    metadata["business_name"] = record.get("business_name", metadata.get("business_name"))
    metadata["business_type"] = record.get("business_type", metadata.get("business_type"))
    if "item" in record: metadata["source"] = "menu_item"; metadata["item_name"] = record.get("item")
    elif "address" in record: metadata["source"] = "listing"; metadata["address"] = record.get("address")
    elif "q" in record and "a" in record: metadata["source"] = "faq"; metadata["question"] = record.get("q")
    return metadata

# --- NEW Main Indexing Logic (Manual Iteration) ---
def main(kb_directory, collection_name_base):
    print(f"Scanning for documents in: {kb_directory}")
    all_documents = []
    # Supported extensions and their corresponding loaders/configs
    supported_loaders = {
        ".json": lambda path: JSONLoader(
                                file_path=path,
                                jq_schema='.',
                                content_key=None,
                                text_content=False,
                                json_lines=False,
                                metadata_func=metadata_func
                            ),
        ".pdf": lambda path: PyPDFLoader(path),
        ".txt": lambda path: TextLoader(path),
        # Add more extensions and loader lambdas here if needed
    }

    # Find all files recursively matching the supported extensions
    file_paths = []
    for ext in supported_loaders.keys():
        # Use recursive=True to search subdirectories as well
        file_paths.extend(glob.glob(os.path.join(kb_directory, '**', f'*{ext}'), recursive=True))

    if not file_paths:
        print(f"No files found with supported extensions ({list(supported_loaders.keys())}) in '{kb_directory}'. Exiting.")
        return

    print(f"Found {len(file_paths)} files to process.")

    # Iterate through found files and load them using the appropriate loader
    # Using tqdm for a progress bar (optional, remove if not needed/installed)
    for file_path in tqdm(file_paths, desc="Loading documents"):
        try:
            # Determine the loader based on the file extension
            ext = os.path.splitext(file_path)[1].lower()
            if ext in supported_loaders:
                # Instantiate the loader using the lambda function
                loader_instance = supported_loaders[ext](file_path)
                # Load documents from the single file
                loaded_docs = loader_instance.load()
                all_documents.extend(loaded_docs)
            else:
                # This shouldn't happen if glob is working correctly, but good practice
                print(f"Warning: Skipping file with unsupported extension: {file_path}")
        except Exception as e:
            # Log errors for specific files but continue with others
            print(f"\nError loading file '{file_path}': {e}")
            # Consider adding more specific error handling if needed

    if not all_documents:
        print("No documents were successfully loaded. Exiting.")
        return

    print(f"Successfully loaded content from {len(all_documents)} document sections.") # Langchain often returns multiple 'documents' per file

    # --- Text Splitting (Remains the same) ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    print("Splitting documents into chunks...")
    docs_split = text_splitter.split_documents(all_documents)
    print(f"Split into {len(docs_split)} chunks.")
    if not docs_split:
         print("No chunks generated after splitting. Exiting.")
         return

    # --- Embedding Model Loading (Remains the same) ---
    print("Loading Embedding Model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={'device': 'cpu'})
    print("Embedding Model loaded.")

    # --- ChromaDB Storage (Remains the same) ---
    collection_name = f"{collection_name_base}_kb_lc"
    print(f"Initializing ChromaDB and embedding documents into collection: '{collection_name}'...")
    try:
        client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        try:
            print(f"Attempting to delete existing collection '{collection_name}'...")
            client.delete_collection(name=collection_name)
            print(f"Existing collection '{collection_name}' deleted.")
        except Exception:
            print(f"Collection '{collection_name}' does not exist or couldn't be deleted.")

        vector_store = Chroma.from_documents(
            documents=docs_split,
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY,
            collection_name=collection_name,
        )
        print("Data successfully embedded and stored in ChromaDB.")
        print(f"Collection '{collection_name}' now contains {vector_store._collection.count()} documents.")

    except Exception as e:
        print(f"Error during ChromaDB embedding/storage: {e}")


if __name__ == "__main__":
    collection_base = "business_data"
    main(KB_DIR, collection_base)