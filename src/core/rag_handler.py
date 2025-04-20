
import chromadb
from langchain_chroma import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings 
import os

class RAGHandler:
    def __init__(self, config):
        self.config = config['rag']
        self.embedding_model_name = self.config['embedding_model']
        self.persist_directory = self.config['persist_directory']
        # Construct collection name based on config base + suffix
        self.collection_name = f"{self.config['collection_name_base']}_kb_lc"
        self.k = self.config['retrieval_k']

        print("Initializing RAG Handler...")
        self._load_embedding_model()
        self._connect_vector_store()
        self._initialize_retriever()
        print("RAG Handler initialized.")

    def _load_embedding_model(self):
        print(f"Loading Embedding Model ({self.embedding_model_name})...")
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={'device': 'cpu'}
            )
            print("Embedding Model loaded.")
        except Exception as e:
            print(f"Error loading Embedding model: {e}"); raise

    def _connect_vector_store(self):
        print(f"Connecting to Vector Store (Collection: '{self.collection_name}')...")
        if not os.path.exists(self.persist_directory):
            print(f"Fatal: ChromaDB persist directory '{self.persist_directory}' not found.")
            print("Please run the indexing script first.")
            raise FileNotFoundError(f"ChromaDB directory not found: {self.persist_directory}")
        try:
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            db_count = self.vector_store._collection.count()
            print(f"Connected. Collection contains {db_count} documents.")
            if db_count == 0:
                print(f"Warning: Vector store collection '{self.collection_name}' is empty!")
        except Exception as e:
            print(f"Error connecting to ChromaDB: {e}"); raise

    def _initialize_retriever(self):
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": self.k})
        print(f"RAG Retriever initialized (k={self.k}).")

    def retrieve_context(self, query: str) -> str:
        """Retrieves relevant context chunks from the vector store."""
        print(f"Retrieving context for query: '{query}'")
        if not query: return ""
        try:
            retrieved_docs = self.retriever.invoke(query)
            if not retrieved_docs: return ""

            context_parts = []
            for i, doc in enumerate(retrieved_docs):
                source = doc.metadata.get('source', 'N/A') # Get source from metadata if available
                content = doc.page_content
                context_parts.append(f"--- Start Context Chunk {i+1} (Source: {source}) ---\n{content}\n--- End Context Chunk {i+1} ---")
            print(f"Retrieved {len(retrieved_docs)} context chunks.")
            return "\n\n".join(context_parts)
        except Exception as e:
            print(f"Error during context retrieval: {e}")
            return "Error retrieving context from knowledge base."
