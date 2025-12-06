import os
import cohere
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
import gradio as gr
from typing import List, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Cohere client
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "your-cohere-api-key-here")
COHERE_CHAT_MODEL = os.getenv("COHERE_CHAT_MODEL", "command-r-plus-08-2024")
COHERE_EMBED_MODEL = os.getenv("COHERE_EMBED_MODEL", "embed-english-v3.0")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").lower()
LOCAL_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

if COHERE_API_KEY == "your-cohere-api-key-here":
    print("⚠️  WARNING: Using placeholder Cohere API key. Update .env to enable Cohere chat responses.")
else:
    print(f"✅ Cohere key loaded (starts with: {COHERE_API_KEY[:8]}...)")

co = cohere.Client(COHERE_API_KEY)

class RAGSystem:
    def __init__(self, data_folder: str = "data"):
        self.data_folder = data_folder
        self.vectorstore = None
        self.embeddings = self._create_embeddings()
        self.vector_dir = os.path.join("storage", "vectorstore")
        os.makedirs(self.vector_dir, exist_ok=True)
        self.load_or_build_vectorstore()

    def _create_embeddings(self):
        if EMBEDDING_PROVIDER == "cohere":
            if COHERE_API_KEY == "your-cohere-api-key-here":
                print("⚠️  Cohere embeddings requested but API key missing. Falling back to local embeddings.")
            else:
                print(f"Using Cohere embeddings model '{COHERE_EMBED_MODEL}'.")
                return CohereEmbeddings(
                    cohere_api_key=COHERE_API_KEY,
                    model=COHERE_EMBED_MODEL,
                )
        print(f"Using local embeddings model '{LOCAL_EMBED_MODEL}'.")
        return HuggingFaceEmbeddings(model_name=LOCAL_EMBED_MODEL)

    def load_or_build_vectorstore(self):
        index_path = os.path.join(self.vector_dir, "faiss_index")
        if os.path.isdir(index_path):
            try:
                self.vectorstore = FAISS.load_local(
                    index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                print("Loaded existing vector store from disk.")
                return
            except Exception as exc:
                print(f"Failed to load existing vector store: {exc}")
        self.load_documents()
        
    def load_documents(self):
        """Load all documents from the data folder"""
        print("Loading documents from data folder...")
        
        # Check if data folder exists
        if not os.path.exists(self.data_folder):
            print(f"⚠️  Warning: Data folder '{self.data_folder}' not found. Creating it...")
            os.makedirs(self.data_folder)
            print(f"Please add .txt or .md files to the '{self.data_folder}' folder and restart.")
            return
        
        # Load text and markdown files separately
        documents = []
        
        # Load .txt files
        try:
            txt_loader = DirectoryLoader(
                self.data_folder,
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={'encoding': 'utf-8'},
                show_progress=True,
                silent_errors=True
            )
            txt_docs = txt_loader.load()
            documents.extend(txt_docs)
            print(f"Loaded {len(txt_docs)} .txt files")
        except Exception as e:
            print(f"Error loading .txt files: {e}")
        
        # Load .md files
        try:
            md_loader = DirectoryLoader(
                self.data_folder,
                glob="**/*.md",
                loader_cls=TextLoader,
                loader_kwargs={'encoding': 'utf-8'},
                show_progress=True,
                silent_errors=True
            )
            md_docs = md_loader.load()
            documents.extend(md_docs)
            print(f"Loaded {len(md_docs)} .md files")
        except Exception as e:
            print(f"Error loading .md files: {e}")
        
        print(f"Total documents loaded: {len(documents)}")
        
        if len(documents) == 0:
            print(f"⚠️  No documents found in '{self.data_folder}' folder!")
            print("Please add .txt or .md files to the folder and restart the application.")
            return
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        splits = text_splitter.split_documents(documents)
        print(f"Split into {len(splits)} chunks")
        
        # Annotate chunks for better citation metadata
        for idx, doc in enumerate(splits, start=1):
            doc.metadata["chunk_index"] = idx
            source_path = doc.metadata.get("source", "Unknown")
            if source_path != "Unknown":
                try:
                    rel_path = os.path.relpath(source_path, start=self.data_folder)
                except ValueError:
                    rel_path = os.path.basename(source_path)
                if rel_path.startswith(".."):
                    rel_path = os.path.basename(source_path)
                doc.metadata["source_name"] = rel_path
            else:
                doc.metadata["source_name"] = "Unknown"

        if len(splits) == 0:
            print("⚠️  No text chunks created. Documents might be empty.")
            return
        
        # Create vector store
        print("Creating vector store...")
        self.vectorstore = FAISS.from_documents(splits, self.embeddings)
        index_path = os.path.join(self.vector_dir, "faiss_index")
        self.vectorstore.save_local(index_path)
        print("Vector store created and cached successfully!")
        
    def retrieve_relevant_docs(self, query: str, k: int = 5):
        """Retrieve relevant documents for the query"""
        if not self.vectorstore:
            return []
        
        docs = self.vectorstore.similarity_search(query, k=k)
        return docs
    
    def format_context(self, docs):
        """Format retrieved documents as context"""
        context = ""
        sources = []
        
        for i, doc in enumerate(docs, start=1):
            source_name = doc.metadata.get("source_name") or os.path.basename(doc.metadata.get("source", "Unknown"))
            page = doc.metadata.get("page")
            chunk_index = doc.metadata.get("chunk_index")
            location_parts = []
            if page is not None:
                location_parts.append(f"page {page}")
            if chunk_index is not None:
                location_parts.append(f"chunk {chunk_index}")
            location_suffix = f" ({', '.join(location_parts)})" if location_parts else ""

            context += f"\n[Source {i} | {source_name}{location_suffix}]: {doc.page_content}\n"
            sources.append(
                {
                    "label": f"Source {i}",
                    "name": source_name,
                    "page": page,
                    "chunk": chunk_index,
                    "raw_metadata": doc.metadata,
                }
            )
        
        return context, sources
    
    def answer_question(self, query: str, chat_history: List[Tuple[str, str]] = None):
        """Answer a question using RAG with Cohere"""
        
        # Check if vectorstore is initialized
        if not self.vectorstore:
            return "⚠️ No documents loaded. Please add .txt or .md files to the 'data' folder and restart the application.", []
        
        # Retrieve relevant documents
        relevant_docs = self.retrieve_relevant_docs(query)
        
        if not relevant_docs:
            return "I couldn't find any relevant information in the database.", []
        
        # Format context and sources
        context, sources = self.format_context(relevant_docs)
        
        # Build conversation history
        conversation_history = []
        if chat_history:
            for user_msg, bot_msg in chat_history:
                conversation_history.append({"role": "USER", "message": user_msg})
                conversation_history.append({"role": "CHATBOT", "message": bot_msg})
        
        # Create prompt with context
        prompt = f"""Based on the following context from Vedic scriptures, please answer the question. 
Always cite your sources by mentioning which source number you're referencing.

Context:
{context}

Question: {query}

Please provide a detailed answer and cite the specific sources you used."""

        # Get response from Cohere
        try:
            if COHERE_API_KEY == "your-cohere-api-key-here":
                return "⚠️ Cohere API key missing. Update the .env file to enable responses.", sources
            response = co.chat(
                message=prompt,
                chat_history=conversation_history,
                model=COHERE_CHAT_MODEL,
                temperature=0.3,
            )
            
            answer = response.text
            
            # Format sources
            unique_sources = []
            seen = set()
            for info in sources:
                key = (info.get("name"), info.get("page"), info.get("chunk"))
                if key in seen:
                    continue
                unique_sources.append(info)
                seen.add(key)

            sources_text = "\n\n**Sources:**\n"
            for i, info in enumerate(unique_sources, 1):
                name = info.get("name", "Unknown source")
                page = info.get("page")
                chunk = info.get("chunk")
                location_parts = []
                if page is not None:
                    location_parts.append(f"page {page}")
                if chunk is not None:
                    location_parts.append(f"chunk {chunk}")
                location = f" — {', '.join(location_parts)}" if location_parts else ""
                sources_text += f"- [{i}] {name}{location}\n"
            
            full_response = answer + sources_text
            
            return full_response, unique_sources
            
        except Exception as e:
            return f"Error generating response: {str(e)}", []

# Initialize RAG system
print("Initializing RAG system...")
rag_system = RAGSystem()

def chat_interface(message, history):
    """Gradio chat interface"""
    # Ensure history is a list
    if history is None:
        history = []
    
    # Get response from RAG system
    response, sources = rag_system.answer_question(message, history)
    
    # Append the new message and response to history
    history.append((message, response))
    
    return history

# Create Gradio interface
with gr.Blocks(title="VedicGPT - RAG System", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🕉️ VedicGPT - Vedic Knowledge Assistant
        
        Ask questions about Vedic scriptures and get answers with source citations.
        """
    )
    
    chatbot = gr.Chatbot(
        height=500,
        bubble_full_width=False,
        show_label=False,
    )
    
    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ask a question about Vedic scriptures...",
            show_label=False,
            scale=9
        )
        submit = gr.Button("Send", scale=1, variant="primary")
    
    with gr.Row():
        clear = gr.Button("Clear Chat")
    
    gr.Markdown(
        """
        ### About
        This system uses Retrieval-Augmented Generation (RAG) to answer questions based on Vedic texts.
        All answers are sourced from the documents in the data folder and include citations.
        """
    )
    
    # Event handlers
    msg.submit(chat_interface, [msg, chatbot], [chatbot], queue=True)
    submit.click(chat_interface, [msg, chatbot], [chatbot], queue=True)
    msg.submit(lambda: "", None, [msg], queue=False)
    submit.click(lambda: "", None, [msg], queue=False)
    clear.click(lambda: [], None, [chatbot], queue=False)

if __name__ == "__main__":
    print("\n✅ RAG System initialized successfully!")
    print("🚀 Starting Gradio interface...")
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
