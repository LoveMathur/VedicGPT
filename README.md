# VedicGPT - RAG-Based Question Answering System

A Retrieval-Augmented Generation (RAG) system that uses Cohere LLM to answer questions about Vedic scriptures with source citations.

## Features

- 📚 Loads all text/markdown files from the `data` folder
- 🔍 Uses FAISS for efficient vector search (cached on disk)
- 🧠 Supports Cohere or local SentenceTransformer embeddings
- 🤖 Powered by Cohere's Command-R-Plus model for answers
- 💬 Interactive chat interface with Gradio
- 📎 Automatic source citation for all answers
- 🕉️ Optimized for Vedic scripture queries

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get Cohere API Key

1. Visit [Cohere Dashboard](https://dashboard.cohere.com/)
2. Sign up or log in
3. Navigate to API Keys section
4. Copy your API key

### 3. Configure Environment

Edit the `.env` file and add your Cohere API key (needed for chat responses):

```
COHERE_API_KEY=your-actual-api-key-here
```

Optional environment variables:

```
# Use Cohere for embeddings (requires higher rate limits) or keep default local model
EMBEDDING_PROVIDER=local   # options: local, cohere
LOCAL_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
COHERE_CHAT_MODEL=command-r-plus-08-2024
COHERE_EMBED_MODEL=embed-english-v3.0
```

### 4. Add Your Data

Place your Vedic text files (`.txt` or `.md`) in the `data` folder. The system will automatically:
- Load all documents
- Split them into chunks
- Create a vector database
- Enable semantic search

## Usage

### Run the Application

```bash
python app.py
```

The application will:
1. Load all documents from the `data` folder
2. Create a vector store for semantic search
3. Launch a web interface at `http://localhost:7860`

### Chat Interface

1. Open your browser to `http://localhost:7860`
2. Type your question in the text box
3. Click "Send" or press Enter
4. Receive answers with source citations

### Example Questions

- "What is the meaning of dharma in the Bhagavad Gita?"
- "Explain the concept of karma"
- "What does the Gita say about meditation?"

## How It Works

1. **Document Loading**: All `.txt` and `.md` files from the `data` folder are loaded
2. **Text Chunking**: Documents are split into manageable chunks with overlap
3. **Embedding**: Each chunk is converted to vector embeddings (Cohere or local models)
4. **Vector Store**: FAISS indexes the embeddings for fast retrieval and is cached on disk
5. **Query Processing**: User questions are embedded and matched against the database
6. **Context Building**: Relevant chunks are retrieved and formatted
7. **LLM Response**: Cohere generates an answer based on the context
8. **Source Citation**: The response includes references to source documents

## Project Structure

```
VedicGPT/
├── app.py                 # Main RAG application
├── ocr.py                 # PDF to text conversion
├── data/                  # Your text documents
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (API keys)
└── README.md             # This file
```

## Technologies Used

- **Cohere**: LLM for question answering
- **LangChain**: Framework for RAG pipeline
- **FAISS**: Vector database for semantic search
- **Gradio**: Web interface for chat
- **PyMuPDF**: PDF processing
- **Tesseract**: OCR for scanned documents

## Tips

- For better results, ensure your documents are well-formatted
- Use the OCR script (`ocr.py`) to convert PDF files to text
- The system works best with clear, structured text
- Adjust `chunk_size` and `k` (number of retrieved docs) for optimization

## Troubleshooting

**Error: "No module named 'cohere'"**
```bash
pip install -r requirements.txt
```

**Error: "Invalid API key"**
- Check your `.env` file has the correct Cohere API key
- Ensure there are no extra spaces or quotes

**Error: "TooManyRequestsError" when building embeddings**
- Leave `EMBEDDING_PROVIDER=local` to avoid Cohere embedding rate limits
- Upgrade to a production Cohere key if you must use Cohere embeddings
- Delete the `storage/vectorstore` folder if you need to rebuild the cache

**No documents loaded**
- Verify files exist in the `data` folder
- Check file extensions are `.txt` or `.md`
- Ensure files have UTF-8 encoding

## License

MIT License