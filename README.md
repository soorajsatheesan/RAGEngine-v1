# RAGEngine-v1

A lightweight, efficient Retrieval-Augmented Generation (RAG) engine built with open-source technologies. RAGEngine-v1 enables you to build intelligent knowledge bases using RAG technology, transforming your documents into searchable, queryable knowledge repositories.

## 🎯 Overview

RAGEngine-v1 is a simple yet powerful RAG pipeline that enables users to upload documents (PDF or TXT), process them into searchable vector embeddings, and query them using natural language. The system combines document processing, vector embeddings, and LLM-powered question answering to deliver accurate, context-aware responses.

This project provides a **solid foundation** for building knowledge base systems using RAG. It can be extended with multi-user support, workspace management, advanced retrieval features, and production-ready infrastructure to create comprehensive knowledge management solutions.

## 🏗️ Architecture

The following diagram illustrates the complete RAG pipeline architecture:

![RAGEngine-v1 Architecture](Design.png)

### Core Components

1. **Document Preprocessing** (`preprocess.py`)
   - Converts PDF files to text format
   - Splits documents into manageable chunks (1000 tokens with 200 token overlap)
   - Ensures continuity between chunks for better context preservation

2. **Vector Embeddings** (`embedding.py`)
   - Uses Sentence Transformers (`all-MiniLM-L6-v2`) for generating embeddings
   - Stores embeddings in ChromaDB for efficient similarity search
   - Provides interface for document and query embedding

3. **Document Ingestion** (`ingestion.py`)
   - Orchestrates the complete ingestion pipeline
   - Handles document conversion, chunking, and vector storage
   - Returns a retriever for query operations

4. **LLM Integration** (`llm.py`)
   - Integrates with Google Gemini 2.5 Flash for answer generation
   - Implements memory management for query caching
   - Provides fallback mechanisms for queries without relevant documents

5. **User Interface** (`main.py`)
   - Streamlit-based web interface
   - Document upload and management
   - Interactive Q&A chat interface

## 🔄 How It Works

### Document Processing Pipeline

1. **Upload**: User uploads a PDF or TXT file through the Streamlit interface
2. **Conversion**: PDF files are converted to plain text using `pdfplumber`
3. **Chunking**: Documents are split into chunks of 1000 tokens with 200 token overlap
4. **Embedding**: Each chunk is converted to a vector embedding using Sentence Transformers
5. **Storage**: Embeddings are stored in ChromaDB with metadata for retrieval

### Query Processing Pipeline

1. **Memory Check**: System first checks if a similar query was answered before
2. **Query Embedding**: If not in memory, the query is embedded using the same model
3. **Similarity Search**: ChromaDB performs cosine similarity search to find relevant chunks
4. **Context Retrieval**: Most relevant document chunks are retrieved
5. **Answer Generation**: 
   - If relevant documents found: LLM generates answer using retrieved context
   - If no relevant documents: LLM provides general answer without context
6. **Memory Storage**: Query-response pair is stored for future reference

## 🚀 Features

- ✅ **Document Support**: PDF and TXT file formats
- ✅ **Intelligent Chunking**: Overlapping chunks for context preservation
- ✅ **Vector Search**: Semantic similarity search using ChromaDB
- ✅ **Memory Management**: Query caching to avoid redundant processing
- ✅ **Fallback Mechanism**: General answers when documents don't contain relevant information
- ✅ **User-Friendly UI**: Simple Streamlit interface for easy interaction
- ✅ **Open-Source**: Built entirely with open-source technologies

## 📋 Prerequisites

- Python 3.8 or higher
- Google Gemini API key (get one from [Google AI Studio](https://makersuite.google.com/app/apikey))

## 🛠️ Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/RAGEngine-v1.git
cd RAGEngine-v1
```

### Step 2: Install Dependencies

```bash
pip install langchain langchain-community langchain-google-genai langchain-chroma pdfplumber sentence-transformers streamlit
```

Or create a `requirements.txt` and install:

```bash
pip install -r requirements.txt
```

### Step 3: Set Up API Key

Create a file named `api_key.txt` in the project root and add your Google Gemini API key:

```
your-gemini-api-key-here
```

Alternatively, you can enter the API key through the Streamlit interface on first run.

## 🎮 Usage

### Starting the Application

```bash
streamlit run main.py
```

The application will open in your default web browser at `http://localhost:8501`

### Using the Application

1. **Upload a Document**: 
   - Use the sidebar to upload a PDF or TXT file
   - Wait for the ingestion process to complete

2. **Ask Questions**:
   - Type your question in the query input field
   - Click "Submit Query" to get an answer
   - The system will search through your documents and provide context-aware answers

3. **View History**:
   - Your query history is displayed below the input field
   - Most recent queries appear at the top

## 📁 Project Structure

```
RAGEngine-v1/
├── main.py              # Streamlit UI and main application entry point
├── ingestion.py         # Document ingestion pipeline orchestration
├── preprocess.py        # PDF conversion and text chunking
├── embedding.py         # Vector embedding generation and ChromaDB management
├── llm.py              # LLM integration and query processing
├── api_key.txt         # Gemini API key (create this file)
├── database/           # ChromaDB storage directory (created automatically)
├── documents/          # Processed text documents (created automatically)
├── temp/              # Temporary file storage (created automatically)
├── Design.png         # Architecture diagram
├── README.md          # This file
└── LICENSE            # MIT License
```

## 🔧 Configuration

### Embedding Model

Default model: `all-MiniLM-L6-v2`

To change the embedding model, modify `embedding.py`:

```python
class SentenceTransfromerEmbeddings:
    def __init__(self, model_name="your-preferred-model"):
        self.model = SentenceTransformer(model_name)
```

### Chunking Parameters

Default: 1000 tokens chunk size, 200 tokens overlap

To modify, edit `preprocess.py`:

```python
def load_and_split_single_file(
    file_path: str,
    chunk_size: int = 1000,      # Modify this
    chunk_overlap: int = 200,    # Modify this
) -> List[str]:
```

### LLM Model

Default: `gemini-2.5-flash`

To change, modify `llm.py`:

```python
def initialize_qa_chain(
    retriever=None,
    model="your-preferred-model",  # Modify this
    ...
):
```

## 🧪 Technology Stack

- **LangChain**: Document processing and RAG orchestration
- **ChromaDB**: Vector database for embedding storage
- **Sentence Transformers**: Open-source embedding models
- **Google Gemini 2.5 Flash**: Large Language Model for answer generation
- **Streamlit**: Web application framework
- **pdfplumber**: PDF text extraction

## 🔮 Future Directions

RAGEngine-v1 is designed as a foundation for building comprehensive knowledge base systems. Potential extensions include:

- **Multi-User Support**: User authentication and authorization
- **Workspace Management**: Multiple document collections and organization
- **Advanced Retrieval**: Hybrid search, re-ranking, query expansion
- **Production Infrastructure**: REST API, microservices, containerization
- **Enhanced Features**: Document versioning, analytics, integrations
- **Advanced Security**: SSO, compliance features, audit logs, enhanced security controls

## 🤝 Contributing

Contributions are welcome! This is a foundational project, and improvements to the core RAG pipeline, documentation, or additional features are appreciated.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [LangChain](https://www.langchain.com/)
- Vector storage powered by [ChromaDB](https://www.trychroma.com/)
- Embeddings generated using [Sentence Transformers](https://www.sbert.net/)
- LLM powered by [Google Gemini](https://deepmind.google/technologies/gemini/)


**Note**: This is version 1 (v1) of RAGEngine, serving as the core foundation for building knowledge base systems using RAG technology. Future enhanced versions with additional features can be built upon this base.
