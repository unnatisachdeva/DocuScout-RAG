# 📚 AskMyPDF-RAG

A **Retrieval-Augmented Generation (RAG) based PDF chatbot** that allows users to upload a PDF and ask questions about its contents.

The application uses **LangChain** to build the RAG pipeline, **Gemini** to generate embeddings and answers, **ChromaDB** as the vector database, and **Streamlit** for the user interface.

## ✨ Features

* 📄 Upload any PDF through the Streamlit UI
* ✂️ Split PDF content into smaller chunks
* 🧠 Generate embeddings using Google's Gemini embedding model
* 🗄️ Store embeddings in ChromaDB
* 🔎 Retrieve relevant document chunks using MMR search
* 🤖 Generate answers using Gemini
* 🛡️ Restrict answers to information available in the uploaded document
* 💬 Interactive chat interface with conversation history

## 🏗️ RAG Architecture

```text
                    User
                     │
                     ▼
              Upload PDF
                     │
                     ▼
              PyPDFLoader
                     │
                     ▼
            Recursive Text Splitter
                     │
                     ▼
             Document Chunks
                     │
                     ▼
          Gemini Embedding Model
                     │
                     ▼
                 ChromaDB
                     │
                     │
User Question ───────┘
       │
       ▼
Query Embedding
       │
       ▼
MMR Retriever
       │
       ▼
Relevant Document Chunks
       │
       ▼
    Prompt + Context
       │
       ▼
   Gemini LLM
       │
       ▼
      Answer
```

## 🛠️ Technologies Used

* **Python**
* **Streamlit** – User interface
* **LangChain** – RAG framework
* **Google Gemini** – Embeddings and LLM
* **ChromaDB** – Vector database
* **PyPDF** – PDF loading
* **RecursiveCharacterTextSplitter** – Document chunking

## 📁 Project Structure

```text
DocuMind-RAG/
│
├── RagUI.py              # Streamlit user interface
├── main.py               # RAG retrieval and answer generation
├── ingestion.py          # PDF loading, chunking and vector DB creation
├── requirements.txt      # Python dependencies
├── .gitignore            # Files excluded from Git
│
└── document loader/      # Local PDF files used for testing
```

## ⚙️ How It Works

### 1. PDF Upload

The user uploads a PDF through the Streamlit interface.

### 2. Document Loading

`PyPDFLoader` extracts the text from the uploaded PDF.

### 3. Chunking

The extracted text is divided into smaller chunks using `RecursiveCharacterTextSplitter`.

Current configuration:

```python
chunk_size=1000
chunk_overlap=200
```

### 4. Embeddings

Each chunk is converted into a numerical vector using:

```text
gemini-embedding-001
```

### 5. Vector Storage

The embeddings and document content are stored in **ChromaDB**.

### 6. Retrieval

When the user asks a question, the question is converted into an embedding and ChromaDB retrieves the most relevant chunks using **Maximal Marginal Relevance (MMR)**.

### 7. Answer Generation

The retrieved chunks are provided as context to Gemini.

The model is instructed to use **only the provided document context** and not rely on its general knowledge.

If the information cannot be found in the document, the chatbot responds:

```text
I could not find this information in the document.
```

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/DocuMind-RAG.git
cd DocuMind-RAG
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 🔑 API Key Setup

Create a `.env` file in the project root:

```text
GOOGLE_API_KEY="your_gemini_api_key"
```

**Never commit your `.env` file or API key to GitHub.**

The `.env` file is included in `.gitignore`.

## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run RagUI.py
```

Then open the local Streamlit URL shown in your terminal.

Upload a PDF and start asking questions.

## 💡 Example

For a PDF about **Convolutional Neural Networks (CNNs)**:

```text
User:
What is CNN?

AI:
CNN is a deep learning model used for image classification
and image recognition...
```

If the user asks something that isn't covered by the PDF:

```text
User:
What is ANN?

AI:
I could not find this information in the document.
```

## 🔒 Security

The following files are intentionally excluded from Git:

```text
.env
.venv/
*.pdf
chroma_db/
__pycache__/
.DS_Store
```

This prevents API keys, local environments, uploaded documents, and generated vector databases from being accidentally committed.

## 🔮 Future Improvements

* Support multiple PDFs simultaneously
* Add document/source citations to answers
* Add PDF page references
* Improve conversation memory
* Add document management
* Add streaming responses
* Deploy the application online
* Add authentication
* Support additional embedding providers


