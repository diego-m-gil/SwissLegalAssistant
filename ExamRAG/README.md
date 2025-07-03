# Swiss Legal Research Assistant

A retrieval-augmented generation system for Swiss and EU legal research using OpenAI embeddings and FAISS vector search.

## 🚀 Quick Start

### Option 1: Easy Setup & GUI

```powershell
# 1. Run the setup script
python setup.py

# 2. Configure your API key in .env file (copy from .env.template)
# Edit the .env file with your OpenAI API key

# 3. Build the index (if needed)
python scripts/build_index.py

# 4. Launch the GUI
python run_gui.py
# OR double-click start_gui.bat on Windows
```

### Option 2: Manual Setup

```powershell
# Create virtual environment
python -m venv .venv

# Activate environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Create .env file from template and add your API key
cp .env.template .env
# Edit the .env file with your favorite text editor
```

### 3. Build the RAG Index

```powershell
# Extract PDFs, chunk text, and build FAISS index
python scripts\build_index.py
```

Expected output:
```
Swiss Legal Research Assistant Index Builder
PDF dir: pdf
Found 21 PDF files

Phase 1: Extracting text from PDFs
✓ Extracted Handout #1 Introduction to Law.pdf
...

Phase 2: Chunking texts
✓ Chunked Handout #1 Introduction to Law.txt → 45 chunks
...

Phase 3: Building FAISS index
✅ RAG index build complete!
Text files: 21
Chunk files: 21
Ready to use!
```

### 4. Use the System

#### GUI Interface (Recommended)
```powershell
# Launch the web-based GUI
python run_gui.py
# OR double-click start_gui.bat on Windows
```

The GUI will open in your browser at `http://localhost:8501` with features:
- 🎯 Simple question input with example questions
- ⚙️ Adjustable number of retrieved chunks with optimization guidance
- 📚 Automatic citation formatting
- 📄 Optional context viewing
- 💰 Token usage and cost tracking
- 🔍 Real-time search and answers

#### Command Line Interface
```powershell
# Ask questions directly from terminal
python scripts\ask.py "What are the basic requirements for trademark protection in Switzerland?"
```

## Example Queries & Expected Format

**Query:** "List the basic requirements for trademark protection in Switzerland."

**Expected Answer:**
• Distinctive sign of goods/services (Handout #4 p. 3)
• Not generic or descriptive (Handout #4 p. 4)  
• Confers exclusive right upon IPI registration (Handout #4 p. 5)

**References:** (Handout #4 p. 3) (Handout #4 p. 4) (Handout #4 p. 5)

## Project Structure

```
SwissLegalResearchAssistant/
├── pdf/              # Original PDF files (21 documents)
├── text/             # Extracted text files (.txt)
├── chunks/           # JSON chunk files (~1000 tokens, 200 overlap)
├── index/            # FAISS vector store + metadata
├── scripts/
│   ├── build_index.py    # Build the RAG pipeline
│   ├── ask.py            # CLI query interface
│   └── streamlit_app.py  # Web GUI interface
├── requirements.txt  # Python dependencies
├── .env.template     # Template for API keys
├── .gitignore        # Git ignore file
└── README.md         # This file
```

## Command Reference

### build_index.py Options

```powershell
python scripts\build_index.py --help

# Custom directories
python scripts\build_index.py --pdf_dir custom_pdf --text_dir custom_text

# Custom chunk settings
python scripts\build_index.py --chunk_size 800 --chunk_overlap 150
```

### ask.py Options

```powershell
python scripts\ask.py --help

# Retrieve more context
python scripts\ask.py "question" --k 10

# Show retrieved context
python scripts\ask.py "question" --verbose
```



## Troubleshooting

### "FAISS index not found"
Run: `python scripts\build_index.py`

### "OPENAI_API_KEY not set"
Check that your `.env` file contains your API key

### "No PDF files found"
Ensure PDFs are in the `pdf/` directory

### Large output / Token limits
Use shorter, more specific questions

## Technical Details

- **Chunking:** 1000 tokens with 200-token overlap using tiktoken
- **Embeddings:** OpenAI text-embedding-3-small
- **Vector Store:** FAISS with cosine similarity
- **LLM:** GPT-4o (default) with temperature=0 for deterministic answers
- **Citations:** Automatic extraction from document metadata

## License

This project is released under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
