# Swiss Legal Research Assistant

A retrieval-augmented generation system for Swiss and EU legal research using OpenAI embeddings and FAISS vector search. This tool helps legal professionals, students, and researchers efficiently find relevant information across legal codes, case law, and academic materials.

> **Important**: This system requires you to add your own PDF files to the `pdf/` directory. The repository does not include any copyrighted legal materials or course content.

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

### 3. Add Your Own PDF Files

> **IMPORTANT:** Due to copyright restrictions, you must add your own legal documents to use this system.

1. Place your legal PDFs in the `pdf/` directory:
   - Legal codes (e.g., Swiss Civil Code, Code of Obligations)
   - Course materials
   - Legal textbooks
   - Case law documents
   - Any other relevant legal materials

2. Build the RAG Index:

```powershell
# Extract PDFs, chunk text, and build FAISS index
python scripts\build_index.py
```

Example output:
```text
Swiss Legal Research Assistant Index Builder
PDF dir: pdf
Found 5 PDF files

Phase 1: Extracting text from PDFs
✓ Extracted Swiss_Civil_Code.pdf
✓ Extracted Code_of_Obligations.pdf
...

Phase 2: Chunking texts
✓ Chunked Swiss_Civil_Code.txt → 78 chunks
...

Phase 3: Building FAISS index
✅ RAG index build complete!
Text files: 5
Chunk files: 5
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

- 🎯 Simple question input with curated example questions
- ⚙️ Adjustable number of retrieved chunks with research-backed optimization guidance
- 📚 Sophisticated citation formatting with categorized sources (Legal Codes, Course Materials, etc.)
- 📄 Source exploration via tabbed interface with context snippets
- 💰 Token usage and estimated cost tracking
- 🔍 Real-time search with clear source attribution
- ⚠️ Fallback handling when no relevant sources are found

#### Command Line Interface

```powershell
# Ask questions directly from terminal
python scripts\ask.py "What are the basic requirements for trademark protection in Switzerland?"
```

## Example Queries & Expected Format

**Query:** "List the basic requirements for trademark protection in Switzerland."

**Expected Answer:**

```text
• Distinctive sign of goods/services (Handout #4 p. 3)
• Not generic or descriptive (Handout #4 p. 4)  
• Confers exclusive right upon IPI registration (Handout #4 p. 5)
```

**Sources Tab:** Organized by category (Legal Codes, Course Materials), with context snippets and page references.

**In-text Citations:** Displayed in a subtle grey styling for readability while maintaining source tracking.

## Project Structure

```
SwissLegalResearchAssistant/
├── pdf/              # Add your legal PDF files here (not included in repo)
├── text/             # Extracted text files (.txt) - generated automatically
├── chunks/           # JSON chunk files (~1000 tokens, 200 overlap) - generated automatically
├── index/            # FAISS vector store + metadata - generated from your documents
├── scripts/
│   ├── build_index.py    # Build the RAG pipeline
│   ├── ask.py            # CLI query interface
│   └── streamlit_app.py  # Web GUI interface
├── requirements.txt  # Python dependencies
├── .env.template     # Template for API keys
├── .gitignore        # Git ignore file (excludes pdf/text/chunks/index contents)
└── README.md         # This file
```

> **Note**: The `pdf/`, `text/`, `chunks/`, and `index/` directories are empty in the repository. You need to add your own PDF files to the `pdf/` directory and run the index builder to generate the other files.

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



## Security Best Practices

This application uses secure API key management:

- **Environment Variables:** API keys are stored in a local `.env` file that is never committed to git
- **Secure Input:** When keys are missing, they can be entered via a password field in the UI
- **Template Sharing:** Use `.env.template` when sharing the project, never share your actual `.env` file
- **Git Protection:** The `.gitignore` file prevents accidental key exposure

## Troubleshooting

### "FAISS index not found"

Run: `python scripts\build_index.py` to create the vector index

### "OPENAI_API_KEY not set"

Create a `.env` file from `.env.template` and add your OpenAI API key

### "No PDF files found"

This project does not include PDF files. You need to add your own legal documents to the `pdf/` directory before building the index. Alternatively, specify a custom directory with `--pdf_dir`.

### Large output / Token limits

Use shorter, more specific questions or increase the max_tokens setting in the code

### "No relevant sources found"

- Try increasing the number of retrieved chunks (k value)
- Rephrase your question to use terminology from the legal texts
- Check that relevant documents are included in your PDF collection

## Technical Details

- **Chunking:** 1000 tokens with 200-token overlap using tiktoken (optimized for legal text retention)
- **Embeddings:** OpenAI text-embedding-3-small (optimized for multilingual legal content)
- **Vector Store:** FAISS with cosine similarity for fast, efficient similarity search
- **LLM:** GPT-4o (default) with temperature=0 for deterministic, factual answers
- **Citations:** Sophisticated extraction from document metadata with legal code mapping
- **Contextual Retrieval:** Research-backed k=5 default with adjustable settings based on query complexity
- **No-Source Handling:** Clear disclaimers when no relevant sources are found
- **Source Categorization:** Automatic grouping by document type (Legal Codes, Course Materials, etc.)

## Project Evolution

This project began as an exam preparation assistant for law students and has evolved into a comprehensive legal research tool. Key improvements include:

- Enhanced UI/UX with modern design and intuitive controls
- Secure API key management with environment variables
- Sophisticated citation handling with legal code mapping
- Source categorization and exploration via tabs
- Research-backed chunk retrieval optimization
- Robust error handling for missing sources
- Token usage tracking and cost estimation

## Copyright and License

This project is released under the MIT License. The code and infrastructure are open source, but:

- **Important Copyright Notice**: This repository does NOT include any copyrighted legal materials or course content.
- **Content Responsibility**: Users are responsible for ensuring they have the proper rights to use any PDFs they add to the system.
- **Educational Use**: If you're using this for academic purposes, ensure you comply with your institution's policies regarding the use of course materials.

## Contributing

Contributions to the code and infrastructure are welcome! Please feel free to submit a Pull Request on GitHub.

## GitHub Repository

Find the complete code and documentation at:
[github.com/diego-m-gil/swiss-legal-research-assistant](https://github.com/diego-m-gil/swiss-legal-research-assistant)
