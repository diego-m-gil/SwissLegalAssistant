#!/usr/bin/env python3
"""
CM_PrivLaw RAG Index Builder

Extracts text from PDFs, chunks with overlap, embeds via OpenAI, stores in FAISS.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import tiktoken
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LTTextContainer, LTChar, LTFigure
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from rich.console import Console
from rich.progress import Progress, TaskID

console = Console()


def extract_pdf_to_text(pdf_path: Path, text_dir: Path) -> Optional[Path]:
    """Extract text from PDF and save to text file with enhanced extraction."""
    text_file = text_dir / f"{pdf_path.stem}.txt"
    
    if text_file.exists():
        console.print(f"[yellow]Skipping {pdf_path.name} - text already exists[/yellow]")
        return text_file
    
    try:
        # Enhanced text extraction with better layout analysis
        text = extract_text(str(pdf_path), laparams=LAParams(
            boxes_flow=0.5,
            word_margin=0.1,
            char_margin=2.0,
            line_margin=0.5
        ))
        
        # Clean up text
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        
        # Try to extract any figure captions or image-related text
        try:
            with open(pdf_path, 'rb') as file:
                rsrcmgr = PDFResourceManager()
                laparams = LAParams(boxes_flow=0.5, word_margin=0.1)
                device = PDFPageAggregator(rsrcmgr, laparams=laparams)
                interpreter = PDFPageInterpreter(rsrcmgr, device)
                
                image_text = []
                for page in PDFPage.get_pages(file):
                    interpreter.process_page(page)
                    layout = device.get_result()
                    
                    for element in layout:
                        if isinstance(element, LTFigure):
                            # Extract any text near figures (captions, etc.)
                            for subelement in element:
                                if isinstance(subelement, LTTextContainer):
                                    image_text.append(subelement.get_text().strip())
                
                if image_text:
                    text += "\n\n" + "\n".join(image_text)
        except:
            pass  # Continue with basic text if image extraction fails
        
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        console.print(f"[green]✓[/green] Extracted {pdf_path.name}")
        return text_file
    
    except Exception as e:
        console.print(f"[red]✗ Failed to extract {pdf_path.name}: {e}[/red]")
        return None


def estimate_page_from_position(text: str, chunk_start: int) -> int:
    """Estimate page number based on character position (rough approximation)."""
    # Very rough estimate: ~2000 chars per page
    return max(1, chunk_start // 2000 + 1)


def chunk_text(text_file: Path, chunk_dir: Path, chunk_size: int, chunk_overlap: int) -> List[Path]:
    """Split text into overlapping chunks and save as JSONL."""
    chunk_file = chunk_dir / f"{text_file.stem}_chunks.jsonl"
    
    if chunk_file.exists():
        console.print(f"[yellow]Skipping {text_file.name} - chunks already exist[/yellow]")
        return [chunk_file]
    
    try:
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Use tiktoken for accurate token counting
        encoding = tiktoken.get_encoding("cl100k_base")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=lambda x: len(encoding.encode(x))
        )
        
        chunks = splitter.split_text(text)
        
        with open(chunk_file, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks):
                # Find approximate position in original text for page estimation
                chunk_start = text.find(chunk[:100]) if len(chunk) > 100 else text.find(chunk)
                page_num = estimate_page_from_position(text, chunk_start)
                
                chunk_data = {
                    "chunk_id": f"{text_file.stem}_{i:03d}",
                    "text": chunk,
                    "metadata": {
                        "source": text_file.stem,
                        "chunk_index": i,
                        "page": page_num
                    }
                }
                f.write(json.dumps(chunk_data, ensure_ascii=False) + '\n')
        
        console.print(f"[green]✓[/green] Chunked {text_file.name} → {len(chunks)} chunks")
        return [chunk_file]
    
    except Exception as e:
        console.print(f"[red]✗ Failed to chunk {text_file.name}: {e}[/red]")
        return []


def build_faiss_index(chunk_files: List[Path], index_dir: Path) -> bool:
    """Build FAISS vector store from chunk files."""
    index_path = index_dir / "faiss_index"
    
    if index_path.exists():
        console.print("[yellow]FAISS index already exists. Delete index/ folder to rebuild.[/yellow]")
        return True
    
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]ERROR: OPENAI_API_KEY environment variable not set![/red]")
        console.print("Set it with: $env:OPENAI_API_KEY='your-key-here'")
        return False
    
    try:
        # Load all chunks
        documents = []
        
        for chunk_file in chunk_files:
            with open(chunk_file, 'r', encoding='utf-8') as f:
                for line in f:
                    chunk_data = json.loads(line.strip())
                    doc = Document(
                        page_content=chunk_data["text"],
                        metadata=chunk_data["metadata"]
                    )
                    documents.append(doc)
        
        console.print(f"[blue]Loaded {len(documents)} chunks total[/blue]")
        
        # Create embeddings with minimal configuration
        import openai
        openai.api_key = api_key
        
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )
        
        # Build FAISS index
        with console.status("[bold green]Building FAISS index..."):
            vectorstore = FAISS.from_documents(documents, embeddings)
            vectorstore.save_local(str(index_path))
        
        console.print(f"[green]✓ Built FAISS index with {len(documents)} chunks[/green]")
        return True
    
    except Exception as e:
        console.print(f"[red]✗ Failed to build FAISS index: {e}[/red]")
        return False


def main():
    parser = argparse.ArgumentParser(description="Build RAG index from PDFs")
    parser.add_argument("--pdf_dir", default="pdf", help="PDF directory")
    parser.add_argument("--text_dir", default="text", help="Text output directory")
    parser.add_argument("--chunk_dir", default="chunks", help="Chunk output directory")
    parser.add_argument("--index_dir", default="index", help="Index output directory")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Chunk size in tokens")
    parser.add_argument("--chunk_overlap", type=int, default=200, help="Chunk overlap in tokens")
    
    args = parser.parse_args()
    
    # Convert to Path objects
    pdf_dir = Path(args.pdf_dir)
    text_dir = Path(args.text_dir)
    chunk_dir = Path(args.chunk_dir)
    index_dir = Path(args.index_dir)
    
    # Verify directories
    if not pdf_dir.exists():
        console.print(f"[red]PDF directory {pdf_dir} does not exist![/red]")
        return
    
    # Create output directories
    text_dir.mkdir(exist_ok=True)
    chunk_dir.mkdir(exist_ok=True)
    index_dir.mkdir(exist_ok=True)
    
    console.print("[bold blue]CM_PrivLaw RAG Index Builder[/bold blue]")
    console.print(f"PDF dir: {pdf_dir}")
    console.print(f"Chunk size: {args.chunk_size} tokens, overlap: {args.chunk_overlap}")
    
    # Find all PDFs
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        console.print(f"[red]No PDF files found in {pdf_dir}[/red]")
        return
    
    console.print(f"Found {len(pdf_files)} PDF files")
    
    # Phase 1: Extract text from PDFs
    console.print("\n[bold]Phase 1: Extracting text from PDFs[/bold]")
    text_files = []
    
    with Progress() as progress:
        task = progress.add_task("Extracting...", total=len(pdf_files))
        
        for pdf_file in pdf_files:
            text_file = extract_pdf_to_text(pdf_file, text_dir)
            if text_file:
                text_files.append(text_file)
            progress.advance(task)
    
    # Phase 2: Chunk texts
    console.print("\n[bold]Phase 2: Chunking texts[/bold]")
    chunk_files = []
    
    with Progress() as progress:
        task = progress.add_task("Chunking...", total=len(text_files))
        
        for text_file in text_files:
            chunk_file_list = chunk_text(text_file, chunk_dir, args.chunk_size, args.chunk_overlap)
            chunk_files.extend(chunk_file_list)
            progress.advance(task)
    
    # Phase 3: Build FAISS index
    console.print("\n[bold]Phase 3: Building FAISS index[/bold]")
    success = build_faiss_index(chunk_files, index_dir)
    
    if success:
        console.print("\n[bold green]✅ RAG index build complete![/bold green]")
        console.print(f"Text files: {len(text_files)}")
        console.print(f"Chunk files: {len(chunk_files)}")
        console.print("Ready to use ask.py!")
    else:
        console.print("\n[bold red]❌ Build failed[/bold red]")


if __name__ == "__main__":
    main()
