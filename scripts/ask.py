#!/usr/bin/env python3
"""
Simple CM_PrivLaw Exam Assistant - Works with LangChain 0.2.1

Usage: python ask_simple.py "Your question here"
"""

import argparse
import os
import sys
from pathlib import Path

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from rich.console import Console

console = Console()

def load_vectorstore(index_dir: Path) -> FAISS:
    """Load the FAISS vector store."""
    index_path = index_dir / "faiss_index"
    
    if not index_path.exists():
        console.print(f"[red]FAISS index not found at {index_path}[/red]")
        console.print("Run: python scripts/build_index.py")
        sys.exit(1)
    
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)
        return vectorstore
    except Exception as e:
        console.print(f"[red]Failed to load FAISS index: {e}[/red]")
        sys.exit(1)


def format_citations(docs) -> str:
    """Extract and format unique citations from retrieved documents."""
    citations = set()
    
    for doc in docs:
        metadata = doc.metadata
        source = metadata.get('source', 'Unknown')
        page = metadata.get('page', '?')
        
        # Format citation based on source type
        if 'Handout' in source:
            citation = f"({source} p. {page})"
        elif source in ['OR', 'ZGB', 'URG', 'actual DSG', 'Criminal Law (english)', 
                       'Bundesgesetz über die Produktehaftpflicht']:
            citation = f"({source})"
        else:
            citation = f"({source} p. {page})"
        
        citations.add(citation)
    
    return " ".join(sorted(citations))


def answer_question(vectorstore: FAISS, question: str, k: int = 6):
    """Answer a question using simple retrieval + LLM."""
    
    # Get relevant documents
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(question)
    
    # Format context
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Create prompt
    prompt = f"""You are CM_PrivLaw Exam Assistant, a retrieval-augmented legal tutor.

Your mission during this 2-hour open-book exam:
1. Return concise, correct answers to Swiss/EU privacy-, IP-, contract-, liability- and AI-law questions
2. ALWAYS cite exact document + page or article number, e.g. (Handout #5 p. 12) or (OR Art. 101 II)
3. NEVER hallucinate an article or page; rely only on retrieved context

Style requirements:
- Answer in max 150 words
- Use bullet points if appropriate
- Include direct quotes only when essential
- English only
- No explanations about process
- If context insufficient: say "Insufficient context; try re-phrasing."

Context from course materials:
{context}

Question: {question}

Answer:"""
    
    # Get answer from LLM
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        max_tokens=256
    )
    
    response = llm.invoke(prompt)
    return response.content, docs


def main():
    parser = argparse.ArgumentParser(description="Query the CM_PrivLaw RAG system")
    parser.add_argument("question", help="Your legal question")
    parser.add_argument("--index_dir", default="index", help="FAISS index directory")
    parser.add_argument("--k", type=int, default=6, help="Number of chunks to retrieve")
    parser.add_argument("--verbose", action="store_true", help="Show retrieved context")
    
    args = parser.parse_args()
    
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]ERROR: OPENAI_API_KEY environment variable not set![/red]")
        console.print("Set it with: $env:OPENAI_API_KEY='your-key-here'")
        sys.exit(1)
    
    index_dir = Path(args.index_dir)
    
    # Load vector store
    vectorstore = load_vectorstore(index_dir)
    
    # Query the system
    try:
        with console.status("[bold green]Searching course materials..."):
            answer, docs = answer_question(vectorstore, args.question, args.k)
        
        # Print answer
        console.print(f"\n[bold blue]Question:[/bold blue] {args.question}")
        console.print(f"\n[bold green]Answer:[/bold green]")
        console.print(answer)
        
        # Print citations
        citations = format_citations(docs)
        if citations:
            console.print(f"\n[bold yellow]References:[/bold yellow]")
            console.print(citations)
        
        # Print retrieved context if verbose
        if args.verbose:
            console.print(f"\n[bold cyan]Retrieved Context ({len(docs)} chunks):[/bold cyan]")
            for i, doc in enumerate(docs, 1):
                metadata = doc.metadata
                console.print(f"\n[cyan]{i}. {metadata.get('source', 'Unknown')} (page {metadata.get('page', '?')}):[/cyan]")
                console.print(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
    
    except Exception as e:
        console.print(f"[red]Query failed: {e}[/red]")
        console.print(f"[red]Error details: {str(e)}[/red]")
        if args.verbose:
            import traceback
            console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
