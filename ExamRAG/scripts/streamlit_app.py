import os
import sys
from pathlib import Path
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Configure page settings (must be the first Streamlit command)
st.set_page_config(
    page_title="Swiss Legal Research Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS to reduce font sizes globally
st.markdown("""
<style>
    /* Reduce font sizes across the app */
    html, body, [class*="css"] {
        font-size: 0.95rem !important;
    }
    h1 {
        font-size: 1.8rem !important;
    }
    h2 {
        font-size: 1.5rem !important;
    }
    h3 {
        font-size: 1.3rem !important;
    }
    .stButton button {
        font-size: 0.85rem !important;
    }
    .stTextArea textarea {
        font-size: 0.9rem !important;
    }
    /* Compact sidebar */
    .css-1544g2n {
        padding-top: 2rem !important;
    }
    /* More compact headers */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Helper Functions ----------
def format_citations(docs) -> str:
    """Extract and format unique citations from retrieved documents."""
    citations = set()
    
    # Legal code abbreviation mapping for consistency
    code_mapping = {
        'OR': 'CO (Code of Obligations)',
        'ZGB': 'CC (Civil Code)',
        'URG': 'CopA (Copyright Act)',
        'actual DSG': 'FADP (Federal Act on Data Protection)',
    }
    
    for doc in docs:
        metadata = doc.metadata
        source = metadata.get('source', 'Unknown')
        page = metadata.get('page', '?')
        
        # Format citation based on source type
        if 'Handout' in source:
            citation = f"({source} p. {page})"
        elif source in code_mapping:
            citation = f"({code_mapping[source]})"
        elif source in ['Criminal Law (english)', 'Bundesgesetz über die Produktehaftpflicht']:
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
    
    # Check if we have any relevant documents at all
    if len(docs) == 0:
        return "No relevant information was found in the database for your query. Please try rephrasing your question or ask about a different topic covered in Swiss/EU legal materials.", [], {}
    
    # Format context
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Create prompt (same as ask.py)
    prompt = f"""You are Swiss Legal Research Assistant, a retrieval-augmented AI that specializes in Swiss and EU law.

Your mission:
1. Provide accurate, well-researched answers to legal questions about Swiss/EU privacy, IP, contract, liability, and technology law
2. ALWAYS cite exact document + page or article number in parentheses immediately after each fact, e.g. (Handout #5 p. 12) or (OR Art. 101 II)
3. NEVER hallucinate an article, law, or page reference; rely ONLY on retrieved context
4. If the retrieved context doesn't contain sufficient information to answer the question fully, state clearly what information is missing

Style requirements:
- Answer in concise, clear language
- Use bullet points for listing key information
- Format sources as separate bullet points at the end using the format:
  • "Law/Document name: Article/Page reference - Brief description"
- Include direct quotes only when essential
- English only
- No explanations about your process
- If the retrieved documents don't provide a clear answer, explicitly state: "The retrieved documents don't contain sufficient information about [specific topic]."

Context from course materials:
{context}

Question: {question}

Answer:"""
    
    # Get answer from LLM with token counting
    model_name = st.session_state.get('model_option', "gpt-4o")
    
    # Define token cost rates based on model (July 2025 pricing)
    model_costs = {
        "gpt-4o": {"input": 0.0005, "output": 0.0015},
        "gpt-4o-mini": {"input": 0.00025, "output": 0.00075},
        "gpt-4o-128k": {"input": 0.001, "output": 0.002},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo-16k": {"input": 0.0005, "output": 0.0015},
    }
    
    # Get token cost for current model
    token_cost = model_costs.get(model_name, {"input": 0.0005, "output": 0.0015})
    
    # Create LLM instance
    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
        max_tokens=300
    )
    
    # Get approximate token count for input (rough estimation)
    import tiktoken
    enc = tiktoken.encoding_for_model(model_name if model_name != "gpt-4o-mini" else "gpt-4o")
    input_tokens = len(enc.encode(prompt))
    
    # Invoke LLM
    response = llm.invoke(prompt)
    
    # Get approximate token count for output
    output_tokens = len(enc.encode(response.content))
    
    # Calculate cost
    input_cost = input_tokens * token_cost["input"] / 1000  # per 1K tokens
    output_cost = output_tokens * token_cost["output"] / 1000  # per 1K tokens
    total_cost = input_cost + output_cost
    
    # Store token usage in session state
    if 'token_usage' not in st.session_state:
        st.session_state.token_usage = []
    
    st.session_state.token_usage.append({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "model": model_name,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost
    })
    
    return response.content, docs, {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost
    }

# ---------- Init (runs once) ----------
# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

@st.cache_resource  # keeps objects alive across reruns
def load_vectorstore():
    """Load the FAISS vector store."""
    # Adjust path to go up from scripts directory to ExamRAG directory
    current_dir = Path(__file__).parent
    index_path = current_dir.parent / "index" / "faiss_index"
    
    if not index_path.exists():
        st.error(f"FAISS index not found at {index_path}")
        st.error("Please run: python scripts/build_index.py")
        st.stop()
    
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)
        return vectorstore
    except Exception as e:
        st.error(f"Failed to load FAISS index: {e}")
        st.stop()

# Load environment variables from .env file if it exists
env_path = Path(__file__).parent.parent / ".env"
template_env_path = Path(__file__).parent.parent / ".env.template"

# Try to load API key from .env file
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    # Status messages moved to debug info section at bottom
elif template_env_path.exists():
    # If only template exists, use template as fallback
    load_dotenv(dotenv_path=template_env_path)

# Check for OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")

# Add API key input in the sidebar if not found or invalid
if not api_key or api_key == 'your-openai-api-key-here':
    with st.sidebar:
        st.warning("⚠️ Valid OpenAI API key not found")
        st.info("Enter your API key below (it will only be stored for this session)")
        
        api_key = st.text_input("OpenAI API Key:", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            st.success("✅ API key set for this session")
        else:
            st.error("OpenAI API key required to use this application")
            # Create instructions for setting up .env file
            st.markdown("""
            ### How to add your API key permanently:
            
            1. Copy the `.env.template` file to a new file named `.env`
            2. Edit the `.env` file and replace the sample key with your actual OpenAI API key
            3. Restart the application
            
            **Note:** The `.env` file should NEVER be committed to Git
            """)
            st.stop()

# Load vectorstore
try:
    vectorstore = load_vectorstore()
    system_status = "✅ Ready"
except Exception as e:
    st.error(f"Failed to initialize system: {e}")
    st.stop()

# ---------- UI ----------
# Settings in sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Document retrieval settings
    st.subheader("🔍 Retrieval Settings")
    
    # Explanation of optimal chunk selection with research-backed recommendations
    with st.expander("ℹ️ How many chunks should I choose?"):
        st.markdown("""
        **Optimal chunk selection depends on:**
        
        1. **Question complexity**: More complex questions need more context
           - Simple factual questions: 3-4 chunks
           - Complex legal analyses: 7-10 chunks
        
        2. **Precision vs. Recall tradeoff**:
           - Fewer chunks (1-3): More precise answers, may miss context
           - More chunks (7-10): More comprehensive, may include irrelevant info
        
        3. **Research findings**:
           - For legal RAG systems, studies show 5-7 chunks typically provides the optimal balance
           - Citation accuracy improves up to ~6 chunks, then plateaus
           
        **Recommendation**: Start with 6 chunks and adjust based on results
        """)
    
    k_chunks = st.slider(
        "Number of document chunks to retrieve", 
        min_value=1, 
        max_value=10, 
        value=6,
        help="More chunks = more comprehensive context, but slower responses and potentially more noise"
    )
    
    show_context = st.checkbox("Show retrieved context", value=False)
    
    # Model settings
    st.subheader("🤖 Model Settings")
    
    # Define model options with descriptions
    model_options = {
        "gpt-4o": "Gold-standard; best accuracy (default)",
        "gpt-4o-mini": "Fast & cheap; good fallback",
        "gpt-4o-128k": "For huge contexts (rarely needed)",
        "gpt-4-turbo": "Same quality as gpt-4o, 10× price",
        "gpt-3.5-turbo-16k": "OK for summaries, weak on nuanced law"
    }
    
    # Add a big, clear model selector with formatting to make it stand out
    st.markdown("**Select AI Model:**")
    model_option = st.selectbox(
        "Choose model",
        list(model_options.keys()),
        index=0,
        format_func=lambda x: f"{x} - {model_options[x]}",
        help="Select which OpenAI model to use for answering questions"
    )
    
    # Add a small note about the current model selection
    st.caption(f"Current: {model_option} - {model_options[model_option]}")
    
    # Model info table
    with st.expander("📊 Model Comparison (July 2025)"):
        st.markdown("""
        | Model | Input Cost | Output Cost | Latency | Accuracy | Suitability |
        |-------|------------|-------------|---------|----------|-------------|
        | **gpt-4o** | **$0.0005/1k tok** | **$0.0015/1k tok** | 2-3 s | ★★★★★ | Gold-standard; default |
        | gpt-4o-mini | $0.00025/1k tok | $0.00075/1k tok | ~1 s | ★★★★☆ | Fast & cheap fallback |
        | gpt-4o-128k | $0.001/1k tok | $0.002/1k tok | 3-4 s | ★★★★★ | For huge contexts only |
        | gpt-4-turbo | $0.01/1k tok | $0.03/1k tok | 4-6 s | ★★★★★ | 10× price; not recommended |
        | gpt-3.5-turbo-16k | $0.0005/1k tok | $0.0015/1k tok | 1-2 s | ★★☆☆☆ | Weak on nuanced law |
        """)
    
    st.markdown("---")
    st.markdown("**About**: This tool uses RAG (Retrieval-Augmented Generation) to answer legal questions based on Swiss & EU legal materials.")

# Save model choice to session state
st.session_state['model_option'] = model_option

# ---------- UI ----------
st.title("⚖️ Swiss Legal Research Assistant")
st.markdown("*Your AI-powered research tool for Swiss/EU legal concepts*")

# Example questions (in a compact expandable section)
with st.expander("💡 Sample Questions"):
    st.caption("Example legal research questions")
    
    # Example questions based on common legal research topics
    example_questions = [
        "What are the key principles of Swiss data protection law?",
        "How does Swiss law regulate AI systems and algorithms?",
        "What are the requirements for trademark protection in Switzerland?",
        "How is online defamation handled under Swiss law?",
        "What are the main elements of product liability in Switzerland?"
    ]

    # Display in a more compact way
    for i, example in enumerate(example_questions):
        if st.button(example, key=f"ex_{i+1}"):
            st.session_state['question'] = example

# Initialize question in session state if not present
if 'question' not in st.session_state:
    st.session_state['question'] = ""

# Main input
st.subheader("💭 Research Query")
question = st.text_area(
    "Enter your legal research question:",
    value=st.session_state['question'],
    height=100,
    placeholder="e.g., What are the main requirements for trademark protection in Switzerland?"
)

# Update session state with current question
st.session_state['question'] = question

col1, col2 = st.columns([3, 1])
submit_button = col1.button("🔍 Research", type="primary")
clear_button = col2.button("🧹 Clear", type="secondary")

if clear_button:
    st.session_state['question'] = ""
    st.experimental_rerun()

if submit_button and question.strip():
    with st.spinner("🔎 Searching legal materials..."):
        try:
            answer, docs, token_info = answer_question(vectorstore, question.strip(), k_chunks)
            citations = format_citations(docs)
            
            # Save to history
            timestamp = datetime.now().strftime("%H:%M:%S")
            st.session_state.history.append({
                "timestamp": timestamp,
                "question": question.strip(),
                "answer": answer,
                "citations": citations,
                "docs": docs,
                "token_info": token_info
            })
            
            # Display answer
            st.markdown("### 📝 Research Results")
            st.markdown(answer)
            
            # Display sources/references
            if citations:
                st.markdown("### 📚 Sources & References")
                
                # Extract and format citations in a more readable way
                citation_list = sorted(list(set([c.strip() for c in citations.replace("(", "").replace(")", "").split()])))
                
                # Format each citation as a bullet point
                formatted_citations = "\n".join([f"• {citation}" for citation in citation_list])
                st.markdown(formatted_citations)
            
            # Show token usage and cost
            with st.expander("📊 Token Usage & Cost"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Input Tokens", f"{token_info['input_tokens']:,}")
                col2.metric("Output Tokens", f"{token_info['output_tokens']:,}")
                col3.metric("Total Tokens", f"{token_info['total_tokens']:,}")
                
                st.markdown(f"**Cost Breakdown:**")
                st.markdown(f"* Input: ${token_info['input_cost']:.5f}")
                st.markdown(f"* Output: ${token_info['output_cost']:.5f}")
                st.markdown(f"* **Total: ${token_info['total_cost']:.5f}**")
                
                if 'token_usage' in st.session_state and len(st.session_state.token_usage) > 1:
                    total_session_cost = sum(item['total_cost'] for item in st.session_state.token_usage)
                    st.markdown(f"**Session total: ${total_session_cost:.5f}**")
            
            # Show retrieved context if requested
            if show_context and docs:
                with st.expander(f"📄 Retrieved Context ({len(docs)} chunks)"):
                    for i, doc in enumerate(docs, 1):
                        metadata = doc.metadata
                        source = metadata.get('source', 'Unknown')
                        page = metadata.get('page', '?')
                        
                        st.markdown(f"**{i}. {source} (page {page})**")
                        content = doc.page_content.strip()
                        if len(content) > 300:
                            st.text(content[:300] + "...")
                        else:
                            st.text(content)
                        st.markdown("---")
                        
        except Exception as e:
            st.error(f"❌ Query failed: {e}")
            st.error("Please try rephrasing your question or check your OpenAI API key.")

# Display History
if st.session_state.history:
    st.markdown("---")
    st.subheader("📜 Research History")
    
    for i, item in enumerate(reversed(st.session_state.history[-5:])):  # Show last 5 questions
        with st.expander(f"{item['timestamp']} - {item['question'][:50]}..."):
            st.markdown(f"**Research Query: {item['question']}**")
            st.markdown(item['answer'])
            if item['citations']:
                st.info(item['citations'])
            
            # Show token usage if available but make it less prominent
            if 'token_info' in item:
                token_info = item['token_info']
                st.caption(f"Tokens: {token_info['input_tokens']} in / {token_info['output_tokens']} out • Cost: ${token_info['total_cost']:.5f}")
            
            # Add button to reuse this question
            if st.button("Reuse this query", key=f"reuse_{i}"):
                st.session_state['question'] = item['question']
                st.experimental_rerun()

# Footer with extra tips and info
st.markdown("---")
st.subheader("📝 Research Tips")
tips_col1, tips_col2, tips_col3 = st.columns([2, 2, 1])

with tips_col1:
    st.markdown("""
    **Effective Research Queries:**
    * Be specific about the legal concept
    * Include the jurisdiction (Swiss/EU)
    * Mention specific laws if known
    * Ask about specific requirements or principles
    """)

with tips_col2:
    st.markdown("""
    **Legal Code References:**
    * OR = Swiss Code of Obligations
    * ZGB = Swiss Civil Code
    * DSG = Federal Data Protection Act
    * URG = Copyright Act
    """)

with tips_col3:
    # System status in a less prominent place
    st.caption("System Status")
    st.caption(f"{system_status}")
    if api_key:
        st.caption("✓ API key configured")
    
    # Add version number
    st.caption("v1.0.3")

st.markdown("*This tool analyzes Swiss and EU legal materials to provide research assistance. Always consult with a qualified legal professional for advice.*")
