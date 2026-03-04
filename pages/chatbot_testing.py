"""
Chatbot Testing UI - Test the conversation/retrieval pipeline
"""
import streamlit as st
import pandas as pd
import os
from souli_pipeline.config_loader import load_config
from souli_pipeline.retrieval.match import run_match

def show():
    st.header("💬 Chatbot Testing Interface")
    st.write("""
    Test the Souli conversation engine and retrieval system.
    Ask questions and get responses with retrieval from your teaching content.
    """)
    
    # Setup section
    st.write("---")
    st.write("### ⚙️ Setup")
    
    col1, col2 = st.columns(2)
    
    with col1:
        config_path = st.text_input(
            "Config file path",
            value="configs/pipeline.yaml",
            help="Path to your pipeline configuration file"
        )
    
    with col2:
        if os.path.exists(config_path):
            st.success("✅ Config found")
        else:
            st.warning(f"⚠️ Config not found: {config_path}")
    
    # Load outputs from previous runs
    gold_path = st.text_input(
        "Energy Framework (gold.xlsx)",
        value="outputs/latest/energy/gold.xlsx",
        help="Path to gold.xlsx from energy pipeline"
    )
    
    teaching_path = st.text_input(
        "Teaching Content (merged_teaching_cards.xlsx)",
        value="outputs/latest/youtube/merged_teaching_cards.xlsx",
        help="Optional: Path to merged teaching cards from YouTube pipeline"
    )
    
    st.write("---")
    
    # Chat interface
    st.write("### 💭 Chat")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if user_input := st.chat_input("Ask a question...", key="chat_input"):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            
            try:
                response = process_query(
                    user_input,
                    config_path,
                    gold_path,
                    teaching_path
                )
                
                response_placeholder.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                response_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Sidebar info
    with st.sidebar:
        st.write("### ℹ️ About")
        st.info("""
        This interface tests the Souli conversation engine.
        
        **Features:**
        - Query diagnosis based on energy framework
        - Retrieve related teaching content
        - Test LLM integration (if enabled)
        
        **Requirements:**
        - Processed energy framework (gold.xlsx)
        - Optional: Teaching content from YouTube pipeline
        """)
        
        st.write("---")
        st.write("### 🔧 Session Settings")
        
        use_embeddings = st.checkbox(
            "Use semantic embeddings",
            value=True,
            help="Use sentence-transformers for better retrieval"
        )
        
        top_k = st.slider(
            "Top-K results",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of teaching results to retrieve"
        )
        
        # Store settings
        st.session_state.use_embeddings = use_embeddings
        st.session_state.top_k = top_k
        
        st.write("---")
        
        # Clear chat button
        if st.button("🔄 Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Test data section
    st.write("---")
    st.write("### 📝 Test Queries")
    
    test_queries = [
        "I feel very sad and unmotivated",
        "I am dealing with anxiety",
        "How can I improve my energy?",
        "I feel overwhelmed and scattered",
        "I need guidance for emotional balance"
    ]
    
    st.write("Try one of these sample queries:")
    for query in test_queries:
        if st.button(f"→ {query}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": query})
            st.rerun()


def process_query(query, config_path, gold_path, teaching_path):
    """Process user query and return response"""
    
    response_parts = []
    
    # Load config
    try:
        cfg = load_config(config_path)
    except Exception as e:
        return f"❌ Error loading config: {str(e)}"
    
    # Try to run match/retrieval
    try:
        # Check if files exist
        if not os.path.exists(gold_path):
            return f"❌ Energy framework file not found: {gold_path}"
        
        # Run diagnosis + retrieval
        use_embeddings = getattr(st.session_state, 'use_embeddings', True)
        top_k = getattr(st.session_state, 'top_k', 5)
        
        # Call the match function (you may need to adjust based on your actual API)
        result = run_match(
            config=cfg,
            gold_path=gold_path,
            teaching_path=teaching_path if os.path.exists(teaching_path) else None,
            query=query,
            output_format="json"
        )
        
        response_parts.append("### 🔍 Diagnosis & Retrieval Results\n")
        
        if isinstance(result, dict):
            if "diagnosis" in result:
                response_parts.append(f"**Diagnosis:** {result['diagnosis']}\n")
            if "solution" in result:
                response_parts.append(f"**Solution:** {result['solution']}\n")
            if "teaching" in result:
                response_parts.append("**Related Teaching Content:**\n")
                for i, item in enumerate(result["teaching"][:top_k], 1):
                    response_parts.append(f"{i}. {item}\n")
        else:
            response_parts.append(str(result))
        
        return "".join(response_parts)
    
    except Exception as e:
        return f"⚠️ Query processing error: {str(e)}\n\nTrying basic keyword matching instead..."


def display_retrieval_results(results, max_items=5):
    """Display retrieval results in a formatted way"""
    if not results:
        return "No results found"
    
    output = []
    for i, result in enumerate(results[:max_items], 1):
        output.append(f"{i}. {result}")
    
    return "\n".join(output)