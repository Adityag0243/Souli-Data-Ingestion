"""
Chatbot Testing UI - Test the conversation/retrieval pipeline
"""
import streamlit as st
import os
from souli_pipeline.config_loader import load_config
from souli_pipeline.conversation.engine import ConversationEngine

def get_engine():
    """Load configuration and initialize the conversation engine"""
    if "engine" not in st.session_state:
        # Auto-detect config
        config_path = "configs/pipeline.gcp.yaml" if os.path.exists("configs/pipeline.gcp.yaml") else "configs/pipeline.yaml"
        
        try:
            cfg = load_config(config_path)
            # Find the latest gold/excel files if available, otherwise just rely on engine defaults
            # In a real deployed customer scenario, these are pre-ingested or pre-loaded.
            # We will just try to find the newest gold.xlsx in outputs to seed the framework.
            gold_path = None
            if os.path.exists("outputs"):
                runs = sorted([r for r in os.listdir("outputs") if os.path.isdir(os.path.join("outputs", r))], reverse=True)
                for r in runs:
                    gp = os.path.join("outputs", r, "energy", "gold.xlsx")
                    if os.path.exists(gp):
                        gold_path = gp
                        break
            
            st.session_state.engine = ConversationEngine.from_config(cfg, gold_path=gold_path)
            st.session_state.config_path_used = config_path
            
        except Exception as e:
            st.error(f"Failed to initialize chatbot engine: {str(e)}")
            return None
            
    return st.session_state.engine

def show():
    st.header("💬 Souli")
    st.write("Your inner wellness companion. Feel free to share what's on your mind.")
    
    engine = get_engine()
    
    if not engine:
        st.warning("Chatbot engine could not be started. Check configuration.")
        return
        
    st.write("---")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        # Start with Souli's greeting
        st.session_state.messages = [{"role": "assistant", "content": engine.greeting()}]
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if user_input := st.chat_input("Type your message here...", key="chat_input"):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            
            try:
                # Use turn_stream to provide a typing effect
                stream = engine.turn_stream(user_input)
                full_response = ""
                for chunk in stream:
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                error_msg = f"❌ Error connecting to Souli core: {str(e)}"
                response_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Sidebar info
    with st.sidebar:
        st.write("### ℹ️ About")
        st.info("""
        This interface tests the Souli conversation engine. It connects directly to your Qdrant vector database and Ollama models just as the final application would.
        """)
        
        if "config_path_used" in st.session_state:
            st.caption(f"Using config: `{st.session_state.config_path_used}`")
            
        st.write("---")
        
        # Display underlying state for debugging nicely
        if engine and st.session_state.messages:
            with st.expander("🔍 Inner State (Debug)", expanded=False):
                diag = engine.diagnosis_summary
                st.json({
                    "energy_node": diag.get("energy_node"),
                    "confidence": diag.get("confidence"),
                    "intent": diag.get("intent"),
                    "phase": diag.get("phase"),
                    "turn_count": diag.get("turn_count")
                })
        
        # Clear chat button
        if st.button("🔄 Restart Session", use_container_width=True):
            engine.reset()
            st.session_state.messages = [{"role": "assistant", "content": engine.greeting()}]
            st.rerun()
