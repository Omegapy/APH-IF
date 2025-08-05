"""
APH-IF Technology Framework - Frontend Application
Advanced Parallel HybridRAG - Intelligent Fusion

Frontend microservice providing multiple interfaces for APH-IF technology.
Includes overview, bot interface, and documentation.

Author: Alexander Ricciardi
Date: 2025-08-05
License: Apache-2.0
"""

import streamlit as st
import os
import sys
import requests
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import uuid
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

# =========================================================================
# Backend Communication
# =========================================================================
def check_backend_health() -> Dict[str, Any]:
    """Check backend service health"""
    try:
        response = requests.get("http://aph-if-backend:8000/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"status": "unreachable", "error": str(e)}

def send_query_to_backend(query: str, session_id: str) -> Dict[str, Any]:
    """Send query to backend for processing"""
    try:
        payload = {
            "query": query,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "source": "streamlit_frontend",
                "user_agent": "APH-IF-Bot/1.0"
            }
        }

        response = requests.post(
            "http://aph-if-backend:8000/api/v1/query",
            json=payload,
            timeout=30,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            return response.json()
        else:
            return {
                "error": f"Backend returned status {response.status_code}",
                "details": response.text
            }

    except requests.exceptions.Timeout:
        return {
            "error": "Request timeout",
            "details": "The backend took too long to respond"
        }
    except requests.exceptions.ConnectionError:
        return {
            "error": "Connection failed",
            "details": "Could not connect to backend service"
        }
    except Exception as e:
        return {
            "error": "Unexpected error",
            "details": str(e)
        }

def main():
    """Main application entry point"""

    # Page configuration
    st.set_page_config(
        page_title="APH-IF Frontend",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Navigation
    page = st.sidebar.selectbox(
        "🧭 Navigate",
        ["🏠 Overview", "🤖 Bot Interface", "📚 Documentation", "⚙️ System Status"]
    )

    if page == "🤖 Bot Interface":
        render_bot_interface()

    elif page == "📚 Documentation":
        render_documentation()

    elif page == "⚙️ System Status":
        render_system_status()

    else:
        render_overview()
    
def render_overview():
    """Render the overview page"""

    # Main header
    st.title("🧠 Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF)")
    st.markdown("**Frontend Microservice - Technology Framework v dev 0.0.1**")
    st.markdown("---")

    # Introduction
    st.markdown("""
    ## Welcome to APH-IF Technology Framework

    This is a novel Retrieval Augmented Generation (RAG) system that differs from traditional RAG approaches by:

    ### 🔄 **Parallel HybridRAG (PH)**
    - **Concurrent Execution**: Multiple retrieval methods run simultaneously rather than sequentially
    - **True Parallelism**: Uses `asyncio.gather(vector_task, graph_task)` for concurrent processing
    - **Dual Search**: VectorRAG (semantic search) and GraphRAG (traversal search) in parallel

    ### 🧠 **Intelligent Fusion (IF)**
    - **Smart Fusion**: LLM/LRM combines results from concurrent queries
    - **Coherent Response**: Single, unified final response from multiple sources
    - **Context Aware**: Intelligent merging of VectorRAG and GraphRAG results
    """)
    
    # Microservices Architecture
    st.header("🏗️ Microservices Architecture")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔧 Backend Service")
        st.markdown("""
        **FastAPI Backend** (Port 8000)
        - Parallel HybridRAG Engine
        - Intelligent Fusion Engine
        - Circuit Breaker Pattern
        - Health Monitoring
        - API Documentation: `/docs`
        """)

        # Backend status check
        import requests
        try:
            response = requests.get("http://aph-if-backend:8000/health", timeout=2)
            if response.status_code == 200:
                st.success("✅ Backend Online")
            else:
                st.warning("⚠️ Backend Issues")
        except:
            st.error("❌ Backend Offline")

    with col2:
        st.subheader("🖥️ Frontend Service")
        st.markdown("""
        **Streamlit Frontend** (Port 8501)
        - Overview Interface
        - Integrated Bot Chat Interface
        - Documentation
        - System Monitoring
        """)
        st.success("✅ Frontend Online")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🏠 Overview", "🚀 Quick Start", "📚 Documentation"])
    
    with tab1:
        st.header("Technology Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔍 Traditional RAG vs APH-IF")
            st.code("""
# Traditional Sequential RAG
if condition:
    result = vector_search()
else:
    result = graph_search()

# APH-IF Parallel Processing
vector_task = vector_search_async()
graph_task = graph_search_async()
results = await asyncio.gather(vector_task, graph_task)
final_result = intelligent_fusion(results)
            """, language="python")
        
        with col2:
            st.subheader("🏗️ Architecture Components")
            st.markdown("""
            - **Vector Store**: Semantic embeddings
            - **Knowledge Graph**: Structured relationships
            - **Parallel Engine**: Concurrent query execution
            - **Fusion Engine**: Intelligent result combination
            - **LLM Integration**: OpenAI, Gemini support
            """)
    
    with tab2:
        st.header("🚀 Quick Start Guide")
        
        st.subheader("1. Configuration")
        st.code("""
# Copy and configure secrets
cp .streamlit/secrets.toml.template .streamlit/secrets.toml

# Edit secrets.toml with your API keys:
OPENAI_API_KEY = "your-openai-key"
GEMINI_API_KEY = "your-gemini-key"
NEO4J_URI = "your-neo4j-uri"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "your-password"
        """, language="bash")
        
        st.subheader("2. Docker Setup")
        st.code("""
# Build and run with Docker Compose
docker-compose up --build

# Or build individual container
docker build -t aph-if .
docker run -p 8501:8501 -p 8000:8000 aph-if
        """, language="bash")
        
        st.subheader("3. Development")
        st.code("""
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py

# Run FastAPI backend (if implemented)
uvicorn main:app --host 0.0.0.0 --port 8000
        """, language="bash")
    
    with tab3:
        st.header("📚 Documentation")
        
        st.markdown("""
        ### 📖 Key Concepts
        
        **Advanced Parallel HybridRAG (APH)**
        - Concurrent execution of multiple retrieval strategies
        - Asynchronous processing with asyncio
        - Multi-threading support for CPU-intensive tasks
        
        **Intelligent Fusion (IF)**
        - LLM-powered result combination
        - Context-aware merging algorithms
        - Quality scoring and ranking
        
        ### 🔗 Related Projects
        - [MRCA APH-IF](https://github.com/Omegapy/MRCA-Advanced-Parallel-HybridRAG-Intelligent-Fusion)
        - [Live Demo](https://mrca-frontend.onrender.com/)
        
        ### 📄 License
        Apache-2.0 License - See LICENSE file for details
        
        ### 👨‍💻 Author
        Alexander Samuel Ricciardi (Omega.py)
        """)
    
def render_documentation():
    """Render documentation page"""
    st.title("📚 APH-IF Documentation")
    st.markdown("---")

    # Quick links
    st.markdown("""
    ### 🔗 Quick Links
    - **[Backend API Docs](http://localhost:8000/docs)** - FastAPI Swagger UI (external access)
    - **[Backend Health](http://localhost:8000/health)** - Service health status (external access)
    - **[Bot Interface](http://localhost:8501)** - Integrated chat bot (🤖 Bot Interface tab)

    *Note: Backend links use localhost for external browser access, while internal Docker communication uses service names.*
    """)

    # Architecture overview
    st.header("🏗️ Architecture Overview")
    st.markdown("""
    The APH-IF system uses a microservices architecture:

    1. **Backend Service** (FastAPI) - Core processing engine
    2. **Frontend Service** (Streamlit) - User interfaces
    3. **Database Service** (Neo4j) - Graph storage
    """)

def render_system_status():
    """Render system status page"""
    st.title("⚙️ System Status")
    st.markdown("---")

    # Service status checks
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🔧 Backend")
        try:
            import requests
            response = requests.get("http://aph-if-backend:8000/health", timeout=2)
            if response.status_code == 200:
                st.success("✅ Online")
                data = response.json()
                st.json(data)
            else:
                st.error("❌ Error")
        except:
            st.error("❌ Offline")

    with col2:
        st.subheader("🖥️ Frontend")
        st.success("✅ Online")
        st.text("Current service")

    with col3:
        st.subheader("🗄️ Database")
        try:
            import requests
            response = requests.get("http://localhost:7474", timeout=2)
            if response.status_code == 200:
                st.success("✅ Online")
            else:
                st.warning("⚠️ Issues")
        except:
            st.error("❌ Offline")

# =========================================================================
# Bot Interface Implementation
# =========================================================================
def initialize_bot_session_state():
    """Initialize session state for bot interface"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

def render_bot_interface():
    """Render the integrated bot chat interface"""
    st.markdown("## 🤖 APH-IF Bot Interface")
    st.markdown("**Advanced Parallel HybridRAG - Intelligent Fusion Assistant**")

    # Initialize session state
    initialize_bot_session_state()

    # Sidebar with status
    with st.sidebar:
        st.header("🔧 Bot Status")

        # Backend health check
        health = check_backend_health()
        if health.get("status") == "healthy":
            st.success("✅ Backend Online")
            st.json(health)
        else:
            st.error("❌ Backend Offline")
            st.json(health)

        st.markdown("---")
        st.markdown("### 📊 Session Info")
        st.markdown(f"**Session ID**: `{st.session_state.session_id[:8]}...`")
        st.markdown(f"**Messages**: {len(st.session_state.messages)}")

        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    # Example queries
    st.markdown("### 💡 Try These Example Queries")
    col1, col2 = st.columns(2)

    examples = [
        "What is Advanced Parallel HybridRAG technology?",
        "How does Intelligent Fusion work?",
        "Explain the APH-IF architecture",
        "What are the benefits of parallel processing?"
    ]

    for i, example in enumerate(examples):
        col = col1 if i % 2 == 0 else col2
        with col:
            if st.button(f"💬 {example}", key=f"example_{i}"):
                st.session_state.messages.append({
                    "role": "user",
                    "content": example,
                    "timestamp": datetime.now().isoformat()
                })
                st.rerun()

    st.markdown("---")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show metadata for assistant messages
            if message["role"] == "assistant" and "metadata" in message:
                with st.expander("📊 Response Details"):
                    st.json(message["metadata"])

    # Chat input
    if prompt := st.chat_input("Ask me about APH-IF technology..."):
        # Add user message to chat history
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        })

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process with backend
        with st.chat_message("assistant"):
            with st.spinner("🧠 Processing with APH-IF technology..."):

                # Send to backend
                response = send_query_to_backend(prompt, st.session_state.session_id)

                if "error" in response:
                    # Handle error
                    error_message = f"❌ **Error**: {response['error']}"
                    if "details" in response:
                        error_message += f"\n\n**Details**: {response['details']}"

                    st.markdown(error_message)

                    # Add error to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message,
                        "timestamp": datetime.now().isoformat(),
                        "error": True
                    })

                else:
                    # Handle successful response
                    assistant_response = response.get("response", "No response received")
                    st.markdown(assistant_response)

                    # Add to chat history with metadata
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": assistant_response,
                        "timestamp": datetime.now().isoformat(),
                        "metadata": {
                            "processing_time": response.get("processing_time", 0),
                            "vector_results_count": response.get("vector_results_count", 0),
                            "graph_results_count": response.get("graph_results_count", 0),
                            "fusion_method": response.get("fusion_method", "intelligent"),
                            "session_id": response.get("session_id", "unknown")
                        }
                    })

if __name__ == "__main__":
    main()
