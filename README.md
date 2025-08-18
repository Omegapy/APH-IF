
# Version 
# Advanced Parallel Hybrid - Intelligent Fusion (APH-IF) Technology

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/uv-package%20manager-blue.svg)](https://github.com/astral-sh/uv)
[![Neo4j](https://img.shields.io/badge/Neo4j-008CC1?style=flat&logo=neo4j&logoColor=white)](https://neo4j.com/)
[![Windows](https://img.shields.io/badge/Windows-native%20development-blue.svg)](https://www.microsoft.com/windows)

---

## Advanced Parallel HybridRAG - Intelligent Fusion Overview

Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) is a novel Retrieval Augmented Generation (RAG) system that differs from traditional RAG approaches by performing semantic and traversal searches concurrently, rather than sequentially, and fusing the the result using a LLM or a LRM to generate the final response.

---

### **Core Innovation: True Parallel Processing and Intelligent Fusion**

**Parallel HybridRAG (PH)**

Conventional HybridRAG systems often process queries sequentially, for instance: `if condition: vector_search() else: graph_search()`. In contrast, APH-IF performs true concurrent execution of multiple retrieval methods, for instance: `asyncio.gather(vector_task, graph_task)`.

This parallelism is achieved using asynchronous programming (asyncio) and multi-threading. Based on the user's prompt, the PH engine executes both VectorRAG (semantic search) and GraphRAG (traversal search) queries in parallel on a Knowledge Graph that uses vector embeddings as properties.

**Intelligent Context Fusion (IF)**

The retrieved results from the concurrent queries are then fused using an Intelligent Context Fusion (IF) engine. The IF engine uses a LLM or a LRM to combine the results from the concurrent queries into a single, coherent final response. For instance: `intelligent_context_fusion(vector_results, graph_results)`

---

This project is based on my MRCA APH-IF (Mining Regulatory Compliance Assistant) project, which is a web application that uses AHP-IF to provide quick, reliable, and easy access to MSHA (Mine Safety and Health Administration) regulations using natural language queries.

MRCA APH-IF Website: https://mrca-frontend.onrender.com/  
MRCA APH-IFGitHub rep.: https://github.com/Omegapy/MRCA-Advanced-Parallel-HybridRAG-Intelligent-Fusion

⚠️ The MRCA APH-IF project has limited funds (I am a student). Once the monthly LLM usage fund limit is reached, the application will stop providing responses and will display an error message.  
Please contact me (a.omegapy@gmail.com) if this happend and you still want to try the application.

---
 
<img width="30" height="30" align="center" src="https://github.com/user-attachments/assets/a8e0ea66-5d8f-43b3-8fff-2c3d74d57f53"> **Alexander Ricciardi (Omega.py)**

[![Website](https://img.shields.io/badge/Website-alexomegapy.com-blue)](https://www.alexomegapy.com)
[![Medium](https://img.shields.io/badge/Medium-12100E?style=flat&logo=medium&logoColor=white)](https://medium.com/@alex.omegapy)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5.svg?logo=linkedin&logoColor=white)](https://linkedin.com/in/alex-ricciardi)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white)](https://github.com/Omegapy)  
Date: 08/05/2025

--- ---

## License & Credits

**© 2025 Alexander Samuel Ricciardi - All rights reserved.**

- **License**: Apache-2.0
- **Technology**: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF)
- **Version**: dev 0.0.4
- **Author**: Alexander Ricciardi (Omega.py)
- **Date**: August 2025

---



---

## 🚀 Quick Start

### Native Windows Development

APH-IF is optimized for native Windows development using `uv` for fast, reliable Python package management:

```powershell
# 1. Install uv (Python package manager)
iwr -useb https://astral.sh/uv/install.ps1 | iex

# 2. Clone and setup
git clone https://github.com/Omegapy/APH-IF-Dev.git
cd APH-IF-Dev
.\setup-dev.ps1

# 3. Configure environment (choose your mode)
.\switch-environment.ps1 -Environment development  # Safe for development
.\switch-environment.ps1 -ShowCurrent              # Check current environment

# 4. Start all services
.\start-dev.ps1

# 5. Access the application
# Frontend: http://localhost:8501
# Backend API: http://localhost:8000/healthz
# Data Processing: http://localhost:8010/healthz
```

### 🌍 Environment Management

APH-IF includes intelligent environment management with centralized `env_manager.py` for safe database operations:

- **Development Mode**: Uses development Neo4j instance (safe for experimentation)
- **Production Mode**: Uses production Neo4j instance (live data - use with caution)
- **Testing Mode**: Uses test Neo4j instance (only for explicit testing scenarios)

```powershell
# Switch environments safely using set_environment.py
python set_environment.py --mode development
python set_environment.py --mode production
python set_environment.py --mode development --force-test-db true  # For testing only

# Verify current environment
python set_environment.py --status
```

### 🧪 Testing with Environment Management

For safe testing that requires database access:

```powershell
# Setup test environment
python set_environment.py --mode development --force-test-db true

# Run your tests here
# (modules automatically use test database)

# Cleanup - switch back to development database
python set_environment.py --mode development --force-test-db false
```

## 🏗️ Project Architecture

### Service Structure

APH-IF uses a microservices architecture with isolated environments:

```
APH-IF-Dev/
├── backend/              # FastAPI REST API service
│   ├── app/             # Application code
│   ├── tests/           # Backend tests
│   ├── pyproject.toml   # Backend dependencies
│   └── .venv/           # Isolated virtual environment
├── data_processing/      # Data ingestion and graph building
│   ├── processing/      # Core processing modules
│   ├── tests/           # Processing tests
│   ├── pyproject.toml   # Processing dependencies
│   └── .venv/           # Isolated virtual environment
├── frontend/            # Streamlit web interface
│   ├── app/             # Frontend application
│   ├── pyproject.toml   # Frontend dependencies
│   └── .venv/           # Isolated virtual environment
├── common/              # Shared utilities and models
│   ├── config.py        # Shared configuration
│   ├── logging.py       # Logging utilities
│   └── models.py        # Shared data models
├── env_manager.py       # Centralized environment management
├── check_environment.py # Environment validation utility
└── .env                 # Shared environment configuration
```

### Key Components

- **Backend Service** (`localhost:8000`): FastAPI REST API with health monitoring
- **Data Processing Service** (`localhost:8010`): Document ingestion and graph building
- **Frontend Service** (`localhost:8501`): Streamlit web interface
- **Environment Management**: Centralized environment and database management via `set_environment.py`
- **Neo4j AuraDB**: Cloud-based graph database with environment-aware instance selection

## 🔧 Environment Management

### Centralized Environment Control

APH-IF uses `env_manager.py` for safe, centralized environment management:

```python
from env_manager import EnvManager

# Environment switching
EnvManager.set_env_mode(dev=True)           # Development mode
EnvManager.set_env_mode(dev=False)          # Production mode

# Test database control (development only)
EnvManager.set_test_db_mode(force_test_db=True)   # Enable test DB
EnvManager.set_test_db_mode(force_test_db=False)  # Disable test DB

# Configuration access
config = EnvManager.get_neo4j_config()
app_env, force_test_db, verbose = EnvManager.get_current_mode()

# Utilities
EnvManager.print_current_status()           # Show current environment
EnvManager.validate_environment()           # Validate configuration
```

### Environment Safety Rules

1. **Development Mode**: Safe for experimentation with development database
2. **Production Mode**: Live data - requires explicit confirmation for modifications
3. **Testing Mode**: Only available in development environment with `FORCE_TEST_DB=true`
4. **Always verify environment** before graph modifications: `python check_environment.py --require-test`

### Database Instance Selection

APH-IF automatically selects the appropriate Neo4j instance:

- **NEO4J_URI_DEV**: Development database (safe for development work)
- **NEO4J_URI_PROD**: Production database (live data - use with extreme caution)
- **NEO4J_URI_TEST**: Test database (only when `FORCE_TEST_DB=true`)

## 📦 UV Package Management

### Service-Specific Environments

Each service maintains its own isolated environment:

```powershell
# Backend service
cd backend
uv sync                    # Install/sync dependencies
uv add fastapi            # Add new dependency
uv remove package-name    # Remove dependency
uv run uvicorn app.main:app --reload --port 8000

# Data processing service
cd data_processing
uv sync
uv add langchain
uv run uvicorn processing.main:app --reload --port 8010

# Frontend service
cd frontend
uv sync
uv add streamlit
uv run streamlit run app/bot.py --server.port 8501
```

### Development Scripts

```powershell
# Complete development workflow
.\setup-dev.ps1                              # Initial setup (run once)
.\switch-environment.ps1 -Environment development  # Set environment
.\start-dev.ps1                              # Start all services
# ... develop and test ...
.\stop-dev.ps1                               # Stop all services

# Selective service management
.\start-dev.ps1 -Backend                     # Start only backend
.\start-dev.ps1 -Backend -Frontend           # Start backend + frontend
.\start-dev.ps1 -All                         # Start all services explicitly

# Environment management
.\switch-environment.ps1 -ShowCurrent        # Check current environment
.\switch-environment.ps1 -Environment production  # Switch to production
.\switch-environment.ps1 -Help               # Show all options
```

**📖 Comprehensive Guides:**
- **[PowerShell Scripts Usage Guide](documents/powershell_scripts_usage_guide.md)** - Complete usage examples and troubleshooting
- **[Cross-System Testing Guide](documents/cross_system_testing_guide.md)** - Testing procedures for different Windows configurations

## 📚 Documentation

### Core Documentation

- **[Environment Management Guide](documents/environment_management_guide.md)** - Complete env_manager.py usage and safety patterns
- **[UV Environment Guide](documents/uv_environment_guide.md)** - UV package manager usage and best practices
- **[Data Processing Modules](documents/data_processing_modules.md)** - Complete guide to all data processing functionality

### Module-Specific Documentation

- **[Initial Graph Build](documents/initial_graph_build_usage.md)** - Building the hybrid knowledge store
- **[Relationship Augmentation](documents/relationship_augmentation_usage.md)** - Adding entity relationships with LLM analysis
- **[Document Embeddings](documents/compute_doc_embeddings_usage.md)** - Document embedding computation and storage
- **[Pipeline Launcher](documents/launch_data_processing_usage.md)** - Unified data processing pipeline

### Development Documentation

- **[Environment Safety Patterns](documents/environment_safety_patterns.md)** - Safe database operation patterns
- **[Testing with Environment Manager](documents/testing_patterns.md)** - Proper test environment setup and cleanup
- **[Configuration Reference](documents/environment_configuration_reference.md)** - Complete environment variable reference

---

## 🧪 Testing and Development

### Safe Testing Patterns

Always use the environment manager for database tests:

```python
from env_manager import EnvManager

def setup_module():
    """Setup test environment before running tests."""
    EnvManager.set_env_mode(dev=True)
    EnvManager.set_test_db_mode(force_test_db=True)

def teardown_module():
    """Cleanup test environment after running tests."""
    EnvManager.set_test_db_mode(force_test_db=False)

def test_database_operation():
    """Example test with proper environment management."""
    EnvManager.set_test_db_mode(force_test_db=True)
    try:
        # Test code here - automatically uses test database
        pass
    finally:
        EnvManager.set_test_db_mode(force_test_db=False)
```

### VS Code Integration

Use the compound debugging configuration for all services:

1. Open VS Code in project root
2. Select "🚀 All Services (Development)" from debug configurations
3. Start debugging - all services launch with proper environment isolation

## Background

### Technology Innovation

**APH-IF** represents a significant advancement in RAG systems by:

1. **True Parallelism**: Concurrent execution vs. sequential processing
2. **Intelligent Fusion**: LLM-powered result synthesis vs. simple concatenation
3. **Native Windows Development**: Fast, reliable development with uv package manager
4. **Environment Safety**: Intelligent environment management for safe database operations
5. **Microservices Architecture**: Scalable, maintainable service deployment
6. **Centralized Configuration**: Single `.env` file with intelligent instance selection




