# Advanced Parallel Hybrid - Intelligent Fusion (APH-IF) Technology

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![Neo4j](https://img.shields.io/badge/Neo4j-008CC1?style=flat&logo=neo4j&logoColor=white)](https://neo4j.com/)

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
- **Version**: dev 0.0.1
- **Author**: Alexander Ricciardi (Omega.py)
- **Date**: August 2025

---

## Quick Start

### Prerequisites
- **Docker** and **Docker Compose** installed
- **8GB+ RAM** recommended
- **Ports 8000, 8501, 7474, 7687** available

### Launch APH-IF in 3 Steps

```bash
# 1. Clone the repository
git clone https://github.com/Omegapy/APH-IF-Dev.git
cd APH-IF-Dev-v1

# 2. Start all services
docker-compose up --build -d

# 3. Access the application
# Frontend: http://localhost:8501
# Backend API: http://localhost:8000/docs
# Neo4j Browser: http://localhost:7474
```

**That's it!** The APH-IF system is now running with all microservices.

---

## Table of Contents

- [Architecture Overview](#️-architecture-overview)
- [Docker Services](#-docker-services)
- [Service Endpoints](#-service-endpoints)
- [Core Technology](#-core-technology)
- [Configuration](#-configuration)
- [API Documentation](#-api-documentation)
- [Testing](#-testing)
- [Development](#️-development)
- [Background](#-background)

---

## Architecture Overview

APH-IF implements a **microservices architecture** with **containerized deployment**:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend      │    │    Database     │
│   (Streamlit)   │◄──►│   (FastAPI)     │◄──►│    (Neo4j)      │
│   Port: 8501    │    │   Port: 8000    │    │   Port: 7474    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                        │                        │
        ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ • Overview UI   │    │ • APH-IF Engine │    │ • Graph Storage │
│ • Bot Interface │    │ • Intelligent   │    │ • Vector Index  │
│ • Documentation │    │   Fusion        │    │ • Relationships │
│ • Monitoring    │    │ • Circuit Break │    │ • Embeddings    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Docker Services

APH-IF runs as **3 containerized microservices**:

| Service | Container | Image | Port | Health Check |
|---------|-----------|-------|------|--------------|
| **Frontend** | `aph_if_frontend` | `aph-if-dev-v1-aph-if-frontend` | `8501` | ✅ Healthy |
| **Backend** | `aph_if_backend` | `aph-if-dev-v1-aph-if-backend` | `8000` | ✅ Healthy |
| **Database** | `aph_if_neo4j` | `neo4j:5.15` | `7474, 7687` | ✅ Healthy |

### Container Details

#### Frontend Container (`aph_if_frontend`)
- **Technology**: Streamlit
- **Purpose**: User interface and bot interaction
- **Internal Communication**: `http://aph-if-backend:8000`
- **External Access**: `http://localhost:8501`

#### Backend Container (`aph_if_backend`)
- **Technology**: FastAPI + Uvicorn
- **Purpose**: APH-IF processing engine
- **Features**: Parallel HybridRAG, Intelligent Fusion, Circuit Breaker
- **External Access**: `http://localhost:8000`

#### Database Container (`aph_if_neo4j`)
- **Technology**: Neo4j Graph Database
- **Purpose**: Knowledge graph storage with vector embeddings
- **Browser Access**: `http://localhost:7474`
- **Bolt Protocol**: `bolt://localhost:7687`

---

## Service Endpoints

### Frontend (Port 8501)
```
http://localhost:8501
├── Overview Interface
├── Bot Chat Interface  
├── Documentation
└── System Monitoring
```

### Backend API (Port 8000)
```
http://localhost:8000
├── /                           # Service info
├── /docs                       # Swagger UI
├── /health                     # Health check
├── /parallel_hybrid/health     # Detailed health
└── /generate_parallel_hybrid   # Main processing endpoint
```

### Neo4j Database (Port 7474)
```
http://localhost:7474
├── Neo4j Browser Interface
├── Graph visualization
├── Cypher query console
└── Database administration
```

---

## Core Technology

### **Advanced Parallel HybridRAG (APH)**

**Traditional Sequential RAG:**
```python
if condition:
    results = vector_search(query)
else:
    results = graph_search(query)
```

**APH-IF Parallel Approach:**
```python
vector_task = vector_search(query)
graph_task = graph_search(query)
vector_results, graph_results = await asyncio.gather(vector_task, graph_task)
```

### **Intelligent Fusion (IF)**

**LLM-Powered Result Combination:**
```python
fused_response = intelligent_fusion_engine.fuse_results(
    vector_results=vector_results,
    graph_results=graph_results,
    original_query=query
)
```

### **Key Components**

1. **ParallelHybridRAGEngine**: Coordinates concurrent VectorRAG and GraphRAG execution
2. **IntelligentFusionEngine**: LLM-based result fusion and synthesis  
3. **CircuitBreaker**: Fault tolerance and graceful degradation
4. **ConfigManager**: Environment-based configuration management

---

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Database Configuration
NEO4J_URI=bolt://aph_if_neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# LLM Configuration  
OPENAI_API_KEY=your_openai_api_key
LLM_MODEL=gpt-4
LLM_TEMPERATURE=0.7

# Parallel HybridRAG Settings
PH_MAX_VECTOR_RESULTS=10
PH_MAX_GRAPH_RESULTS=10
PH_VECTOR_THRESHOLD=0.7
PH_GRAPH_DEPTH=3
PH_ENABLE_CACHING=true
PH_CACHE_TTL=3600

# Circuit Breaker Settings
CB_FAILURE_THRESHOLD=5
CB_RECOVERY_TIMEOUT=60
CB_EXPECTED_EXCEPTION=Exception
```

### Docker Compose Configuration

The `docker-compose.yml` handles:
- **Service orchestration**
- **Network configuration** 
- **Volume mounting**
- **Health checks**
- **Environment variables**

---

## API Documentation

### Main Processing Endpoint

**POST** `/generate_parallel_hybrid`

**Request:**
```json
{
  "query": "What is Advanced Parallel HybridRAG?",
  "session_id": "optional-session-id",
  "max_results": 10,
  "include_metadata": true
}
```

**Response:**
```json
{
  "response": "Generated intelligent response...",
  "session_id": "session-123",
  "processing_time": 1.234,
  "vector_results_count": 8,
  "graph_results_count": 6,
  "fusion_method": "intelligent",
  "metadata": {
    "vector_search_time": 0.456,
    "graph_search_time": 0.389,
    "total_results": 14,
    "timestamp": "2025-08-05T10:30:00Z"
  }
}
```

### Health Check Endpoints

**GET** `/health` - Basic service health  
**GET** `/parallel_hybrid/health` - Detailed component health

---

## Testing

### Manual Testing

```bash
# Test backend health
curl http://localhost:8000/health

# Test processing endpoint
curl -X POST "http://localhost:8000/generate_parallel_hybrid" \
  -H "Content-Type: application/json" \
  -d '{"query": "Test query", "max_results": 5}'

# Check service status
docker-compose ps
```

### Automated Testing

```bash
# Run test script
python test_docker_setup.py

# Check logs
docker-compose logs aph_if_backend
docker-compose logs aph_if_frontend
```

---

## Development

### Local Development Setup

```bash
# Clone repository
git clone https://github.com/Omegapy/APH-IF-Dev.git
cd APH-IF-Dev-v1

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Start services
docker-compose up --build
```

### Development Commands

```bash
# Rebuild specific service
docker-compose up --build -d aph-if-backend

# View logs
docker-compose logs -f aph_if_backend

# Stop all services
docker-compose down

# Clean rebuild
docker-compose down
docker-compose up --build -d
```

### Project Structure

```
APH-IF-Dev-v1/
├── backend/                    # Backend microservice
│   ├── main.py                # FastAPI application
│   ├── parallel_hybrid.py     # ParallelHybridRAGEngine
│   ├── context_fusion.py      # IntelligentFusionEngine
│   ├── config.py              # Configuration management
│   ├── llm.py                 # LLM integration
│   ├── circuit_breaker.py     # Fault tolerance
│   └── Dockerfile.backend     # Backend container
├── frontend/                   # Frontend microservice
│   ├── app.py                 # Streamlit application
│   └── Dockerfile.frontend    # Frontend container
├── docker-compose.yml         # Service orchestration
├── requirements.txt           # Python dependencies
└── README.md                  # This documentation
```

---

## Background

### Technology Innovation

**APH-IF** represents a significant advancement in RAG systems by:

1. **True Parallelism**: Concurrent execution vs. sequential processing
2. **Intelligent Fusion**: LLM-powered result synthesis vs. simple concatenation
3. **Microservices Architecture**: Scalable, maintainable, containerized deployment
4. **Fault Tolerance**: Circuit breaker patterns for production reliability




