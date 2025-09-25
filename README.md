# Advanced Parallel Hybrid - Intelligent Fusion (APH-IF) Technology

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/uv-package%20manager-blue.svg)](https://github.com/astral-sh/uv)
[![Neo4j](https://img.shields.io/badge/Neo4j-008CC1?style=flat&logo=neo4j&logoColor=white)](https://neo4j.com/)


---

## Advanced Parallel HybridRAG - Intelligent Fusion Overview

Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) is a novel Retrieval Augmented Generation (RAG) system that differs from traditional RAG approaches by performing semantic and traversal searches concurrently, rather than sequentially, and fusing the the result using a LLM or a LRM to generate the final response.


## Backend is in version Alpha 0.1.0

```
backend/
├── app/                           # FastAPI application package
│   ├── api/                       # Routers, API-specific models, lifecycle helpers, shared state
│   ├── core/                      # Configuration, async LLM client, CPU pool utilities
│   ├── models/                    # Pydantic response models and structured payloads
│   ├── monitoring/                # Performance/timing collectors, circuit breakers
│   ├── processing/                # Citation processor and fusion post-processing helpers
│   ├── schema/                    # Schema manager, exporters, and cache utilities
│   ├── search/                    # Parallel retrieval engines and fusion logic
│   ├── __init__.py
│   └── main.py                    # FastAPI app entrypoint wiring routers and middleware
├── console_semantic_demo.py       # VectorRAG console demonstration
├── console_traversal_demo.py      # LLM Structural Cypher demo
├── console_fusion_demo.py         # Parallel + fusion workflow demo
├── console_test_neo4j_connection.py # Neo4j connectivity tester
├── schema_cli.py                  # Schema management CLI utilities
└── README.md                     # backend 
```