# Advanced Parallel Hybrid - Intelligent Fusion (APH-IF) Technology

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/uv-package%20manager-blue.svg)](https://github.com/astral-sh/uv)
[![Neo4j](https://img.shields.io/badge/Neo4j-008CC1?style=flat&logo=neo4j&logoColor=white)](https://neo4j.com/)


---

## Advanced Parallel HybridRAG - Intelligent Fusion Overview

Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) is a novel Retrieval Augmented Generation (RAG) system that differs from traditional RAG approaches by performing semantic and traversal searches concurrently, rather than sequentially, and fusing the the result using a LLM or a LRM to generate the final response.


## Advanced Parallel HybridRAG - Intelligent Fusion Overview

I am developing a new Retrieval-Augmented Generation (RAG) system, Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) is a novel Retrieval Augmented Generation (RAG) system that differs from traditional RAG approaches by performing semantic and traversal searches concurrently, rather than sequentially, and fusing the results using an LLM or an LRM to generate the final response.

---

### **Core Innovation: True Parallel Processing and Intelligent Fusion**

**Parallel HybridRAG (PH)**

Conventional HybridRAG systems often process queries sequentially, for instance: `if condition: vector_search() else: graph_search()`.   
In contrast, **APH-IF PH** performs true concurrent execution of multiple retrieval methods, for instance: `asyncio.gather(vector_task, graph_task)`.

This parallelism is achieved using asynchronous programming (asyncio) and multi-threading. Based on the user's prompt, the PH engine executes both VectorRAG (semantic search) and GraphRAG (traversal search) queries in parallel on a Knowledge Graph that uses vector embeddings as properties.

**Intelligent Context Fusion (IF)**

The retrieved results from the concurrent queries are then fused using an Intelligent Context Fusion (IF) engine. The IF engine uses a LLM or a LRM to combine the results from the concurrent queries into a single, coherent final response. For instance: `intelligent_context_fusion(vector_results, graph_results)`

---

This project is curently being developped in a different repository.  
The frontend module is being developed. 

---

## Backend module is in version Alpha 0.2.0

The backend module is available and finished, it is in this repository:

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
├── .env.example                   # Environment configuration
└── README.md                      # backend 
```
---

## Data Processing is in version Alpha 0.2.0

The data processing is available and finished, it is in this repository:

```
data_processing/
├── README.md                 # frontend
├── .env.example              # Environment configuration
├── config.py                 # Configuration loader
├── docling_adapter.py        # PDF → pages conversion
├── chunker.py                # Page-aware chunking
├── embeddings.py             # OpenAI embedding wrapper
├── neo4j_writer.py           # Database operations
├── run_ingest.py             # Main CLI runner (unified interface)
├── page_num_adjust.py        # Migration: adjust Chunk.page numbering
├── entities/
│   ├── __init__.py           # Entity package exports
│   ├── rules.py              # Regex patterns for legal entities
│   ├── normalizer.py         # Entity deduplication & canonicalization
│   ├── pipeline.py           # spaCy pipeline builder
│   ├── extract.py            # Main entity extraction orchestrator
│   ├── augment.py            # LLM-based relationship detection
│   ├── evaluator.py          # Evaluation harness
│   ├── patterns/
│   │   └── legal_patterns.jsonl  # spaCy EntityRuler patterns
│   ├── lexicons/
│   │   └── legal_terms.json      # PhraseMatcher dictionary
│   └── evaluation/
│       ├── labeled_samples.jsonl # Test data
│       └── test_patterns.py      # Unit tests
├── utils/
│   └── logging.py            # Logging utilities
├── data_pdf/                 # PDF files to process
└── monitoring_logs/          # Processing logs (auto-created)
```
---

This project is based on my MRCA APH-IF (Mining Regulatory Compliance Assistant) project, which is a web application that uses AHP-IF to provide quick, reliable, and easy access to MSHA (Mine Safety and Health Administration) regulations using natural language queries.

MRCA APH-IFGitHub rep.: https://github.com/Omegapy/MRCA-Advanced-Parallel-HybridRAG-Intelligent-Fusion

---
 
**Alexander Ricciardi (Omega.py)**

<i><a href="https://www.alexomegapy.com" target="_blank"><img width="25" height="25" src="https://github.com/user-attachments/assets/a8e0ea66-5d8f-43b3-8fff-2c3d74d57f53"></i>
<i><a href="https://www.alexomegapy.com" target="_blank"><img width="150" height="23" src="https://github.com/user-attachments/assets/caa139ba-6b78-403f-902b-84450ff4d563"></i>
[![Medium](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=whit)](https://medium.com/@alex.omegapy)
<i><a href="https://dev.to/alex_ricciardi" target="_blank"><img width="53" height="20" src="https://github.com/user-attachments/assets/3dee9933-d8c9-4a38-b32e-b7a3c55e7e97"></i>
[![Facebook](https://img.shields.io/badge/Facebook-%231877F2.svg?logo=Facebook&logoColor=white)](https://www.facebook.com/profile.php?id=100089638857137)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5.svg?logo=linkedin&logoColor=white)](https://linkedin.com/in/alex-ricciardi)
<i><a href="https://www.threads.net/@alexomegapy?hl=en" target="_blank"><img width="53" height="20" src="https://github.com/user-attachments/assets/58c9e833-4501-42e4-b4fe-39ffafba99b2"></i>
[![X](https://img.shields.io/badge/X-black.svg?logo=X&logoColor=white)](https://x.com/AlexOmegapy)
[![YouTube](https://img.shields.io/badge/YouTube-%23FF0000.svg?logo=YouTube&logoColor=white)](https://www.youtube.com/channel/UC4rMaQ7sqywMZkfS1xGh2AA)    
Date: 08/05/2025

---

## License & Credits

**© 2025 Alexander Samuel Ricciardi - All rights reserved.**

- **License**: Apache-2.0
- **Technology**: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF)
- **Author**: Alexander Ricciardi (Omega.py)
- **Date**: August 2025

---

My Links:   

<i><a href="https://www.alexomegapy.com" target="_blank"><img width="25" height="25" src="https://github.com/user-attachments/assets/a8e0ea66-5d8f-43b3-8fff-2c3d74d57f53"></i>
<i><a href="https://www.alexomegapy.com" target="_blank"><img width="150" height="23" src="https://github.com/user-attachments/assets/caa139ba-6b78-403f-902b-84450ff4d563"></i>
[![Medium](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=whit)](https://medium.com/@alex.omegapy)
<i><a href="https://dev.to/alex_ricciardi" target="_blank"><img width="53" height="20" src="https://github.com/user-attachments/assets/3dee9933-d8c9-4a38-b32e-b7a3c55e7e97"></i>
[![Facebook](https://img.shields.io/badge/Facebook-%231877F2.svg?logo=Facebook&logoColor=white)](https://www.facebook.com/profile.php?id=100089638857137)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5.svg?logo=linkedin&logoColor=white)](https://linkedin.com/in/alex-ricciardi)
<i><a href="https://www.threads.net/@alexomegapy?hl=en" target="_blank"><img width="53" height="20" src="https://github.com/user-attachments/assets/58c9e833-4501-42e4-b4fe-39ffafba99b2"></i>
[![X](https://img.shields.io/badge/X-black.svg?logo=X&logoColor=white)](https://x.com/AlexOmegapy)
[![YouTube](https://img.shields.io/badge/YouTube-%23FF0000.svg?logo=YouTube&logoColor=white)](https://www.youtube.com/channel/UC4rMaQ7sqywMZkfS1xGh2AA)    
