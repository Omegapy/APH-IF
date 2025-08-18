# APH-IF Data Processing Service

The data processing service provides comprehensive document ingestion, hybrid knowledge graph construction, and relationship discovery for the APH-IF system. It creates the foundational "Figure-1" hybrid store combining GraphRAG and VectorRAG capabilities.

## Features

- **Hybrid Knowledge Store**: Creates both vector embeddings and graph relationships
- **Document Processing**: PDF parsing, text extraction, and intelligent chunking
- **Entity Extraction**: GPT-5 powered entity and relationship discovery
- **Vector Embeddings**: OpenAI embedding generation for semantic search
- **Knowledge Graph**: Neo4j graph construction with intelligent relationships
- **Environment-aware Configuration**: Automatic Neo4j instance selection via `set_environment.py`
- **Native Windows Development**: Fast development with uv package manager
- **Unified Pipeline**: Complete processing pipeline with configurable steps
- **FastAPI Service**: HTTP service interface with health monitoring

## Quick Start

```powershell
# From project root
cd data_processing

# Install dependencies (includes ML/AI packages)
uv sync

# Verify environment configuration
python ../check_environment.py

# Option 1: Start FastAPI service
uv run uvicorn processing.main:app --reload --port 8010

# Option 2: Run complete processing pipeline
uv run python -m processing.launch_data_processing --test-mode
```

## Environment Management

The data processing service integrates with the centralized environment management system:

### Environment Modes
- **Development**: Uses `NEO4J_URI_DEV` (safe for development work)
- **Production**: Uses `NEO4J_URI_PROD` (live data - use with extreme caution)
- **Testing**: Uses `NEO4J_URI_TEST` (only when `FORCE_TEST_DB=true`)

### Environment Control

```powershell
# Set development environment
python set_environment.py --mode development

# Enable test database for safe processing
python set_environment.py --mode development --force-test-db true

# Check current environment
python set_environment.py --status
```

⚠️ **Critical**: Always use test database mode when modifying graph structure or processing test data.

## Processing Pipeline

The data processing service provides a comprehensive pipeline for building hybrid knowledge stores:

### Phase 1: Initial Graph Construction
- **Module**: `initial_graph_build.py` / `run_initial_graph_build.py`
- **Purpose**: Build foundational hybrid knowledge store
- **Output**: Documents, chunks, entities with vector embeddings

### Phase 2: Document Embeddings
- **Module**: `compute_doc_embeddings.py`
- **Purpose**: Create document-level vector representations
- **Output**: Document nodes with aggregated embeddings

### Phase 3: Relationship Augmentation
- **Module**: `add_relationships_to_graph.py` / `run_relationship_augmentation.py`
- **Purpose**: Discover and add intelligent relationships
- **Output**: Enhanced graph with labeled entity and document relationships

### Phase 4: Unified Pipeline
- **Module**: `launch_data_processing.py`
- **Purpose**: Orchestrate complete or partial processing
- **Output**: Fully constructed hybrid knowledge store

## Usage Patterns

### Complete Pipeline (Recommended)

```powershell
# Run complete pipeline with test settings
$env:FORCE_TEST_DB = "true"
$env:MAX_PAGES = "5"
$env:CHUNK_SIZE_CHARS = "3000"
uv run python -m processing.launch_data_processing --test-mode
```

### Step-by-Step Processing

```powershell
# Step 1: Build initial graph
uv run python -m processing.run_initial_graph_build

# Step 2: Compute document embeddings
uv run python -m processing.compute_doc_embeddings

# Step 3: Add relationships
uv run python -m processing.run_relationship_augmentation
```

### Service Mode

```powershell
# Start FastAPI service
uv run uvicorn processing.main:app --reload --port 8010

# Check service health
curl http://localhost:8010/healthz
```

## Development Workflow

### Local Development

```powershell
# 1. Setup development environment
cd data_processing
python set_environment.py --mode development

# 2. Install dependencies
uv sync

# 3. Run quick test
$env:FORCE_TEST_DB = "true"
$env:MAX_DOCS = "1"
$env:MAX_PAGES = "3"
uv run python -m processing.launch_data_processing --test-mode

# 4. Start service for API access
uv run uvicorn processing.main:app --reload --port 8010
```

### Testing

```powershell
# Setup test environment
python set_environment.py --mode development --force-test-db true

# Run tests
uv run pytest

# Run specific test
uv run pytest tests/test_complete_graph_pipeline.py

# Cleanup test environment
python set_environment.py --mode development --force-test-db false
```

### Environment Validation

```powershell
# Check environment configuration
python ../check_environment.py

# Require test database for graph modifications
python ../check_environment.py --require-test

# Validate processing environment
python set_environment.py --status
```

## Configuration

### Environment Variables

#### Core Configuration (Managed by set_environment.py)
- `APP_ENV`: Environment mode (development/production)
- `FORCE_TEST_DB`: Test database override
- `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`: Active database credentials

#### Processing Configuration
- `PDF_DIR`: PDF document directory (default: `processing/data_pdf_test`)
- `CHUNK_SIZE_CHARS`: Text chunk size (default: 1000)
- `EXTRACT_EVERY_N_CHUNKS`: Entity extraction frequency (default: 2)
- `MAX_PAGES`: Maximum pages per document
- `MAX_DOCS`: Maximum documents to process
- `CLEAR_DB`: Clear database before processing

#### Relationship Configuration
- `ENTITY_SIM_CUTOFF`: Entity similarity threshold (default: 0.25)
- `DOC_SIM_CUTOFF`: Document similarity threshold (default: 0.8)
- `ENTITY_LIMIT`: Maximum entity pairs to process (default: 1000)
- `AUGMENT_ENTITY`: Enable entity relationships (default: true)
- `AUGMENT_DOCUMENT`: Enable document relationships (default: false)

#### LLM Configuration
- `OPENAI_API_KEY`: OpenAI API key
- `OPENAI_MODEL`: Model for entity extraction (default: gpt-5-nano)

### Configuration Examples

#### Quick Test Configuration
```powershell
$env:FORCE_TEST_DB = "true"
$env:MAX_DOCS = "1"
$env:MAX_PAGES = "3"
$env:CHUNK_SIZE_CHARS = "5000"
$env:EXTRACT_EVERY_N_CHUNKS = "10"
$env:CLEAR_DB = "true"
```

#### Development Configuration
```powershell
$env:CHUNK_SIZE_CHARS = "2000"
$env:EXTRACT_EVERY_N_CHUNKS = "3"
$env:MAX_DOCS = "10"
$env:VERBOSE = "true"
```

#### Production Configuration
```powershell
$env:APP_ENV = "production"
$env:PDF_DIR = "processing/data_pdf"
$env:CHUNK_SIZE_CHARS = "1500"
$env:EXTRACT_EVERY_N_CHUNKS = "1"
```

## Data Directories

### Production Data
- **Location**: `processing/data_pdf/`
- **Purpose**: Production PDF documents for live processing
- **Usage**: Set `PDF_DIR` environment variable to this path

### Test Data
- **Location**: `processing/data_pdf_test/`
- **Purpose**: Test PDF documents for development and testing
- **Usage**: Default directory for development and testing

## UV Package Management

### Dependency Management

```powershell
# Add ML/AI dependency
uv add langchain

# Add development dependency
uv add --dev pytest-asyncio

# Remove dependency
uv remove package-name

# Update dependencies
uv sync --upgrade

# Install from lock file
uv sync
```

### Virtual Environment

```powershell
# Run commands in virtual environment
uv run python -m processing.initial_graph_build
uv run pytest
uv run uvicorn processing.main:app --reload
```

## Dependencies

### Core Dependencies
- **LangChain & LangChain Community**: Document processing and LLM integration
- **OpenAI**: GPT-5 and embedding models
- **Neo4j Python Driver**: Graph database connectivity
- **FastAPI**: Service interface
- **Pydantic**: Data validation and settings

### Processing Dependencies
- **PyPDF2**: PDF document processing
- **NumPy, Pandas**: Data manipulation and analysis
- **Uvicorn**: ASGI server

### Development Dependencies
- **pytest**: Testing framework
- **pytest-asyncio**: Async testing support

See `pyproject.toml` and `uv.lock` for complete dependency specifications.

## Monitoring and Observability

### Health Checks

```bash
# Service health check
curl http://localhost:8010/healthz

# Health check with environment info
curl -s http://localhost:8010/healthz | jq .
```

### Progress Monitoring

```powershell
# Enable verbose logging
$env:VERBOSE = "true"
uv run python -m processing.launch_data_processing --test-mode
```

### Performance Monitoring

```powershell
# Monitor processing time and resource usage
uv run python -c "
import time
start = time.time()
# Run processing
end = time.time()
print(f'Processing took {end-start:.2f} seconds')
"
```

## Troubleshooting

### Common Issues

**Issue**: Environment validation failed
```
Solution: Check Neo4j credentials and environment configuration
python set_environment.py --status
```

**Issue**: No PDF files found
```
Solution: Check PDF directory and add test files
ls processing/data_pdf_test/
# Add PDF files to the directory
```

**Issue**: OpenAI API rate limits
```
Solution: Reduce processing frequency and batch sizes
$env:EXTRACT_EVERY_N_CHUNKS = "10"
$env:ENTITY_LIMIT = "50"
```

**Issue**: Memory issues with large documents
```
Solution: Use smaller chunks and limit processing
$env:CHUNK_SIZE_CHARS = "1000"
$env:MAX_PAGES = "10"
$env:MAX_DOCS = "5"
```

### Error Recovery

```powershell
# Reset to safe development environment
python set_environment.py --mode development --force-test-db true
python set_environment.py --status

# Clear database and restart processing
$env:CLEAR_DB = "true"
uv run python -m processing.launch_data_processing --test-mode
```

## Documentation

For detailed module documentation, see:
- [Data Processing Modules Overview](../documents/data_processing_modules.md)
- [Initial Graph Build Usage](../documents/initial_graph_build_usage.md)
- [Document Embeddings Usage](../documents/compute_doc_embeddings_usage.md)
- [Add Relationships Usage](../documents/add_relationships_to_graph_usage.md)
- [Launch Data Processing Usage](../documents/launch_data_processing_usage.md)
- [Environment Management Usage](../documents/set_environment_usage.md)

This service provides the core data processing capabilities for building and enhancing hybrid knowledge stores in the APH-IF system.