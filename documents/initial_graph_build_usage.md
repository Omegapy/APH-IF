# Initial Graph Build Module Documentation

## Overview

The `initial_graph_build.py` module is the core component of the APH-IF data processing pipeline responsible for building the initial hybrid knowledge store (Figure-1). It creates both GraphRAG (entity-relationship graphs) and VectorRAG (semantic embeddings) components from Title 30 CFR regulatory PDF documents.

## Purpose

This module builds the foundational hybrid knowledge store that enables:

- **GraphRAG**: Entity-relationship graphs for traversal and structured queries
- **VectorRAG**: Semantic embeddings for similarity search and retrieval
- **Parallel Processing**: Support for both graph and vector queries downstream

## Architecture

### Core Components

1. **HybridStoreBuilder Class**: Main orchestrator for hybrid store construction
2. **PDF Processing**: Document parsing and text extraction
3. **Text Chunking**: Intelligent text segmentation with overlap
4. **Vector Embeddings**: OpenAI embedding generation for semantic search
5. **Entity Extraction**: LLM-powered entity and relationship discovery
6. **Graph Construction**: Neo4j graph database population
7. **Vector Indexing**: Cosine similarity index creation

### Data Flow

```
PDF Documents → Text Extraction → Chunking → Embeddings + Entities → Neo4j Storage
                                     ↓
                              Vector Index Creation
```

## Configuration

### Environment Variables

#### Required Variables
- `OPENAI_API_KEY`: OpenAI API key for embeddings and entity extraction
- `NEO4J_URI`: Neo4j database connection URI
- `NEO4J_USERNAME`: Neo4j database username
- `NEO4J_PASSWORD`: Neo4j database password

#### Processing Configuration
- `PDF_DIR`: Input directory for PDF files (default: `processing/data_pdf`)
- `CLEAR_DB`: Clear existing database before processing (default: `false`)
- `MAX_DOCS`: Maximum number of documents to process (default: unlimited)
- `MAX_PAGES_PER_DOC`: Maximum pages per document (default: unlimited)

#### Chunking Configuration
- `CHUNK_SIZE_CHARS`: Character size for text chunks (default: `1000`)
- `CHUNK_OVERLAP_CHARS`: Character overlap between chunks (default: `200`)

#### Entity Extraction Configuration
- `EXTRACT_EVERY_N_CHUNKS`: Extract entities every N chunks (default: `1`)
- `DISABLE_ENTITY_EXTRACTION`: Disable entity extraction (default: `false`)
- `OPENAI_MODEL`: OpenAI model for entity extraction (default: `gpt-5-nano`)

#### Performance Configuration
- `SLEEP_BETWEEN_CHUNKS_MS`: Delay between chunk processing (default: `0`)
- `SLEEP_BETWEEN_BATCHES_MS`: Delay between batch processing (default: `0`)
- `VERBOSE`: Enable verbose logging (default: `false`)

### Test Mode Configuration

When `FORCE_TEST_DB=true`, the module automatically applies test-friendly defaults:

- `CHUNK_SIZE_CHARS`: `3000` (larger chunks for faster processing)
- `EXTRACT_EVERY_N_CHUNKS`: `5` (less frequent entity extraction)
- `MAX_PAGES`: `10` (limit pages for quick testing)
- `PDF_DIR`: `processing/data_pdf_test` (test data directory)
- `CLEAR_DB`: `true` (clear database for clean testing)

## Usage

### Basic Usage

```python
from processing.initial_graph_build import HybridStoreBuilder

# Create builder instance
builder = HybridStoreBuilder()

# Process directory of PDFs
builder.process_directory("path/to/pdf/directory")

# Create vector index for semantic search
builder.create_vector_index()

# Print final summary
builder.print_final_summary()
```

### Command Line Usage

```powershell
# Run from data_processing directory
cd data_processing

# Basic execution
uv run python -m processing.initial_graph_build

# With environment variables
$env:CHUNK_SIZE_CHARS = "2000"
$env:MAX_DOCS = "5"
uv run python -m processing.initial_graph_build

# Clear database and rebuild
$env:CLEAR_DB = "true"
uv run python -m processing.initial_graph_build
```

### Test Mode Usage

```powershell
# Setup test environment
python ../set_environment.py --mode development --force-test-db true

# Run with test configuration
$env:FORCE_TEST_DB = "true"
$env:MAX_PAGES = "5"
uv run python -m processing.initial_graph_build
```

## API Reference

### HybridStoreBuilder Class

#### Constructor

```python
def __init__(self):
    """Initialize the hybrid store builder with environment configuration."""
```

**Configuration loaded from environment:**
- Neo4j connection parameters
- OpenAI API configuration
- Processing limits and chunking settings
- Entity extraction configuration

#### Core Methods

##### process_directory(directory_path: str) -> None

```python
def process_directory(self, directory_path: str) -> None:
    """Process all PDF files to build the complete hybrid store with progress tracking.
    
    Args:
        directory_path (str): Path to directory containing PDF files
    """
```

**Features:**
- Scans directory for PDF files
- Applies document limits if configured
- Provides progress tracking and workload estimation
- Handles errors gracefully with detailed reporting

##### process_pdf(file_path: str) -> bool

```python
def process_pdf(self, file_path: str) -> bool:
    """Process PDF to create hybrid store components.
    
    Args:
        file_path (str): Full path to the PDF file to process
        
    Returns:
        bool: True if processing succeeds, False otherwise
    """
```

**Processing Steps:**
1. Extract text from PDF pages
2. Split text into overlapping chunks
3. Generate embeddings for each chunk
4. Extract entities using LLM (configurable frequency)
5. Store documents, chunks, and entities in Neo4j
6. Create relationships between components

##### create_vector_index() -> None

```python
def create_vector_index(self) -> None:
    """Create vector index for semantic similarity search."""
```

**Features:**
- Creates cosine similarity index on chunk embeddings
- Enables fast vector-based retrieval
- Handles index creation errors gracefully

##### extract_entities_msha(text: str) -> list[tuple[str, str]]

```python
def extract_entities_msha(self, text: str) -> list[tuple[str, str]]:
    """Extract entities using GPT-5 via LLMGraphTransformer.
    
    Args:
        text (str): Text chunk to extract entities from
        
    Returns:
        list[tuple[str, str]]: List of (entity_name, entity_type) pairs
    """
```

**Features:**
- Uses LangChain's LLMGraphTransformer
- Returns up to 8 entities per chunk
- Normalizes entity names and types
- Handles extraction errors gracefully

#### Utility Methods

##### print_progress_header() -> None

```python
def print_progress_header(self) -> None:
    """Print configuration and progress header."""
```

##### print_final_summary() -> None

```python
def print_final_summary(self) -> None:
    """Print final processing summary with statistics."""
```

## Graph Schema

### Node Types

#### Document Nodes
```cypher
(:Document {
    doc_id: string,      # Unique document identifier
    title: string,       # Document title
    source: string,      # Source file path
    created_at: integer  # Creation timestamp
})
```

#### Chunk Nodes
```cypher
(:Chunk {
    chunk_id: string,    # Unique chunk identifier
    text: string,        # Chunk text content
    page: integer,       # Source page number
    tokens: integer,     # Token count
    embedding: list      # Vector embedding (1536 dimensions)
})
```

#### Entity Nodes
```cypher
(:Entity {
    name: string,        # Normalized entity name (lowercase)
    type: string,        # Entity type (e.g., "regulation", "procedure")
    display: string      # Display name (original case)
})
```

### Relationship Types

- `(:Document)-[:HAS_CHUNK]->(:Chunk)`: Document contains chunk
- `(:Chunk)-[:PART_OF]->(:Document)`: Chunk belongs to document
- `(:Chunk)-[:HAS_ENTITY]->(:Entity)`: Chunk contains entity
- `(:Entity)-[:PART_OF]->(:Chunk)`: Entity belongs to chunk

## Performance Considerations

### Memory Usage

- **Embeddings**: Each chunk stores a 1536-dimension vector (~6KB per chunk)
- **Text Storage**: Full text content stored in chunk nodes
- **Batch Processing**: Processes one chunk at a time to manage memory

### Rate Limiting

- **OpenAI API**: Built-in retry logic with exponential backoff
- **Neo4j**: Configurable delays between operations
- **Entity Extraction**: Configurable frequency to reduce API calls

### Optimization Settings

```powershell
# For faster processing (fewer entities)
$env:EXTRACT_EVERY_N_CHUNKS = "10"
$env:DISABLE_ENTITY_EXTRACTION = "true"

# For better quality (more entities)
$env:EXTRACT_EVERY_N_CHUNKS = "1"
$env:CHUNK_SIZE_CHARS = "500"

# For large documents
$env:SLEEP_BETWEEN_CHUNKS_MS = "100"
$env:SLEEP_BETWEEN_BATCHES_MS = "500"
```

## Error Handling

### Common Errors

1. **Missing API Keys**: Validates OpenAI and Neo4j credentials
2. **PDF Processing Errors**: Continues with remaining files
3. **Embedding Failures**: Retries with exponential backoff
4. **Entity Extraction Errors**: Continues without entities
5. **Database Connection Issues**: Provides clear error messages

### Recovery Strategies

- **Partial Processing**: Can resume from where it left off
- **Database Clearing**: Use `CLEAR_DB=true` for fresh start
- **Incremental Processing**: Processes only new documents by default

## Integration Points

### With Other Modules

- **run_initial_graph_build.py**: Wrapper with environment-specific defaults
- **compute_doc_embeddings.py**: Can recompute embeddings if needed
- **add_relationships_to_graph.py**: Adds additional relationships to the graph
- **launch_data_processing.py**: Orchestrates complete pipeline

### With APH-IF System

- **Backend Service**: Queries the created graph and vector index
- **Frontend Service**: Displays results from hybrid retrieval
- **Environment Management**: Uses `set_environment.py` for configuration

## Best Practices

### Development

1. **Use Test Mode**: Always test with `FORCE_TEST_DB=true` first
2. **Start Small**: Use `MAX_DOCS` and `MAX_PAGES` for initial testing
3. **Monitor Progress**: Enable `VERBOSE=true` for detailed logging
4. **Clear Database**: Use `CLEAR_DB=true` for clean testing

### Production

1. **Backup Database**: Always backup before running with `CLEAR_DB=true`
2. **Monitor Resources**: Watch Neo4j memory and storage usage
3. **Rate Limiting**: Configure delays for large document sets
4. **Incremental Updates**: Process new documents without clearing existing data

### Troubleshooting

1. **Check Environment**: Verify all required environment variables
2. **Test Connections**: Validate Neo4j and OpenAI connectivity
3. **Review Logs**: Enable verbose logging for detailed error information
4. **Start Fresh**: Use `CLEAR_DB=true` if data corruption is suspected

## Examples

### Basic Processing

```powershell
# Setup environment
python ../set_environment.py --mode development

# Process with default settings
cd data_processing
uv run python -m processing.initial_graph_build
```

### Test Processing

```powershell
# Setup test environment
python ../set_environment.py --mode development --force-test-db true

# Quick test with limited data
$env:FORCE_TEST_DB = "true"
$env:MAX_DOCS = "2"
$env:MAX_PAGES = "5"
uv run python -m processing.initial_graph_build
```

### Custom Configuration

```powershell
# Large chunks, less frequent entity extraction
$env:CHUNK_SIZE_CHARS = "2000"
$env:CHUNK_OVERLAP_CHARS = "400"
$env:EXTRACT_EVERY_N_CHUNKS = "5"
$env:VERBOSE = "true"
uv run python -m processing.initial_graph_build
```

This module forms the foundation of the APH-IF hybrid knowledge store, enabling both graph-based and vector-based retrieval for comprehensive document analysis and query processing.
