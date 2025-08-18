# APH-IF Data Processing Unit - Comprehensive Guide

## Overview

The APH-IF (Advanced Parallel HybridRAG - Intelligent Fusion) Data Processing Unit is a sophisticated pipeline that transforms regulatory PDF documents into a hybrid knowledge store combining GraphRAG and VectorRAG capabilities. This comprehensive guide covers the entire data processing ecosystem, from initial setup through production deployment.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
3. [Environment Management](#environment-management)
4. [Processing Pipeline](#processing-pipeline)
5. [Monitoring and Protection](#monitoring-and-protection)
6. [Configuration Management](#configuration-management)
7. [Deployment Workflows](#deployment-workflows)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)
10. [Integration Points](#integration-points)

## Architecture Overview

### System Architecture

The APH-IF data processing unit implements a hybrid knowledge store architecture:

```
PDF Documents → Text Processing → Hybrid Store Creation → Enhanced Graph
      ↓              ↓                    ↓                    ↓
   Extraction → Chunking/Embedding → GraphRAG + VectorRAG → Relationships
```

### Core Technologies

- **Neo4j**: Graph database for entity relationships and document structure
- **OpenAI**: Embeddings (text-embedding-3-small) and LLM processing (GPT-5)
- **LangChain**: Entity extraction and relationship discovery
- **Python**: Core processing language with async/await support
- **UV**: Modern package manager for dependency management

### Hybrid Knowledge Store Components

1. **GraphRAG**: Entity-relationship graphs for structured queries and traversal
2. **VectorRAG**: Semantic embeddings for similarity search and retrieval
3. **Document Nodes**: Hierarchical document organization with metadata
4. **Chunk Nodes**: Text segments with overlapping windows for context
5. **Entity Nodes**: Extracted entities with type classification
6. **Relationship Edges**: Semantic connections between entities and documents

## Core Components

### 1. Initial Graph Build (`initial_graph_build.py`)

**Purpose**: Creates the foundational hybrid knowledge store from PDF documents.

**Key Features**:
- PDF text extraction with page-level tracking
- Intelligent text chunking with configurable overlap
- OpenAI embedding generation for semantic search
- LLM-powered entity extraction with type classification
- Neo4j graph construction with optimized schema
- Vector index creation for fast similarity search

**Configuration**:
- `CHUNK_SIZE_CHARS`: Text chunk size (default: 1000, test: 3000)
- `EXTRACT_EVERY_N_CHUNKS`: Entity extraction frequency (default: 2, test: 5)
- `MAX_PAGES`: Page limit per document (test: 10)
- `OPENAI_MODEL`: LLM model for entity extraction (default: gpt-5-nano)

### 2. Document Embeddings (`compute_doc_embeddings.py`)

**Purpose**: Aggregates chunk embeddings into document-level representations.

**Key Features**:
- Element-wise mean computation of chunk vectors
- Document-level similarity enabling coarse-grained retrieval
- Defensive handling of mixed-dimension vectors
- Efficient streaming processing for memory optimization

**Output**: Document nodes with `doc_embedding` property (1536-dimension vectors)

### 3. Relationship Augmentation (`add_relationships_to_graph.py`)

**Purpose**: Discovers and adds semantic relationships between entities and documents.

**Key Features**:
- Jaccard similarity for entity relationships
- Cosine similarity for document relationships
- Evidence collection from supporting text chunks
- LLM-powered relationship classification and labeling
- Structured relationship merging with confidence scores

**Configuration**:
- `ENTITY_SIM_CUTOFF`: Entity similarity threshold (default: 0.25)
- `DOC_SIM_CUTOFF`: Document similarity threshold (default: 0.8)
- `ENTITY_LIMIT`: Maximum entity pairs to process (default: 1000)

### 4. Pipeline Orchestration (`launch_data_processing.py`)

**Purpose**: Unified interface for complete data processing workflow.

**Key Features**:
- Flexible execution control (complete pipeline or individual steps)
- Environment-aware configuration with test-mode safety
- Comprehensive validation and error handling
- Progress monitoring and status reporting

**Execution Modes**:
- `--initial-only`: Run only initial graph build
- `--augmentation-only`: Run only relationship augmentation
- `--test-mode`: Force test database with safe defaults
- `--dry-run`: Preview operations without database writes

### 5. Monitoring and Protection

#### Enhanced Monitoring (`run_data_processing_with_monitoring.py`)

**Purpose**: Comprehensive API monitoring and performance tracking.

**Key Features**:
- Real-time API call monitoring (OpenAI and Neo4j)
- Performance metrics and cost tracking
- Error analysis and categorization
- HTML report generation with optimization recommendations
- Live dashboard with configurable refresh rates

#### Timeout Protection (`run_with_timeout_protection.py`)

**Purpose**: Process protection against hangs and resource exhaustion.

**Key Features**:
- Multi-layer timeout protection (total time and output activity)
- Automatic process cleanup with graceful termination
- Environment stabilization for reliable processing
- Connection pool management to prevent exhaustion

### 6. Environment Management (`set_environment.py`)

**Purpose**: Centralized environment configuration and database selection.

**Key Features**:
- Safe switching between development, production, and test environments
- Automatic Neo4j credential selection based on mode
- Test mode override for isolated testing
- Configuration persistence via `.env` file updates

## Environment Management

### Environment Modes

#### Development Mode (Default)
```powershell
python set_environment.py --mode development
```
- Uses development Neo4j instance
- Safe for experimentation and daily development
- Default chunking and extraction settings

#### Production Mode
```powershell
python set_environment.py --mode production
```
- Uses production Neo4j instance
- Requires explicit activation
- Use with extreme caution

#### Test Mode
```powershell
python set_environment.py --force-test-db true
```
- Uses isolated test Neo4j instance
- Overrides environment mode
- Automatic test-friendly defaults:
  - Larger chunks (3000 chars) for faster processing
  - Reduced entity extraction frequency (every 5 chunks)
  - Limited pages (10) for quick testing
  - Automatic database clearing

### Environment Variables

#### Core Configuration
- `APP_ENV`: Environment mode (development/production)
- `FORCE_TEST_DB`: Test database override flag
- `NEO4J_URI/USERNAME/PASSWORD`: Active database credentials

#### Processing Parameters
- `PDF_DIR`: Input directory for PDF files
- `CLEAR_DB`: Clear database before processing
- `MAX_DOCS/MAX_PAGES`: Processing limits
- `CHUNK_SIZE_CHARS/CHUNK_OVERLAP_CHARS`: Text chunking configuration
- `EXTRACT_EVERY_N_CHUNKS`: Entity extraction frequency
- `OPENAI_MODEL`: LLM model selection

## Processing Pipeline

### Complete Pipeline Workflow

```powershell
# 1. Environment Setup
python set_environment.py --mode development --force-test-db true

# 2. Complete Pipeline Execution
cd data_processing
uv run python -m processing.launch_data_processing --test-mode

# 3. With Enhanced Monitoring
uv run python run_data_processing_with_monitoring.py --test-mode --watch

# 4. With Timeout Protection
uv run python run_with_timeout_protection.py --test-mode --timeout 30
```

### Step-by-Step Execution

```powershell
# Step 1: Initial Graph Build
uv run python -m processing.run_initial_graph_build

# Step 2: Document Embeddings (optional but recommended)
uv run python -m processing.compute_doc_embeddings

# Step 3: Relationship Augmentation
uv run python -m processing.run_relationship_augmentation
```

### Pipeline Stages

1. **Initialization**
   - Environment validation
   - Configuration loading
   - Database connection verification

2. **Document Processing**
   - PDF text extraction
   - Text chunking with overlap
   - Embedding generation
   - Entity extraction (configurable frequency)

3. **Graph Construction**
   - Document node creation
   - Chunk node creation with embeddings
   - Entity node creation with type classification
   - Relationship creation (document-chunk, chunk-entity)

4. **Vector Indexing**
   - Cosine similarity index creation
   - Document embedding aggregation

5. **Relationship Discovery**
   - Entity similarity computation (Jaccard)
   - Document similarity computation (cosine)
   - Evidence collection for relationships
   - LLM-powered relationship labeling

6. **Graph Enhancement**
   - Semantic relationship merging
   - Confidence score assignment
   - Final validation and optimization

## Monitoring and Protection

### Real-Time Monitoring

```powershell
# Enable live dashboard
uv run python run_data_processing_with_monitoring.py --watch --watch-refresh 5

# Generate comprehensive reports
uv run python run_data_processing_with_monitoring.py --generate-report
```

**Monitoring Features**:
- API call tracking (OpenAI and Neo4j)
- Performance metrics and response times
- Error categorization and frequency
- Cost tracking and token usage
- Resource utilization monitoring

### Timeout Protection

```powershell
# Basic timeout protection
uv run python run_with_timeout_protection.py --timeout 60

# Extended timeout for large datasets
uv run python run_with_timeout_protection.py --timeout 180
```

**Protection Features**:
- Total process timeout (configurable)
- Output activity monitoring (15-minute hang detection)
- Automatic process cleanup
- Connection pool management
- Environment stabilization

### Health Monitoring

```powershell
# Check system health
uv run python monitor_process_health.py

# Kill stuck processes
uv run python kill_stuck_process.py
```

## Configuration Management

### Test Mode Configuration

Automatic test-friendly defaults when `FORCE_TEST_DB=true`:

```python
{
    'PDF_DIR': 'processing/data_pdf_test',
    'CHUNK_SIZE_CHARS': '3000',        # Larger for speed
    'EXTRACT_EVERY_N_CHUNKS': '5',     # Less frequent
    'MAX_PAGES': '10',                 # Limited scope
    'CLEAR_DB': 'true',                # Safe clearing
    'ENTITY_SIM_CUTOFF': '0.1',        # Lower threshold
    'ENTITY_LIMIT': '10'               # Limited processing
}
```

### Production Configuration

Optimized settings for production workloads:

```python
{
    'PDF_DIR': 'processing/data_pdf',
    'CHUNK_SIZE_CHARS': '1000',        # Standard chunks
    'EXTRACT_EVERY_N_CHUNKS': '2',     # Regular extraction
    'CLEAR_DB': 'false',               # Preserve data
    'ENTITY_SIM_CUTOFF': '0.25',       # Quality threshold
    'AUGMENT_ENTITY': 'true',          # Enable relationships
    'AUGMENT_DOCUMENT': 'false'        # Conservative default
}
```

### Custom Configuration Examples

```powershell
# High-quality processing (slower)
$env:CHUNK_SIZE_CHARS = "500"
$env:EXTRACT_EVERY_N_CHUNKS = "1"
$env:ENTITY_SIM_CUTOFF = "0.4"

# Fast processing (lower quality)
$env:CHUNK_SIZE_CHARS = "2000"
$env:EXTRACT_EVERY_N_CHUNKS = "10"
$env:ENTITY_SIM_CUTOFF = "0.15"

# Large dataset processing
$env:MAX_DOCS = "100"
$env:SLEEP_BETWEEN_CHUNKS_MS = "100"
$env:TIMEOUT = "240"
```

## Deployment Workflows

### Development Workflow

```powershell
# 1. Setup development environment
python set_environment.py --mode development
cd data_processing && uv sync

# 2. Test with small dataset
python ../set_environment.py --force-test-db true
$env:MAX_DOCS = "2"; $env:MAX_PAGES = "5"
uv run python -m processing.launch_data_processing --test-mode

# 3. Full development run
python ../set_environment.py --force-test-db false
uv run python -m processing.launch_data_processing

# 4. Monitor and analyze
uv run python run_data_processing_with_monitoring.py --generate-report
```

### Testing Workflow

```powershell
# 1. Isolated test environment
python set_environment.py --mode development --force-test-db true

# 2. Quick validation test
$env:FORCE_TEST_DB = "true"
$env:MAX_DOCS = "1"; $env:MAX_PAGES = "3"
uv run python run_with_timeout_protection.py --test-mode --timeout 15

# 3. Comprehensive test
uv run python run_data_processing_with_monitoring.py --test-mode --watch

# 4. Verify results
# (Check Neo4j browser for test data)
```

### Production Deployment

```powershell
# 1. Backup production database
# (Perform Neo4j backup)

# 2. Set production environment
python set_environment.py --mode production

# 3. Verify configuration
python set_environment.py --preview

# 4. Run with full protection
uv run python run_with_timeout_protection.py --timeout 180 --monitoring-dir ./prod_logs

# 5. Monitor and validate
# (Check monitoring reports and database state)
```

## Best Practices

### Development Best Practices

1. **Always Use Test Mode First**: Test with `--force-test-db true` before production
2. **Start Small**: Use `MAX_DOCS` and `MAX_PAGES` for initial testing
3. **Monitor API Usage**: Track OpenAI costs and rate limits
4. **Enable Verbose Logging**: Use `VERBOSE=true` for debugging
5. **Validate Environment**: Check environment status before processing

### Production Best Practices

1. **Backup Before Processing**: Always backup production database
2. **Use Timeout Protection**: Run with timeout protection for reliability
3. **Monitor Resources**: Watch Neo4j memory and storage usage
4. **Schedule Appropriately**: Run during low-traffic periods
5. **Validate Results**: Verify processing completion and data quality

### Performance Optimization

1. **Chunk Size Tuning**: Balance quality vs. speed with chunk size
2. **Entity Extraction Frequency**: Adjust based on quality requirements
3. **Similarity Thresholds**: Tune for relationship quality vs. coverage
4. **Connection Management**: Use connection pooling for stability
5. **Memory Management**: Monitor and optimize memory usage

### Security Best Practices

1. **Environment Isolation**: Use separate databases for dev/test/prod
2. **Credential Management**: Secure API keys and database passwords
3. **Access Control**: Limit production database access
4. **Audit Logging**: Enable comprehensive logging for production
5. **Regular Backups**: Implement automated backup procedures

## Troubleshooting

### Common Issues

#### Environment Issues
```powershell
# Problem: Wrong database being used
# Solution: Verify and reset environment
python set_environment.py --preview
python set_environment.py --mode development --force-test-db true
```

#### Processing Hangs
```powershell
# Problem: Process appears stuck
# Solution: Use timeout protection and check health
uv run python run_with_timeout_protection.py --timeout 60
uv run python monitor_process_health.py
```

#### API Errors
```powershell
# Problem: OpenAI API failures
# Solution: Check API key and rate limits
echo $env:OPENAI_API_KEY
# Reduce processing frequency or add delays
$env:EXTRACT_EVERY_N_CHUNKS = "10"
$env:SLEEP_BETWEEN_CHUNKS_MS = "1000"
```

#### Memory Issues
```powershell
# Problem: Out of memory errors
# Solution: Reduce batch sizes and enable monitoring
$env:CHUNK_SIZE_CHARS = "2000"
$env:MAX_DOCS = "10"
uv run python run_data_processing_with_monitoring.py --watch
```

### Diagnostic Commands

```powershell
# Check environment status
python set_environment.py --preview

# Monitor system health
uv run python monitor_process_health.py

# Kill stuck processes
uv run python kill_stuck_process.py

# Check Neo4j connectivity
# (Use Neo4j Browser or cypher-shell)

# Verify API connectivity
# (Test OpenAI API key)
```

## Integration Points

### With APH-IF Backend

The processed knowledge graph integrates with the backend service for:
- Hybrid query processing (GraphRAG + VectorRAG)
- Entity relationship traversal
- Semantic similarity search
- Document retrieval and ranking

### With APH-IF Frontend

The frontend leverages the processed data for:
- Interactive query interfaces
- Relationship visualization
- Document exploration
- Search result presentation

### With External Systems

The data processing unit can integrate with:
- Document management systems
- Regulatory update feeds
- Compliance monitoring tools
- Analytics and reporting platforms

## Conclusion

The APH-IF Data Processing Unit provides a comprehensive, production-ready solution for transforming regulatory documents into an intelligent hybrid knowledge store. By following the workflows, best practices, and troubleshooting guidance in this documentation, teams can successfully deploy and maintain the system for reliable regulatory document processing and analysis.

For specific component details, refer to the individual module documentation:
- [UV Environment Guide](uv_environment_guide.md)
- [Initial Graph Build Usage](initial_graph_build_usage.md)
- [Compute Document Embeddings Usage](compute_doc_embeddings_usage.md)
- [Add Relationships to Graph Usage](add_relationships_to_graph_usage.md)
- [Run Initial Graph Build Usage](run_initial_graph_build_usage.md)
- [Run Relationship Augmentation Usage](run_relationship_augmentation_usage.md)
- [Launch Data Processing Usage](launch_data_processing_usage.md)
- [Run Data Processing with Monitoring Usage](run_data_processing_with_monitoring_usage.md)
- [Run with Timeout Protection Usage](run_with_timeout_protection_usage.md)
- [Set Environment Usage](set_environment_usage.md)
