# APH-IF Data Processing Service Alpha Version 0.1.0

The data processing service provides comprehensive document ingestion, hybrid knowledge graph construction, and relationship discovery for the APH-IF system. It creates the foundational hybrid store combining GraphRAG and VectorRAG capabilities.

## Features

- **Hybrid Knowledge Store**: Creates both vector embeddings and graph relationships
- **Document Processing**: PDF parsing, text extraction, and intelligent chunking
- **Entity Extraction**: GPT-5-NANO powered entity and relationship discovery
- **Vector Embeddings**: OpenAI embedding generation for semantic search
- **Knowledge Graph**: Neo4j graph construction with intelligent relationships
- **Environment-aware Configuration**: Automatic Neo4j instance selection via `set_environment.py`
- **Native Windows Development**: Fast development with uv package manager
- **Unified Pipeline**: Complete processing pipeline with configurable steps
- **FastAPI Service**: HTTP service interface with health monitoring

## Quick Start

### Environment Setup

The data processing service uses a simplified two-mode environment system:

```bash
# Development mode (default) - uses NEO4J_URI from data_processing/.env
uv run python set_environment.py --mode development

# Test mode - uses NEO4J_URI_TEST for safe testing
uv run python set_environment.py --force-test-db true

# Check connectivity after setup
uv run python set_environment.py --mode development --check
```

### Running the Pipeline

```bash
# Complete data processing pipeline with monitoring
uv run python run_data_processing_with_monitoring.py

# Development with test database
uv run python set_environment.py --force-test-db true
uv run python run_data_processing_with_monitoring.py --test-mode

# Individual pipeline steps
uv run python run_data_processing_with_monitoring.py --initial-only
uv run python run_data_processing_with_monitoring.py --augmentation-only
```

## Configuration

### Environment Management

**Important**: Production mode has been removed from data_processing to simplify environment management. Only development and test modes are supported.

#### Development Mode
```bash
# Set development environment (default)
uv run python set_environment.py --mode development --force-test-db false

# Uses these variables from data_processing/.env:
# NEO4J_URI=neo4j+s://your-dev-instance.databases.neo4j.io
# NEO4J_USERNAME=neo4j  
# NEO4J_PASSWORD=your-dev-password
```

#### Test Mode
```bash
# Set test environment for explicit testing scenarios
uv run python set_environment.py --force-test-db true

# Uses these variables from data_processing/.env:
# NEO4J_URI_TEST=neo4j+s://your-test-instance.databases.neo4j.io
# NEO4J_USERNAME_TEST=neo4j
# NEO4J_PASSWORD_TEST=your-test-password
```

### Environment Variables

All configuration is managed through `data_processing/.env`:

```bash
# Environment Control
APP_ENV=development                    # Application mode
FORCE_TEST_DB=false                   # Override to use test database

# Neo4j Development Instance (Default)
NEO4J_URI=neo4j+s://dev-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-dev-password

# Neo4j Test Instance (Used with FORCE_TEST_DB=true)
NEO4J_URI_TEST=neo4j+s://test-instance.databases.neo4j.io
NEO4J_USERNAME_TEST=neo4j  
NEO4J_PASSWORD_TEST=your-test-password

# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-5-nano-2025-08-07

# Processing Configuration
PDF_DIR=processing/data_pdf           # Source PDF directory
EXTRACT_EVERY_N_CHUNKS=5             # API call frequency
CHUNK_SIZE_CHARS=3000                # Text chunk size
```

## Usage Examples

### Basic Workflows

```bash
# 1. Set up development environment and verify connectivity
cd data_processing
uv run python set_environment.py --mode development --check

# 2. Run complete pipeline with monitoring  
uv run python run_data_processing_with_monitoring.py

# 3. Check processing results
uv run python check_processed_documents.py
```

### Testing Workflows

```bash
# 1. Switch to test database
uv run python set_environment.py --force-test-db true --check

# 2. Run pipeline in test mode with smaller dataset
uv run python run_data_processing_with_monitoring.py --test-mode

# 3. Validate test results
uv run python check_processed_documents.py
```

### Development Workflows  

```bash
# Quick development iteration
uv run python set_environment.py --mode development
uv run python run_data_processing_with_monitoring.py --initial-only --dry-run

# Monitor specific processing step
uv run python run_data_processing_with_monitoring.py --augmentation-only --log-level detailed
```

## Troubleshooting

### Common Issues

1. **Database Connection Errors**
   ```bash
   # Verify environment configuration
   uv run python set_environment.py --preview
   
   # Test database connectivity
   uv run python set_environment.py --check
   ```

2. **Missing Environment Variables**
   ```bash
   # Check required variables are set
   uv run python set_environment.py --preview
   ```

3. **Advanced Monitoring Unavailable**
   - If you see "⚠️ Advanced monitoring disabled", the common package is not available
   - The script continues with basic logging - this is normal after common package removal
   - Basic monitoring files are still saved to the monitoring directory

### Environment Status Check

```bash
# Show current environment configuration
uv run python set_environment.py --preview

# Output example:
# ============================================================
# APH-IF DATA PROCESSING ENVIRONMENT STATUS
# ============================================================
# App Environment: DEVELOPMENT
# Database Mode: Development (NEO4J_URI)
# Force Test DB: false
# Verbose Logging: true
# Active Neo4j URI: neo4j+s://dev-instance.databases.neo4j.io
# Neo4j Username: neo4j
# ============================================================
```

