# Run Initial Graph Build Module Documentation

## Overview

The `run_initial_graph_build.py` module is a configurable wrapper for the initial graph build process that provides environment-specific defaults, safety checks, and progress reporting. It serves as the primary entry point for building the APH-IF hybrid knowledge store with intelligent configuration management and validation.

## Purpose

This module provides:

- **Environment Validation**: Comprehensive checks for Neo4j, OpenAI, and PDF data prerequisites
- **Configuration Management**: Environment-specific defaults with test-mode conveniences
- **Safety Mechanisms**: Validation and confirmation for destructive operations
- **Progress Reporting**: Detailed status updates and configuration display
- **Error Handling**: Graceful failure handling with informative error messages

## Architecture

### Core Components

1. **Environment Validation**: Checks for required credentials and data availability
2. **Configuration Builder**: Creates environment-specific configuration maps
3. **Safety Checks**: Validates operations before execution
4. **Wrapper Integration**: Seamless integration with core `HybridStoreBuilder`
5. **Progress Monitoring**: Real-time status updates and final summaries

### Processing Flow

```
Environment Setup → Validation → Configuration → Graph Building → Summary
       ↓              ↓             ↓              ↓             ↓
   set_environment → Credentials → Config Map → HybridStoreBuilder → Results
```

## Configuration

### Environment Variables

#### Required Variables
- `NEO4J_URI`: Neo4j database connection URI
- `NEO4J_USERNAME`: Neo4j database username
- `NEO4J_PASSWORD`: Neo4j database password
- `OPENAI_API_KEY`: OpenAI API key for embeddings and entity extraction

#### Processing Configuration
- `PDF_DIR`: Input directory for PDF files
- `CLEAR_DB`: Clear existing database before processing (default: `false`)
- `MAX_DOCS`: Maximum number of documents to process
- `MAX_PAGES`: Maximum pages per document
- `CHUNK_SIZE_CHARS`: Character size for text chunks (default: `1000`)
- `CHUNK_OVERLAP_CHARS`: Character overlap between chunks (default: `200`)
- `EXTRACT_EVERY_N_CHUNKS`: Extract entities every N chunks (default: `2`)
- `OPENAI_MODEL`: OpenAI model for entity extraction (default: `gpt-5-nano`)

#### Environment Control
- `APP_ENV`: Application environment (`development` | `production`)
- `FORCE_TEST_DB`: Force test database usage (default: `false`)
- `VERBOSE`: Enable verbose logging (default: `false`)

### Environment-Specific Defaults

#### Development Mode
```python
config = {
    'PDF_DIR': 'processing/data_pdf',
    'CHUNK_SIZE_CHARS': '1000',
    'EXTRACT_EVERY_N_CHUNKS': '2',
    'CLEAR_DB': 'false'
}
```

#### Test Mode (FORCE_TEST_DB=true)
```python
config = {
    'PDF_DIR': 'processing/data_pdf_test',
    'CHUNK_SIZE_CHARS': '3000',        # Larger chunks for speed
    'EXTRACT_EVERY_N_CHUNKS': '5',     # Less frequent extraction
    'MAX_PAGES': '10',                 # Limited pages
    'CLEAR_DB': 'true'                 # Default to clearing
}
```

## Usage

### Command Line Usage

```powershell
# Basic execution
cd data_processing
uv run python -m processing.run_initial_graph_build

# With custom configuration
$env:CHUNK_SIZE_CHARS = "2000"
$env:MAX_DOCS = "5"
$env:VERBOSE = "true"
uv run python -m processing.run_initial_graph_build

# Clear database and rebuild
$env:CLEAR_DB = "true"
uv run python -m processing.run_initial_graph_build
```

### Test Mode Usage

```powershell
# Setup test environment
python ../set_environment.py --mode development --force-test-db true

# Run with test configuration
$env:FORCE_TEST_DB = "true"
$env:MAX_PAGES = "5"
uv run python -m processing.run_initial_graph_build

# Quick test with minimal data
$env:FORCE_TEST_DB = "true"
$env:MAX_DOCS = "2"
$env:MAX_PAGES = "3"
uv run python -m processing.run_initial_graph_build
```

### Programmatic Usage

```python
from processing.run_initial_graph_build import run_initial_graph_build

# Execute with current environment
result = run_initial_graph_build()
if result == 0:
    print("Graph build completed successfully")
else:
    print("Graph build failed")
```

## API Reference

### Core Functions

#### get_processing_config() -> Dict[str, str]

```python
def get_processing_config() -> Dict[str, str]:
    """Get processing configuration from environment variables.
    
    Returns:
        Dict[str, str]: Flat config for the initial build runner, including
                        environment, chunking, extraction, and limits.
    """
```

**Features:**
- Reads all relevant environment variables
- Applies environment-specific defaults
- Handles test mode configuration automatically
- Returns flat dictionary for easy environment export

#### validate_environment() -> bool

```python
def validate_environment() -> bool:
    """Validate that the environment is properly configured.
    
    Returns:
        bool: True if the runner can proceed; False otherwise.
    """
```

**Validation Checks:**
- Neo4j connection parameters (URI, username, password)
- OpenAI API key availability
- PDF directory existence and accessibility
- PDF file availability in the specified directory

#### run_initial_graph_build() -> int

```python
def run_initial_graph_build() -> int:
    """Run the initial graph build process.
    
    Returns:
        int: 0 on success; non-zero on failure conditions.
    """
```

**Process Flow:**
1. Display header and configuration
2. Validate environment prerequisites
3. Build and display configuration
4. Set environment variables for subprocess
5. Execute HybridStoreBuilder
6. Handle database clearing if requested
7. Process PDFs and create vector index
8. Display final summary and results

#### get_neo4j_config() -> Dict[str, str]

```python
def get_neo4j_config() -> Dict[str, str]:
    """Get Neo4j configuration from environment variables set by set_environment.py
    
    Returns:
        Dict containing 'uri', 'username', and 'password' keys
    """
```

## Configuration Display

The module provides comprehensive configuration display before execution:

```
APH-IF Initial Graph Build Runner
================================================================================

Current Configuration:
  Environment: development
  Test Database: true
  PDF Directory: processing/data_pdf_test
  Clear Database: true
  Max Documents: 2
  Max Pages: 10
  Chunk Size: 3000 chars
  Extract Every: 5 chunks
  Model: gpt-5-nano
  Verbose: true
  Database: bolt://localhost:7687
```

## Safety Features

### Environment Validation

```python
# Comprehensive validation before execution
if not validate_environment():
    return 1  # Exit with error code

# Checks include:
# - Neo4j credentials completeness
# - OpenAI API key presence
# - PDF directory existence
# - PDF file availability
```

### Test Mode Protection

```python
# Automatic test-friendly defaults
if force_test_db:
    config.update({
        'CHUNK_SIZE_CHARS': '3000',      # Faster processing
        'EXTRACT_EVERY_N_CHUNKS': '5',   # Reduced API calls
        'MAX_PAGES': '10',               # Limited scope
        'CLEAR_DB': 'true',              # Safe clearing
    })
```

### Database Clearing Confirmation

```python
# Clear database only when explicitly requested
if os.getenv("CLEAR_DB", "false").lower() == "true":
    print("Clearing existing database...")
    builder.graph.query("MATCH (n) DETACH DELETE n")
    print("[SUCCESS] Database cleared - starting fresh build")
```

## Error Handling

### Validation Errors

```python
# Neo4j configuration incomplete
if not all([config['uri'], config['username'], config['password']]):
    print("❌ Neo4j configuration incomplete")
    return False

# OpenAI API key missing
if not os.getenv('OPENAI_API_KEY'):
    print("❌ OPENAI_API_KEY not found in environment")
    return False

# PDF directory not found
if not full_pdf_path.exists():
    print(f"❌ PDF directory not found: {full_pdf_path}")
    return False
```

### Runtime Errors

```python
try:
    # Execute graph building process
    builder = HybridStoreBuilder()
    # ... processing ...
    return 0
    
except KeyboardInterrupt:
    print("\n⚠️  Process interrupted by user")
    return 1
    
except Exception as e:
    print(f"\n❌ Initial graph build failed: {e}")
    import traceback
    traceback.print_exc()
    return 1
```

## Integration Points

### With Environment Management

```powershell
# Setup environment first
python ../set_environment.py --mode development --force-test-db true

# Then run graph build
uv run python -m processing.run_initial_graph_build
```

### With Core Module

```python
# Direct integration with HybridStoreBuilder
from processing.initial_graph_build import HybridStoreBuilder

# Set environment variables from config
for key, value in config.items():
    os.environ[key] = value

# Create and run builder
builder = HybridStoreBuilder()
builder.process_directory(data_path)
builder.create_vector_index()
```

### With Pipeline Launcher

```python
# Used by launch_data_processing.py
from processing.run_initial_graph_build import run_initial_graph_build

def run_initial_step(config):
    # Set environment variables
    for key, value in config.items():
        os.environ[key] = value
    
    # Run initial graph build
    result = run_initial_graph_build()
    return result == 0
```

## Best Practices

### Development Workflow

1. **Setup Environment**: Use `set_environment.py` to configure database and API keys
2. **Start with Test Mode**: Always test with `FORCE_TEST_DB=true` first
3. **Validate Configuration**: Review displayed configuration before proceeding
4. **Monitor Progress**: Enable `VERBOSE=true` for detailed logging
5. **Verify Results**: Check final summary for completion status

### Production Deployment

1. **Backup Database**: Always backup before using `CLEAR_DB=true`
2. **Validate Environment**: Ensure production credentials are correct
3. **Monitor Resources**: Watch Neo4j memory and storage during processing
4. **Schedule Appropriately**: Run during low-traffic periods
5. **Verify Completion**: Check final summary and database state

### Troubleshooting

1. **Check Prerequisites**: Verify all environment variables are set
2. **Test Connectivity**: Validate Neo4j and OpenAI connections
3. **Review Configuration**: Ensure PDF directory and files exist
4. **Start Small**: Use `MAX_DOCS` and `MAX_PAGES` for testing
5. **Enable Logging**: Use `VERBOSE=true` for detailed error information

## Examples

### Basic Development Setup

```powershell
# Setup development environment
python ../set_environment.py --mode development

# Run with default settings
cd data_processing
uv run python -m processing.run_initial_graph_build
```

### Test Environment Setup

```powershell
# Setup test environment
python ../set_environment.py --mode development --force-test-db true

# Quick test with limited data
$env:FORCE_TEST_DB = "true"
$env:MAX_DOCS = "2"
$env:MAX_PAGES = "5"
$env:VERBOSE = "true"
uv run python -m processing.run_initial_graph_build
```

### Custom Configuration

```powershell
# Large chunks, less frequent entity extraction
$env:CHUNK_SIZE_CHARS = "2000"
$env:CHUNK_OVERLAP_CHARS = "400"
$env:EXTRACT_EVERY_N_CHUNKS = "5"
$env:OPENAI_MODEL = "gpt-5-nano"
uv run python -m processing.run_initial_graph_build
```

### Production Setup

```powershell
# Setup production environment (use with caution)
python ../set_environment.py --mode production

# Run with production data
$env:PDF_DIR = "processing/data_pdf"
$env:CLEAR_DB = "false"  # Don't clear existing data
uv run python -m processing.run_initial_graph_build
```

## Output Example

```
APH-IF Initial Graph Build Runner
================================================================================

Current Configuration:
  Environment: development
  Test Database: true
  PDF Directory: processing/data_pdf_test
  Clear Database: true
  Max Documents: unlimited
  Max Pages: 10
  Chunk Size: 3000 chars
  Extract Every: 5 chunks
  Model: gpt-5-nano
  Verbose: true
  Database: bolt://localhost:7687

✅ Found 3 PDF file(s) in processing/data_pdf_test

Clearing existing database...
[SUCCESS] Database cleared - starting fresh build

[Processing details...]

🎉 INITIAL GRAPH BUILD COMPLETED SUCCESSFULLY!
✅ Graph database has been populated with documents, chunks, and entities
```

This module provides a robust, safe, and user-friendly interface for building the APH-IF hybrid knowledge store with comprehensive validation, configuration management, and error handling.
