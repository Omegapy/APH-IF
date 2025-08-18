# Launch Data Processing Module Documentation

## Overview

The `launch_data_processing.py` module is the unified pipeline launcher for the complete APH-IF data processing workflow. It orchestrates the entire process from initial graph building through relationship augmentation with environment-aware configuration, test-mode safety, and flexible execution controls.

## Purpose

This module provides:

- **Complete Pipeline Orchestration**: Unified interface for the entire data processing workflow
- **Flexible Execution Control**: Run complete pipeline or individual steps as needed
- **Environment-Aware Configuration**: Intelligent defaults based on environment settings
- **Test-Mode Safety**: Safe testing with automatic test database usage
- **Progress Monitoring**: Comprehensive status reporting and error handling
- **Configuration Management**: Centralized parameter control with environment integration

## Architecture

### Core Components

1. **Argument Parsing**: Command-line interface for execution control
2. **Configuration Management**: Environment-aware parameter building
3. **Environment Validation**: Comprehensive prerequisite checking
4. **Step Orchestration**: Coordinated execution of processing phases
5. **Progress Monitoring**: Real-time status updates and error reporting
6. **Safety Controls**: Test mode protection and state restoration

### Processing Pipeline

```
Environment Setup → Validation → Configuration → Step 1: Initial Build → Step 2: Augmentation → Completion
       ↓              ↓             ↓                    ↓                      ↓                ↓
   CLI Arguments → Prerequisites → Config Map → Graph Building → Relationship Discovery → Final Status
```

## Configuration

### Command Line Arguments

#### Execution Control
- `--initial-only`: Run only the initial graph build step
- `--augmentation-only`: Run only the relationship augmentation step
- `--skip-initial`: Skip the initial graph build step
- `--skip-augmentation`: Skip the relationship augmentation step

#### Safety and Testing
- `--test-mode`: Force test database usage with safe defaults
- `--dry-run`: Preview operations without database writes

### Environment Variables

#### Required Variables
- `NEO4J_URI`: Neo4j database connection URI
- `NEO4J_USERNAME`: Neo4j database username
- `NEO4J_PASSWORD`: Neo4j database password
- `OPENAI_API_KEY`: OpenAI API key for embeddings and LLM processing

#### Processing Configuration
- `PDF_DIR`: Input directory for PDF files
- `CLEAR_DB`: Clear existing database before processing
- `MAX_DOCS`: Maximum number of documents to process
- `MAX_PAGES`: Maximum pages per document
- `CHUNK_SIZE_CHARS`: Character size for text chunks
- `EXTRACT_EVERY_N_CHUNKS`: Extract entities every N chunks
- `OPENAI_MODEL`: OpenAI model for processing

#### Augmentation Configuration
- `AUGMENT_ENTITY`: Enable entity relationship discovery
- `AUGMENT_DOCUMENT`: Enable document relationship discovery
- `ENTITY_SIM_CUTOFF`: Entity similarity threshold
- `ENTITY_LIMIT`: Maximum entity pairs to process
- `DRY_RUN`: Preview relationships without writing

### Environment-Specific Defaults

#### Development Mode
```python
config = {
    'PDF_DIR': 'processing/data_pdf',
    'CHUNK_SIZE_CHARS': '1000',
    'EXTRACT_EVERY_N_CHUNKS': '2',
    'CLEAR_DB': 'false',
    'AUGMENT_ENTITY': 'true',
    'AUGMENT_DOCUMENT': 'false'
}
```

#### Test Mode (--test-mode or FORCE_TEST_DB=true)
```python
config = {
    'PDF_DIR': 'processing/data_pdf_test',
    'CHUNK_SIZE_CHARS': '3000',        # Larger chunks for speed
    'EXTRACT_EVERY_N_CHUNKS': '5',     # Less frequent extraction
    'MAX_PAGES': '10',                 # Limited pages
    'CLEAR_DB': 'true',                # Safe clearing
    'ENTITY_SIM_CUTOFF': '0.1',        # Lower threshold
    'ENTITY_LIMIT': '10'               # Limited processing
}
```

## Usage

### Complete Pipeline

```powershell
# Run complete pipeline with current environment
cd data_processing
uv run python -m processing.launch_data_processing

# Run complete pipeline in test mode
uv run python -m processing.launch_data_processing --test-mode

# Run complete pipeline with dry run preview
uv run python -m processing.launch_data_processing --dry-run
```

### Individual Steps

```powershell
# Run only initial graph build
uv run python -m processing.launch_data_processing --initial-only

# Run only relationship augmentation
uv run python -m processing.launch_data_processing --augmentation-only

# Run complete pipeline skipping initial build
uv run python -m processing.launch_data_processing --skip-initial
```

### Test Mode Usage

```powershell
# Setup test environment
python ../set_environment.py --mode development --force-test-db true

# Run complete pipeline in test mode
uv run python -m processing.launch_data_processing --test-mode

# Run with custom test settings
$env:MAX_DOCS = "2"
$env:MAX_PAGES = "5"
uv run python -m processing.launch_data_processing --test-mode
```

### Custom Configuration

```powershell
# Custom chunking and extraction settings
$env:CHUNK_SIZE_CHARS = "2000"
$env:EXTRACT_EVERY_N_CHUNKS = "3"
$env:VERBOSE = "true"
uv run python -m processing.launch_data_processing

# Custom relationship discovery settings
$env:AUGMENT_ENTITY = "true"
$env:AUGMENT_DOCUMENT = "true"
$env:ENTITY_SIM_CUTOFF = "0.3"
uv run python -m processing.launch_data_processing --augmentation-only
```

## API Reference

### Core Functions

#### parse_arguments() -> argparse.Namespace

```python
def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for pipeline control.
    
    Returns:
        argparse.Namespace: Parsed flags controlling pipeline behavior
    """
```

**Available Arguments:**
- `--initial-only`: Run only initial graph build
- `--augmentation-only`: Run only relationship augmentation
- `--skip-initial`: Skip initial graph build step
- `--skip-augmentation`: Skip relationship augmentation step
- `--test-mode`: Force test database usage
- `--dry-run`: Preview operations without writes

#### get_pipeline_config(args: argparse.Namespace) -> Dict[str, str]

```python
def get_pipeline_config(args: argparse.Namespace) -> Dict[str, str]:
    """Get complete pipeline configuration.
    
    Precedence:
        1) Current environment settings (via set_environment.py and process env)
        2) Test-mode overrides (when --test-mode is passed or FORCE_TEST_DB is true)
    
    Returns:
        Dict[str, str]: A flat config map for both steps (initial + augmentation)
    """
```

#### validate_pipeline_environment() -> bool

```python
def validate_pipeline_environment() -> bool:
    """Validate that the environment is ready for pipeline execution.
    
    Validates:
        - Neo4j credentials (uri/username/password) are available
        - OPENAI_API_KEY is set for model access
    
    Returns:
        bool: True if ready, False otherwise (with printed diagnostics)
    """
```

#### run_initial_step(config: Dict[str, str]) -> bool

```python
def run_initial_step(config: Dict[str, str]) -> bool:
    """Run the initial graph build step.
    
    Behavior:
        - Exports the provided configuration into the process env
        - Invokes run_initial_graph_build() which performs the full build
    
    Returns:
        bool: True on success, False on failure
    """
```

#### run_augmentation_step(config: Dict[str, str]) -> bool

```python
def run_augmentation_step(config: Dict[str, str]) -> bool:
    """Run the relationship augmentation step.
    
    Behavior:
        - Exports the provided configuration into the process env
        - Invokes the augmentation runner which handles similarity + labeling
    
    Returns:
        bool: True on success, False on failure
    """
```

#### main() -> int

```python
def main() -> int:
    """Main pipeline execution function.
    
    Returns:
        int: 0 on success, non-zero on failure
    """
```

## Pipeline Flow

### Step 1: Initial Graph Build

```python
# Configuration export
for key, value in config.items():
    os.environ[key] = value

# Execute initial graph build
result = run_initial_graph_build()

# Process includes:
# - PDF text extraction and chunking
# - Vector embedding generation
# - Entity extraction (optional)
# - Graph node and relationship creation
# - Vector index creation
```

### Step 2: Relationship Augmentation

```python
# Configuration export
for key, value in config.items():
    os.environ[key] = value

# Execute relationship augmentation
run_augmentation()

# Process includes:
# - Entity similarity computation (Jaccard)
# - Document similarity computation (cosine)
# - Evidence collection for relationships
# - LLM-powered relationship labeling
# - Graph relationship merging
```

## Configuration Display

The module provides comprehensive configuration display before execution:

```
📋 Pipeline Configuration:
  Environment: development
  Test Database: true
  Dry Run: false
  Database: bolt://localhost:7687
  Run Initial Build: true
  Run Augmentation: true

================================================================================
🏗️  STEP 1: Initial Graph Build
================================================================================
[Initial build progress...]

================================================================================
🔗 STEP 2: Relationship Augmentation
================================================================================
[Augmentation progress...]

================================================================================
🎉 COMPLETE DATA PROCESSING PIPELINE SUCCESSFUL!
================================================================================
✅ Knowledge graph has been built and augmented
✅ Ready for querying and retrieval operations
```

## Safety Features

### Test Mode Protection

```python
# Automatic test-friendly defaults
if force_test_db:
    config.update({
        'CHUNK_SIZE_CHARS': '3000',      # Faster processing
        'EXTRACT_EVERY_N_CHUNKS': '5',   # Reduced API calls
        'MAX_PAGES': '10',               # Limited scope
        'CLEAR_DB': 'true',              # Safe clearing
        'ENTITY_SIM_CUTOFF': '0.1',      # Lower threshold
        'ENTITY_LIMIT': '10'             # Limited processing
    })
```

### Environment Validation

```python
# Comprehensive validation before execution
if not validate_pipeline_environment():
    return 1

# Checks include:
# - Neo4j credentials completeness
# - OpenAI API key presence
```

### State Restoration

```python
# Store and restore original test mode
original_force_test_db = os.getenv('FORCE_TEST_DB', 'false').lower() == 'true'

try:
    # Pipeline execution
    pass
finally:
    # Restore original test mode if it was changed
    if args.test_mode:
        os.environ['FORCE_TEST_DB'] = str(original_force_test_db).lower()
```

## Error Handling

### Validation Errors

```python
# Environment validation
if not validate_pipeline_environment():
    print("❌ Environment validation failed")
    return 1

# Step execution errors
if not run_initial_step(config):
    print("❌ Pipeline failed at initial graph build step")
    return 1
```

### Step Failures

```python
# Individual step error handling
try:
    result = run_initial_graph_build()
    if result == 0:
        print("✅ Initial graph build completed successfully")
        return True
    else:
        print(f"❌ Initial graph build failed with return code: {result}")
        return False
except Exception as e:
    print(f"❌ Initial graph build failed: {e}")
    return False
```

## Integration Points

### With Environment Management

```powershell
# Setup environment first
python ../set_environment.py --mode development --force-test-db true

# Then run pipeline
uv run python -m processing.launch_data_processing --test-mode
```

### With Individual Modules

```python
# Integrates with core processing modules
from processing.run_initial_graph_build import run_initial_graph_build
from processing.run_relationship_augmentation import main as run_augmentation

# Orchestrates their execution with shared configuration
```

### With Monitoring Systems

```python
# Returns appropriate exit codes for CI/CD
# 0 = success, non-zero = failure
result = main()
sys.exit(result)
```

## Best Practices

### Development Workflow

1. **Start with Test Mode**: Always use `--test-mode` for initial testing
2. **Validate Environment**: Ensure all credentials and data are available
3. **Monitor Progress**: Watch console output for status updates
4. **Use Dry Run**: Preview operations with `--dry-run` before committing
5. **Step-by-Step Testing**: Use `--initial-only` and `--augmentation-only` for debugging

### Production Deployment

1. **Backup Database**: Always backup before running with `CLEAR_DB=true`
2. **Validate Configuration**: Review displayed configuration before proceeding
3. **Monitor Resources**: Watch Neo4j memory and API usage during processing
4. **Schedule Appropriately**: Run during low-traffic periods
5. **Verify Completion**: Check final status and database state

### Troubleshooting

1. **Check Prerequisites**: Verify all environment variables are set
2. **Test Individual Steps**: Use step-specific flags to isolate issues
3. **Review Configuration**: Ensure parameters are appropriate for data size
4. **Enable Verbose Logging**: Use `VERBOSE=true` for detailed information
5. **Start Small**: Use `MAX_DOCS` and `MAX_PAGES` for testing

## Examples

### Complete Pipeline Examples

```powershell
# Basic complete pipeline
cd data_processing
uv run python -m processing.launch_data_processing

# Test mode with safe defaults
uv run python -m processing.launch_data_processing --test-mode

# Production run with custom settings
$env:MAX_DOCS = "100"
$env:CHUNK_SIZE_CHARS = "1500"
uv run python -m processing.launch_data_processing
```

### Step-Specific Examples

```powershell
# Only build initial graph
uv run python -m processing.launch_data_processing --initial-only --test-mode

# Only run relationship augmentation
uv run python -m processing.launch_data_processing --augmentation-only

# Skip initial build (if already done)
uv run python -m processing.launch_data_processing --skip-initial
```

### Testing and Development

```powershell
# Quick test with minimal data
$env:FORCE_TEST_DB = "true"
$env:MAX_DOCS = "1"
$env:MAX_PAGES = "3"
uv run python -m processing.launch_data_processing --test-mode

# Dry run preview
uv run python -m processing.launch_data_processing --dry-run

# Debug specific step
uv run python -m processing.launch_data_processing --initial-only --test-mode
```

This module provides the primary interface for APH-IF data processing, orchestrating the complete workflow from raw PDFs to an enhanced hybrid knowledge graph ready for intelligent querying and retrieval.
