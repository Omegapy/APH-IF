# Run Relationship Augmentation Module Documentation

## Overview

The `run_relationship_augmentation.py` module orchestrates relationship discovery and augmentation using the utilities in `add_relationships_to_graph.py`. It provides environment-aware configuration, safety mechanisms, and intelligent relationship discovery between entities and documents in the APH-IF hybrid knowledge store.

## Purpose

This module enhances the knowledge graph by:

- **Entity Relationship Discovery**: Finding semantic relationships between entities using Jaccard similarity
- **Document Relationship Discovery**: Finding relationships between documents using cosine similarity  
- **LLM-Powered Classification**: Using GPT-5 to classify and label discovered relationships
- **Configurable Processing**: Environment-driven configuration with safety controls
- **Dry-Run Capabilities**: Testing relationship discovery without database modifications

## Architecture

### Core Components

1. **Configuration Management**: Environment-driven parameter control
2. **Precondition Validation**: Graph readiness and schema validation
3. **Similarity Processing**: Jaccard (entities) and cosine (documents) similarity computation
4. **Evidence Collection**: Supporting text chunk gathering for relationship validation
5. **LLM Integration**: OpenAI GPT-5 for relationship classification and labeling
6. **Safe Merging**: Controlled insertion of relationships with validation

### Processing Flow

```
Environment Setup → Preconditions → Similarity Computation → Evidence Collection → LLM Labeling → Graph Merging
       ↓               ↓                    ↓                      ↓                ↓              ↓
   Parameters → Graph Validation → Candidate Pairs → Supporting Chunks → Classified Relations → Enhanced Graph
```

## Configuration

### Environment Variables

#### Required Variables
- `NEO4J_URI`: Neo4j database connection URI
- `NEO4J_USERNAME`: Neo4j database username  
- `NEO4J_PASSWORD`: Neo4j database password
- `OPENAI_API_KEY`: OpenAI API key for LLM labeling

#### Processing Control
- `AUGMENT_ENTITY`: Enable entity relationship discovery (default: `true`)
- `AUGMENT_DOCUMENT`: Enable document relationship discovery (default: `false`)
- `DRY_RUN`: Perform discovery without database writes (default: `false`)

#### Entity Processing Parameters
- `ENTITY_SIM_CUTOFF`: Minimum Jaccard similarity threshold (default: `0.25`)
- `ENTITY_LIMIT`: Maximum entity pairs to process (default: `1000`)

#### Document Processing Parameters  
- `DOC_SIM_CUTOFF`: Minimum cosine similarity threshold (default: `0.8`)

#### LLM Configuration
- `OPENAI_MODEL`: OpenAI model for relationship labeling (default: `gpt-5-nano`)

### Prerequisites

1. **Initial Graph Build**: Must have completed with entities and documents
2. **Document Embeddings**: Document-level embeddings must exist (from `compute_doc_embeddings.py`)
3. **Graph Schema**: Proper constraints and indexes must be in place
4. **API Access**: Valid OpenAI API key for LLM processing

## Usage

### Command Line Usage

```powershell
# Basic execution (entity relationships only)
cd data_processing
uv run python -m processing.run_relationship_augmentation

# Enable both entity and document relationships
$env:AUGMENT_ENTITY = "true"
$env:AUGMENT_DOCUMENT = "true"
uv run python -m processing.run_relationship_augmentation

# Dry run to preview relationships without writing
$env:DRY_RUN = "true"
uv run python -m processing.run_relationship_augmentation
```

### Custom Configuration

```powershell
# Entity-only with custom thresholds
$env:AUGMENT_ENTITY = "true"
$env:AUGMENT_DOCUMENT = "false"
$env:ENTITY_SIM_CUTOFF = "0.15"
$env:ENTITY_LIMIT = "50"
uv run python -m processing.run_relationship_augmentation

# Document relationships with high similarity threshold
$env:AUGMENT_ENTITY = "false"
$env:AUGMENT_DOCUMENT = "true"
$env:DOC_SIM_CUTOFF = "0.85"
uv run python -m processing.run_relationship_augmentation
```

### Test Environment Usage

```powershell
# Setup test environment
python ../set_environment.py --mode development --force-test-db true

# Run with test-friendly settings
$env:FORCE_TEST_DB = "true"
$env:ENTITY_SIM_CUTOFF = "0.1"
$env:ENTITY_LIMIT = "10"
$env:DRY_RUN = "true"
uv run python -m processing.run_relationship_augmentation
```

## API Reference

### Core Functions

#### env_bool(name: str, default: bool) -> bool

```python
def env_bool(name: str, default: bool) -> bool:
    """Parse a boolean environment variable with common truthy values.
    
    Truthy values: {"1", "true", "yes", "on"} (case-insensitive)
    
    Args:
        name: Environment variable name to read
        default: Fallback value if the variable is not set
        
    Returns:
        bool: Parsed boolean value
    """
```

#### main() -> int

```python
def main() -> int:
    """Run relationship augmentation based on current environment settings.
    
    Flow:
        - Initialize GraphRelationshipAdder and run preconditions
        - If AUGMENT_ENTITY=true, generate entity candidates via Jaccard over
          shared chunk references, collect evidence, label via LLM, and MERGE
          unless DRY_RUN=true
        - If AUGMENT_DOCUMENT=true, generate document candidates via cosine
          similarity over document embeddings, collect evidence, label via LLM,
          and MERGE unless DRY_RUN=true
    
    Returns:
        int: 0 on success
    """
```

### Processing Flow Details

#### Entity Relationship Processing

```python
if augment_entity:
    # 1. Generate candidate pairs using Jaccard similarity
    entity_pairs = stream_entity_similarity(
        adder.run_cypher,
        similarity_cutoff=entity_sim_cutoff,
        limit=entity_limit
    )
    
    # 2. Build evidence-backed inputs
    batch = []
    for row in entity_pairs:
        e1_id = int(row["e1_id"])
        e2_id = int(row["e2_id"])
        similarity = float(row["similarity"])
        
        # Collect supporting evidence
        evidence = fetch_entity_evidence(adder.run_cypher, e1_id, e2_id)
        
        batch.append(RelationInput(
            domain="ENTITY",
            subject_id=e1_id,
            object_id=e2_id,
            evidence_chunks=evidence,
            seed_score=similarity
        ))
    
    # 3. LLM labeling and merging
    outputs = label_with_llm(batch)
    if not dry_run:
        merged = merge_entity_relations(adder.run_cypher, outputs)
```

#### Document Relationship Processing

```python
if augment_document:
    # 1. Generate candidate pairs using cosine similarity
    doc_pairs = stream_doc_similarity(
        adder.run_cypher,
        similarity_cutoff=doc_sim_cutoff
    )
    
    # 2. Build evidence-backed inputs
    batch = []
    for row in doc_pairs:
        d1_id = int(row["d1_id"])
        d2_id = int(row["d2_id"])
        similarity = float(row["similarity"])
        
        # Collect supporting evidence
        evidence = fetch_document_evidence(adder.run_cypher, d1_id, d2_id)
        
        batch.append(RelationInput(
            domain="DOCUMENT",
            subject_id=d1_id,
            object_id=d2_id,
            evidence_chunks=evidence,
            seed_score=similarity
        ))
    
    # 3. LLM labeling and merging
    outputs = label_with_llm(batch)
    if not dry_run:
        merged = merge_document_relations(adder.run_cypher, outputs)
```

## Configuration Examples

### Conservative Settings (Production)

```powershell
# High-quality relationships only
$env:AUGMENT_ENTITY = "true"
$env:AUGMENT_DOCUMENT = "false"
$env:ENTITY_SIM_CUTOFF = "0.4"
$env:ENTITY_LIMIT = "100"
$env:DRY_RUN = "false"
```

### Exploratory Settings (Development)

```powershell
# Lower thresholds for discovery
$env:AUGMENT_ENTITY = "true"
$env:AUGMENT_DOCUMENT = "true"
$env:ENTITY_SIM_CUTOFF = "0.15"
$env:ENTITY_LIMIT = "500"
$env:DOC_SIM_CUTOFF = "0.7"
$env:DRY_RUN = "true"  # Preview first
```

### Test Settings

```powershell
# Fast testing with minimal data
$env:FORCE_TEST_DB = "true"
$env:AUGMENT_ENTITY = "true"
$env:AUGMENT_DOCUMENT = "false"
$env:ENTITY_SIM_CUTOFF = "0.1"
$env:ENTITY_LIMIT = "10"
$env:DRY_RUN = "true"
```

## Safety Features

### Precondition Validation

```python
# Validates before processing
adder.run_preconditions()

# Checks include:
# - Neo4j connectivity
# - Required node types exist
# - Vector indexes are present
# - Document embeddings available (for document processing)
```

### Dry Run Mode

```python
# Preview relationships without database writes
if dry_run:
    print(f"Dry-run: prepared {len(outputs)} ENTITY relations (not merged)")
else:
    merged = merge_entity_relations(adder.run_cypher, outputs)
    print(f"Merged {merged} ENTITY relations")
```

### Environment-Aware Defaults

```python
# Safer defaults for different environments
augment_entity = env_bool("AUGMENT_ENTITY", True)      # Enabled by default
augment_document = env_bool("AUGMENT_DOCUMENT", False)  # Disabled by default
dry_run = env_bool("DRY_RUN", False)                   # Write by default

# Conservative similarity thresholds
entity_sim_cutoff = float(os.getenv("ENTITY_SIM_CUTOFF", "0.25"))
doc_sim_cutoff = float(os.getenv("DOC_SIM_CUTOFF", "0.8"))
```

## Error Handling

### Precondition Failures

```python
try:
    adder.run_preconditions()
except Exception as e:
    print(f"Precondition validation failed: {e}")
    return 1
```

### LLM Processing Errors

- **API Failures**: Falls back to stub labeling
- **Rate Limiting**: Built-in retry logic with exponential backoff
- **Invalid Responses**: Validation and fallback mechanisms

### Database Errors

- **Connection Issues**: Graceful failure with error reporting
- **Constraint Violations**: MERGE operations handle duplicates safely
- **Transaction Failures**: Isolated per-batch processing

## Performance Considerations

### Similarity Computation

- **Entity Processing**: O(n²) where n = number of entities
- **Document Processing**: O(m²) where m = number of documents
- **Optimization**: Use cutoff thresholds to reduce candidate pairs

### LLM Processing

- **Rate Limiting**: Batch processing to avoid API limits
- **Cost Management**: Use appropriate model for task complexity
- **Caching**: Results are persisted to avoid reprocessing

### Memory Usage

```python
# Typical memory usage per batch
entity_batch_size = min(entity_limit, 1000)  # Configurable limit
memory_per_entity = evidence_size + metadata_size
# For 100 entities: 100 * (6 chunks * 1KB + 1KB metadata) ≈ 700KB
```

## Integration Points

### With Other Modules

- **initial_graph_build.py**: Provides the base graph structure with entities and documents
- **compute_doc_embeddings.py**: Provides document embeddings required for document similarity
- **add_relationships_to_graph.py**: Core utilities for relationship discovery and merging
- **launch_data_processing.py**: Includes this module in the complete pipeline

### With APH-IF System

- **Enhanced Queries**: Enables traversal of semantic relationships in backend queries
- **Improved Retrieval**: Better context through relationship-aware search
- **Knowledge Discovery**: Reveals hidden connections in regulatory documents

## Best Practices

### Development Workflow

1. **Start with Dry Run**: Always preview relationships before writing
2. **Use Test Environment**: Test with `FORCE_TEST_DB=true` first
3. **Conservative Thresholds**: Start with higher similarity cutoffs
4. **Monitor API Usage**: Track OpenAI API calls and costs
5. **Validate Results**: Review generated relationships for quality

### Production Deployment

1. **Backup Database**: Always backup before relationship augmentation
2. **Batch Processing**: Process relationships in manageable batches
3. **Monitor Resources**: Watch Neo4j memory and API usage
4. **Quality Control**: Review high-impact relationships manually
5. **Schedule Appropriately**: Run during low-traffic periods

### Optimization Strategies

```powershell
# For speed (fewer, higher-quality relationships)
$env:ENTITY_SIM_CUTOFF = "0.4"
$env:ENTITY_LIMIT = "100"
$env:DOC_SIM_CUTOFF = "0.85"

# For coverage (more relationships, lower quality)
$env:ENTITY_SIM_CUTOFF = "0.15"
$env:ENTITY_LIMIT = "1000"
$env:DOC_SIM_CUTOFF = "0.7"

# For testing (fast, minimal processing)
$env:ENTITY_LIMIT = "10"
$env:DRY_RUN = "true"
```

## Examples

### Basic Entity Relationship Discovery

```powershell
# Setup environment
python ../set_environment.py --mode development

# Run entity relationship discovery
cd data_processing
$env:AUGMENT_ENTITY = "true"
$env:AUGMENT_DOCUMENT = "false"
uv run python -m processing.run_relationship_augmentation
```

### Document Relationship Discovery

```powershell
# Ensure document embeddings exist first
uv run python -m processing.compute_doc_embeddings

# Run document relationship discovery
$env:AUGMENT_ENTITY = "false"
$env:AUGMENT_DOCUMENT = "true"
$env:DOC_SIM_CUTOFF = "0.8"
uv run python -m processing.run_relationship_augmentation
```

### Complete Relationship Augmentation

```powershell
# Run both entity and document relationship discovery
$env:AUGMENT_ENTITY = "true"
$env:AUGMENT_DOCUMENT = "true"
$env:ENTITY_SIM_CUTOFF = "0.25"
$env:DOC_SIM_CUTOFF = "0.8"
uv run python -m processing.run_relationship_augmentation
```

### Dry Run Preview

```powershell
# Preview relationships without writing to database
$env:DRY_RUN = "true"
$env:ENTITY_SIM_CUTOFF = "0.2"
$env:ENTITY_LIMIT = "50"
uv run python -m processing.run_relationship_augmentation
```

## Output Example

```
Running relationship augmentation...
✅ Preconditions passed

Processing ENTITY relationships...
Found 45 entity pairs above similarity threshold 0.25
Collected evidence for 45 entity pairs
LLM labeling completed for 45 relationships
Merged 42 ENTITY relations

Processing DOCUMENT relationships...
Found 12 document pairs above similarity threshold 0.8
Collected evidence for 12 document pairs  
LLM labeling completed for 12 relationships
Merged 12 DOCUMENT relations

Relationship augmentation completed successfully!
```

This module significantly enhances the APH-IF knowledge graph by discovering and adding semantic relationships, enabling more sophisticated queries and improved retrieval performance through intelligent relationship discovery and LLM-powered classification.
