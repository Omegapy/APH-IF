# Compute Document Embeddings Module Documentation

## Overview

The `compute_doc_embeddings.py` module is responsible for computing and persisting document-level embeddings by averaging chunk embeddings for each `:Document` node in the APH-IF hybrid knowledge store. This enables document-to-document similarity comparisons and downstream retrieval at the document granularity level.

## Purpose

This module serves as a post-processing step after the initial graph build to:

- **Aggregate chunk embeddings** into document-level representations
- **Enable document similarity** comparisons for retrieval
- **Support document-level queries** in addition to chunk-level searches
- **Provide coarse-grained retrieval** for high-level document matching

## Architecture

### Processing Flow

```
Existing Graph → Chunk Embeddings → Document Aggregation → Document Embeddings
     ↓                ↓                      ↓                     ↓
Document Nodes → HAS_CHUNK → Chunk Nodes → Mean Vector → doc_embedding Property
```

### Core Components

1. **Vector Aggregation**: Element-wise mean computation of chunk embeddings
2. **Database Integration**: Direct Neo4j operations for reading and writing embeddings
3. **Dimension Handling**: Defensive handling of mixed-dimension vectors
4. **Batch Processing**: Efficient streaming and updating of document embeddings

## Configuration

### Environment Variables

#### Required Variables
- `NEO4J_URI`: Neo4j database connection URI (default: `bolt://aph_if_neo4j:7687`)
- `NEO4J_USERNAME` or `NEO4J_USER`: Neo4j database username (default: `neo4j`)
- `NEO4J_PASSWORD`: Neo4j database password (default: `password`)

#### Optional Variables
- No additional configuration variables - the module uses the existing graph structure

### Prerequisites

1. **Initial Graph Build**: Must have completed successfully with chunk embeddings
2. **Chunk Embeddings**: All `:Chunk` nodes must have `embedding` properties
3. **Graph Relationships**: `(:Document)-[:HAS_CHUNK]->(:Chunk)` relationships must exist

## Usage

### Command Line Usage

```powershell
# Run from data_processing directory
cd data_processing

# Basic execution
uv run python -m processing.compute_doc_embeddings

# With environment variables
$env:NEO4J_URI = "bolt://localhost:7687"
$env:NEO4J_USERNAME = "neo4j"
$env:NEO4J_PASSWORD = "your_password"
uv run python -m processing.compute_doc_embeddings
```

### Programmatic Usage

```python
from processing.compute_doc_embeddings import main

# Execute document embedding computation
result = main()
if result == 0:
    print("Document embeddings computed successfully")
else:
    print("Document embedding computation failed")
```

### Integration with Pipeline

```powershell
# Complete pipeline execution
cd data_processing

# Step 1: Build initial graph with chunk embeddings
uv run python -m processing.run_initial_graph_build

# Step 2: Compute document embeddings
uv run python -m processing.compute_doc_embeddings

# Step 3: Add relationships (optional)
uv run python -m processing.run_relationship_augmentation
```

## API Reference

### Core Functions

#### main() -> int

```python
def main() -> int:
    """Aggregate chunk embeddings into :Document.doc_embedding vectors.
    
    Returns:
        int: 0 on success; non-zero on failure conditions.
    """
```

**Process:**
1. Connects to Neo4j database using environment variables
2. Queries all chunk embeddings grouped by document
3. Computes mean vectors for each document
4. Updates document nodes with computed embeddings
5. Reports the number of documents processed

**Error Handling:**
- Database connection failures
- Missing chunk embeddings
- Dimension mismatches in vectors
- Neo4j query errors

#### _mean_vector(vectors: Iterable[List[float]]) -> List[float]

```python
def _mean_vector(vectors: Iterable[List[float]]) -> List[float]:
    """Compute the element-wise mean for a collection of same-dimension vectors.
    
    Args:
        vectors: Iterable of numeric vectors (list of floats) with equal length
                 where possible. Mixed lengths are handled by aligning to the
                 shortest dimension.
    
    Returns:
        List[float]: Mean vector. Returns an empty list if no valid vectors.
    """
```

**Features:**
- **Element-wise averaging** of vector components
- **Defensive dimension handling** for mixed-length vectors
- **Empty vector handling** returns empty list for no input
- **Type safety** with explicit float conversion

## Database Schema Impact

### Before Processing

```cypher
(:Document {
    doc_id: string,
    title: string,
    source: string,
    created_at: integer
})
```

### After Processing

```cypher
(:Document {
    doc_id: string,
    title: string,
    source: string,
    created_at: integer,
    doc_embedding: list  # NEW: 1536-dimension vector
})
```

### Query Pattern

The module uses this Cypher query to aggregate embeddings:

```cypher
MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
WHERE c.embedding IS NOT NULL
RETURN id(d) AS d_id, c.embedding AS emb
```

And updates documents with:

```cypher
MATCH (d:Document) 
WHERE id(d) = $d_id 
SET d.doc_embedding = $emb
```

## Vector Mathematics

### Mean Vector Computation

For a document with chunks containing embeddings `[e1, e2, ..., en]`:

```
doc_embedding[i] = (e1[i] + e2[i] + ... + en[i]) / n
```

Where:
- `n` = number of chunks with embeddings
- `i` = vector dimension index (0 to 1535 for OpenAI embeddings)

### Dimension Handling

The module handles dimension mismatches defensively:

```python
# If vectors have different lengths, align to shortest
if len(vec) != len(sums):
    m = min(len(vec), len(sums))
    for i in range(m):
        sums[i] += float(vec[i])
```

This ensures robustness against:
- Corrupted embeddings with wrong dimensions
- Mixed embedding models with different output sizes
- Partial or incomplete embedding data

## Performance Characteristics

### Memory Usage

- **Streaming Processing**: Processes one document at a time
- **Minimal Memory Overhead**: Only stores vectors for current document
- **Efficient Aggregation**: Uses defaultdict for grouping by document ID

### Processing Speed

- **Database I/O Bound**: Performance limited by Neo4j query speed
- **Linear Complexity**: O(n) where n = number of chunks
- **Batch Updates**: Single query per document for efficient updates

### Scalability

```python
# Memory usage per document
memory_per_doc = num_chunks * embedding_dimension * 8_bytes
# For typical document: 10 chunks * 1536 dims * 8 bytes = ~120KB
```

## Error Handling

### Common Errors

1. **Database Connection Issues**
   ```
   Error: Could not connect to Neo4j
   Solution: Verify NEO4J_URI, username, and password
   ```

2. **Missing Chunk Embeddings**
   ```
   Error: No chunk embeddings found
   Solution: Run initial_graph_build.py first
   ```

3. **Dimension Mismatches**
   ```
   Warning: Vector dimension mismatch detected
   Solution: Module handles automatically by aligning to shortest
   ```

4. **Empty Documents**
   ```
   Info: Document has no chunks with embeddings
   Solution: Document will be skipped (normal behavior)
   ```

### Recovery Strategies

- **Partial Processing**: Can be re-run safely (idempotent operation)
- **Incremental Updates**: Only processes documents without existing embeddings
- **Error Isolation**: Continues processing other documents if one fails

## Verification and Testing

### Verification Queries

After running the module, verify results with these Cypher queries:

```cypher
-- Count documents with embeddings
MATCH (d:Document) 
WHERE d.doc_embedding IS NOT NULL 
RETURN count(d) AS docs_with_embeddings

-- Check embedding dimensions
MATCH (d:Document) 
WHERE d.doc_embedding IS NOT NULL 
RETURN size(d.doc_embedding) AS dimension 
LIMIT 5

-- Compare document and chunk embedding dimensions
MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
WHERE d.doc_embedding IS NOT NULL AND c.embedding IS NOT NULL
RETURN size(d.doc_embedding) AS doc_dim, 
       size(c.embedding) AS chunk_dim 
LIMIT 5

-- Find documents without embeddings
MATCH (d:Document) 
WHERE d.doc_embedding IS NULL 
RETURN d.doc_id, d.title
```

### Test Scenarios

```powershell
# Test with small dataset
$env:FORCE_TEST_DB = "true"
$env:MAX_DOCS = "2"
uv run python -m processing.run_initial_graph_build
uv run python -m processing.compute_doc_embeddings

# Verify results
# (Run verification queries in Neo4j Browser)
```

## Integration Points

### With Other Modules

- **initial_graph_build.py**: Provides the chunk embeddings that this module aggregates
- **add_relationships_to_graph.py**: Can use document embeddings for document similarity
- **launch_data_processing.py**: Includes this module in the complete pipeline
- **Backend queries**: Document embeddings enable document-level retrieval

### With APH-IF System

- **Document Similarity**: Enables finding similar documents using cosine similarity
- **Coarse-grained Retrieval**: Allows retrieval at document level before drilling down to chunks
- **Hybrid Queries**: Supports both document-level and chunk-level semantic search

## Best Practices

### Development

1. **Run After Initial Build**: Always run after `initial_graph_build.py` completes
2. **Verify Prerequisites**: Check that chunk embeddings exist before running
3. **Test with Small Data**: Use test environment for initial validation
4. **Monitor Progress**: Check console output for processing statistics

### Production

1. **Backup Database**: Backup before running in production
2. **Monitor Resources**: Watch Neo4j memory usage during processing
3. **Validate Results**: Run verification queries after completion
4. **Schedule Appropriately**: Run during low-traffic periods

### Troubleshooting

1. **Check Prerequisites**: Ensure initial graph build completed successfully
2. **Verify Connections**: Test Neo4j connectivity and credentials
3. **Review Logs**: Check console output for error messages
4. **Validate Data**: Use verification queries to check results

## Examples

### Basic Usage

```powershell
# Setup environment
python ../set_environment.py --mode development

# Run initial graph build first
cd data_processing
uv run python -m processing.run_initial_graph_build

# Compute document embeddings
uv run python -m processing.compute_doc_embeddings
```

### Test Environment

```powershell
# Setup test environment
python ../set_environment.py --mode development --force-test-db true

# Quick test with limited data
$env:FORCE_TEST_DB = "true"
$env:MAX_DOCS = "2"
uv run python -m processing.run_initial_graph_build
uv run python -m processing.compute_doc_embeddings
```

### Verification

```powershell
# After running, verify in Neo4j Browser:
# MATCH (d:Document) WHERE d.doc_embedding IS NOT NULL RETURN count(d)
# Expected: Number should match processed documents count
```

## Output Example

```
Computed doc embeddings for 15 documents
```

This indicates successful processing of 15 documents with their chunk embeddings aggregated into document-level embeddings.

The module is essential for enabling document-level similarity comparisons and coarse-grained retrieval in the APH-IF hybrid knowledge store, complementing the chunk-level embeddings created during initial graph build.
