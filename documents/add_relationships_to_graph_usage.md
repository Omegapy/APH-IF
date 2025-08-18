# Add Relationships to Graph Module Documentation

## Overview

The `add_relationships_to_graph.py` module provides programmatic utilities to discover and add higher-level relationships between graph nodes (entities and documents) in the APH-IF hybrid knowledge store. It augments the base graph produced by `initial_graph_build.py` with intelligent relationship discovery using similarity algorithms and LLM-powered labeling.

## Purpose

This module enhances the knowledge graph by:

- **Entity Relationship Discovery**: Finding relationships between entities using Jaccard similarity
- **Document Relationship Discovery**: Finding relationships between documents using cosine similarity
- **LLM-Powered Labeling**: Using GPT-5 to classify and label discovered relationships
- **Graph Enhancement**: Adding structured relationships with evidence and confidence scores
- **Semantic Enrichment**: Creating a more connected and queryable knowledge graph

## Architecture

### Core Components

1. **GraphRelationshipAdder**: Main orchestrator for relationship augmentation
2. **Similarity Algorithms**: Jaccard (entities) and cosine (documents) similarity computation
3. **Evidence Collection**: Gathering supporting text chunks for relationship validation
4. **LLM Integration**: OpenAI GPT-5 for relationship classification and labeling
5. **Graph Merging**: Structured insertion of relationships into Neo4j with property validation

### Processing Flow

```
Base Graph → Similarity Computation → Evidence Collection → LLM Labeling → Relationship Merging
     ↓              ↓                      ↓                   ↓               ↓
Entities/Docs → Candidate Pairs → Supporting Chunks → Classified Relations → Enhanced Graph
```

## Configuration

### Environment Variables

#### Required Variables
- `NEO4J_URI`: Neo4j database connection URI
- `NEO4J_USERNAME`: Neo4j database username
- `NEO4J_PASSWORD`: Neo4j database password
- `OPENAI_API_KEY`: OpenAI API key for LLM labeling

#### Optional Variables
- `OPENAI_MODEL`: OpenAI model for relationship labeling (default: `gpt-5-nano`)
- `VERBOSE`: Enable verbose logging (default: `false`)
- `ENTITY_SIM_CUTOFF`: Entity similarity threshold (default: `0.2`)
- `DOC_SIM_CUTOFF`: Document similarity threshold (default: `0.75`)

### Prerequisites

1. **Initial Graph Build**: Must have completed with entities and documents
2. **Document Embeddings**: Document-level embeddings must exist (from `compute_doc_embeddings.py`)
3. **Neo4j Constraints**: Proper node constraints and indexes
4. **APOC Procedures**: Optional but recommended for enhanced functionality

## Usage

### Basic Usage

```python
from processing.add_relationships_to_graph import (
    Neo4jConfig, GraphRelationshipAdder,
    stream_entity_similarity, stream_doc_similarity,
    fetch_entity_evidence, label_with_llm, merge_entity_relations
)

# Initialize configuration and adder
cfg = Neo4jConfig.from_env()
adder = GraphRelationshipAdder(cfg)
adder.run_preconditions()

# Entity relationship discovery
entity_pairs = stream_entity_similarity(
    adder.run_cypher, 
    similarity_cutoff=0.25, 
    limit=1000
)

# Build inputs with evidence
inputs = []
for row in entity_pairs[:50]:
    e1, e2, sim = int(row["e1_id"]), int(row["e2_id"]), float(row["similarity"])
    evidence = fetch_entity_evidence(adder.run_cypher, e1, e2, top_n=6)
    inputs.append(RelationInput(
        domain="ENTITY", 
        subject_id=e1, 
        object_id=e2, 
        evidence_chunks=evidence, 
        seed_score=sim
    ))

# LLM labeling and merging
outputs = label_with_llm(inputs)
merge_entity_relations(adder.run_cypher, outputs)

adder.close()
```

### Command Line Usage

```powershell
# Run via wrapper module
cd data_processing
uv run python -m processing.run_relationship_augmentation

# With custom parameters
$env:ENTITY_SIM_CUTOFF = "0.3"
$env:DOC_SIM_CUTOFF = "0.8"
$env:VERBOSE = "true"
uv run python -m processing.run_relationship_augmentation
```

## API Reference

### Core Classes

#### Neo4jConfig

```python
@dataclass
class Neo4jConfig:
    uri: str
    username: str
    password: str
    
    @classmethod
    def from_env(cls) -> 'Neo4jConfig':
        """Create configuration from environment variables."""
```

#### GraphRelationshipAdder

```python
class GraphRelationshipAdder:
    """Utility for relationship augmentation with graph precondition checks."""
    
    def __init__(self, config: Neo4jConfig):
        """Initialize with Neo4j configuration."""
    
    def run_preconditions(self) -> bool:
        """Validate cluster capabilities and schema constraints."""
    
    def run_cypher(self, query: str, params: dict = None) -> List[Dict]:
        """Execute Cypher query and return results."""
    
    def close(self) -> None:
        """Close database connection."""
```

### Data Classes

#### RelationInput

```python
@dataclass
class RelationInput:
    domain: str                 # "ENTITY" | "DOCUMENT"
    subject_id: int             # Neo4j internal id()
    object_id: int              # Neo4j internal id()
    evidence_chunks: List[Dict] # Supporting text chunks
    seed_score: float = 0.0     # Initial similarity score
```

#### RelationOutput

```python
@dataclass
class RelationOutput:
    domain: str                 # "ENTITY" | "DOCUMENT"
    type_name: str             # "RELATED_TO" | "RELATIONSHIP"
    subject_id: int            # Neo4j internal id()
    object_id: int             # Neo4j internal id()
    props: Dict[str, Any]      # Relationship properties
```

### Similarity Functions

#### stream_entity_similarity()

```python
def stream_entity_similarity(
    run_cypher_fn,
    similarity_cutoff: float = 0.2,
    limit: int = 50000
) -> List[Dict[str, Any]]:
    """Compute entity-to-entity Jaccard similarity over shared chunks.
    
    Args:
        run_cypher_fn: Function to execute Cypher queries
        similarity_cutoff: Minimum similarity threshold
        limit: Maximum number of pairs to return
        
    Returns:
        List of {e1_id, e2_id, similarity} dictionaries
    """
```

**Algorithm:**
- Computes Jaccard similarity: `jaccard = co / (deg1 + deg2 - co)`
- Where `co` = shared chunks, `deg1/deg2` = entity degrees
- Filters by similarity threshold and limits results

#### stream_doc_similarity()

```python
def stream_doc_similarity(
    run_cypher_fn,
    similarity_cutoff: float = 0.75,
    limit: int = 50000
) -> List[Dict[str, Any]]:
    """Compute document-to-document cosine similarity.
    
    Args:
        run_cypher_fn: Function to execute Cypher queries
        similarity_cutoff: Minimum similarity threshold
        limit: Maximum number of pairs to return
        
    Returns:
        List of {d1_id, d2_id, similarity} dictionaries
    """
```

**Requirements:**
- Document embeddings must exist (from `compute_doc_embeddings.py`)
- Uses cosine similarity on document embedding vectors

### Evidence Collection

#### fetch_entity_evidence()

```python
def fetch_entity_evidence(
    run_cypher_fn,
    e1_id: int,
    e2_id: int,
    top_n: int = 3
) -> List[Dict]:
    """Fetch text chunks that mention both entities.
    
    Args:
        run_cypher_fn: Function to execute Cypher queries
        e1_id: First entity internal ID
        e2_id: Second entity internal ID
        top_n: Maximum chunks per entity
        
    Returns:
        List of {chunk_id, text} dictionaries
    """
```

#### fetch_document_evidence()

```python
def fetch_document_evidence(
    run_cypher_fn,
    d1_id: int,
    d2_id: int,
    per_doc: int = 3
) -> List[Dict]:
    """Fetch representative chunks from both documents.
    
    Args:
        run_cypher_fn: Function to execute Cypher queries
        d1_id: First document internal ID
        d2_id: Second document internal ID
        per_doc: Maximum chunks per document
        
    Returns:
        List of {chunk_id, text} dictionaries
    """
```

### LLM Integration

#### label_with_llm()

```python
def label_with_llm(
    batch: List[RelationInput],
    model: str = None
) -> List[RelationOutput]:
    """Label relationships using OpenAI LLM.
    
    Args:
        batch: List of RelationInput objects to process
        model: OpenAI model to use (defaults to gpt-5-nano)
        
    Returns:
        List of RelationOutput objects with LLM-generated labels
    """
```

**Features:**
- Uses GPT-5 for relationship classification
- Provides structured prompts with evidence chunks
- Returns confidence scores and justifications
- Falls back to stub labeling on API failures

### Graph Merging

#### merge_entity_relations()

```python
def merge_entity_relations(
    run_cypher_fn,
    rels: List[RelationOutput]
) -> int:
    """MERGE entity relationships into the graph.
    
    Args:
        run_cypher_fn: Function to execute Cypher queries
        rels: List of RelationOutput objects to merge
        
    Returns:
        Number of relationships processed
    """
```

#### merge_document_relations()

```python
def merge_document_relations(
    run_cypher_fn,
    rels: List[RelationOutput]
) -> int:
    """MERGE document relationships into the graph.
    
    Args:
        run_cypher_fn: Function to execute Cypher queries
        rels: List of RelationOutput objects to merge
        
    Returns:
        Number of relationships processed
    """
```

## Relationship Schema

### Entity Relationships

```cypher
(:Entity)-[:RELATED_TO {
    rel_type: string,           # Controlled vocabulary label
    score: float,               # Confidence score (0.0-1.0)
    source: string,             # "llm" | "heuristic" | "human"
    evidence_chunks: [string],  # Supporting chunk IDs
    justification: string,      # LLM explanation
    created_at: datetime        # Creation timestamp
}]->(:Entity)
```

### Document Relationships

```cypher
(:Document)-[:RELATIONSHIP {
    rel_type: string,           # Controlled vocabulary label
    score: float,               # Confidence score (0.0-1.0)
    source: string,             # "llm" | "heuristic" | "human"
    evidence_chunks: [string],  # Supporting chunk IDs
    justification: string,      # LLM explanation
    created_at: datetime        # Creation timestamp
}]->(:Document)

(:Document)-[:SIMILAR_TO {
    score: float,               # Similarity score (0.0-1.0)
    method: string,             # "knn" | "nodesim" | "manual"
    created_at: string          # Creation timestamp
}]->(:Document)
```

## Controlled Vocabularies

### Entity Relationship Types

- `RELATED_TO`: General semantic relationship
- `PART_OF`: Hierarchical containment
- `REGULATES`: Regulatory relationship
- `DEFINES`: Definition relationship
- `REFERENCES`: Citation or reference
- `CONFLICTS_WITH`: Contradictory relationship

### Document Relationship Types

- `SIMILAR_TO`: Content similarity
- `SUPERSEDES`: Version replacement
- `AMENDS`: Modification relationship
- `REFERENCES`: Citation relationship
- `IMPLEMENTS`: Implementation relationship

## Performance Considerations

### Similarity Computation

- **Entity Similarity**: O(n²) where n = number of entities
- **Document Similarity**: O(m²) where m = number of documents
- **Optimization**: Use similarity cutoffs to reduce candidate pairs

### LLM Processing

- **Rate Limiting**: Batch requests to avoid API limits
- **Cost Management**: Use smaller models for initial filtering
- **Fallback Strategy**: Stub labeling when API fails

### Memory Usage

```python
# Typical memory usage per batch
memory_per_batch = batch_size * (evidence_size + embedding_size)
# For 50 pairs: 50 * (6 chunks * 1KB + 1536 * 4 bytes) ≈ 600KB
```

## Error Handling

### Common Errors

1. **Missing Prerequisites**
   ```
   Error: Document embeddings not found
   Solution: Run compute_doc_embeddings.py first
   ```

2. **API Failures**
   ```
   Error: OpenAI API rate limit exceeded
   Solution: Implement rate limiting and retry logic
   ```

3. **Similarity Computation Errors**
   ```
   Error: Division by zero in similarity calculation
   Solution: Module handles automatically with defensive checks
   ```

### Recovery Strategies

- **Idempotent Operations**: Can be re-run safely using MERGE operations
- **Partial Processing**: Continues with remaining pairs if some fail
- **Fallback Labeling**: Uses stub labeling when LLM fails

## Integration Points

### With Other Modules

- **initial_graph_build.py**: Provides the base graph structure
- **compute_doc_embeddings.py**: Provides document embeddings for similarity
- **run_relationship_augmentation.py**: Wrapper with environment-specific defaults
- **launch_data_processing.py**: Includes in complete pipeline

### With APH-IF System

- **Enhanced Queries**: Enables traversal of semantic relationships
- **Improved Retrieval**: Better context through relationship awareness
- **Knowledge Discovery**: Reveals hidden connections in regulatory documents

## Best Practices

### Development

1. **Start with Small Batches**: Test with limited entity/document pairs
2. **Monitor API Usage**: Track OpenAI API calls and costs
3. **Validate Results**: Review generated relationships for quality
4. **Use Test Environment**: Always test with `FORCE_TEST_DB=true` first

### Production

1. **Batch Processing**: Process relationships in manageable batches
2. **Rate Limiting**: Implement delays between API calls
3. **Quality Control**: Review and validate high-impact relationships
4. **Monitoring**: Track relationship creation and quality metrics

### Optimization

```powershell
# For faster processing (higher thresholds)
$env:ENTITY_SIM_CUTOFF = "0.4"
$env:DOC_SIM_CUTOFF = "0.85"

# For better coverage (lower thresholds)
$env:ENTITY_SIM_CUTOFF = "0.15"
$env:DOC_SIM_CUTOFF = "0.7"

# For debugging
$env:VERBOSE = "true"
```

## Examples

### Entity Relationship Discovery

```python
# Find similar entities
entity_pairs = stream_entity_similarity(
    adder.run_cypher,
    similarity_cutoff=0.25,
    limit=100
)

# Process first 10 pairs
for row in entity_pairs[:10]:
    e1_id = int(row["e1_id"])
    e2_id = int(row["e2_id"])
    similarity = float(row["similarity"])
    
    # Get evidence
    evidence = fetch_entity_evidence(adder.run_cypher, e1_id, e2_id)
    
    # Create input for LLM
    relation_input = RelationInput(
        domain="ENTITY",
        subject_id=e1_id,
        object_id=e2_id,
        evidence_chunks=evidence,
        seed_score=similarity
    )
```

### Document Relationship Discovery

```python
# Find similar documents
doc_pairs = stream_doc_similarity(
    adder.run_cypher,
    similarity_cutoff=0.8,
    limit=50
)

# Process and label
inputs = []
for row in doc_pairs:
    d1_id = int(row["d1_id"])
    d2_id = int(row["d2_id"])
    similarity = float(row["similarity"])
    
    evidence = fetch_document_evidence(adder.run_cypher, d1_id, d2_id)
    inputs.append(RelationInput(
        domain="DOCUMENT",
        subject_id=d1_id,
        object_id=d2_id,
        evidence_chunks=evidence,
        seed_score=similarity
    ))

# LLM labeling and merging
outputs = label_with_llm(inputs)
merge_document_relations(adder.run_cypher, outputs)
```

This module significantly enhances the APH-IF knowledge graph by discovering and adding semantic relationships between entities and documents, enabling more sophisticated queries and improved retrieval performance.
