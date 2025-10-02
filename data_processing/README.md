# APH-IF Data Processing Module (alpha v2.0)

Docling-based PDF ingestion and Neo4j knowledge graph builder for the APH-IF project.
This PDF injection was optimized for the Legal domain, more specifically for CFR-30 - MSHA Regulations.

## Overview

This module processes PDF documents through three distinct phases:
- **Phase 1**: PDF parsing, chunking, and embeddings
- **Phase 2**: Entity extraction from chunks
- **Phase 3**: Relationship augmentation between entities

**Key Features:**
- PDF parsing with Docling (preserves page numbers and structure)
- Page-aware chunking (~3000 characters per chunk)
- OpenAI text-embedding-3-large embeddings (3072 dimensions)
- Hybrid rule-based entity extraction (spaCy + regex)
- LLM-powered relationship detection (GPT-5-mini)
- Neo4j vector index creation (`chunk_embedding_index`)
- Idempotent MERGE-based database operations
- Backend-compatible data model (`Document`, `Chunk`, `Entity` nodes)

## Architecture

**Technology Stack:**
- **PDF Parsing**: Docling 2.0+ (handles complex PDFs with tables, headings)
- **Embeddings**: OpenAI text-embedding-3-large (3072d vectors)
- **Entity Extraction**: spaCy 3.7+ with EntityRuler and PhraseMatcher
- **Relationship Detection**: OpenAI GPT-5-mini
- **Database**: Neo4j (local or AuraDB)
- **Language**: Python 3.12+
- **Package Manager**: UV

**Data Model:**
```
(Document {document_id, title, filename, created_at})
  -[:HAS_CHUNK]->
(Chunk {chunk_id, document_id, page, text, embedding, section, tokens})
  -[:HAS_ENTITY]->
(Entity {name, type, canonical_name, occurrences})
  -[:RELATED_TO {type, confidence, source, reason}]->
(Entity)
```

## Installation

### Prerequisites
- Python 3.12+
- UV package manager
- Neo4j database (running locally or AuraDB)
- OpenAI API key

### Setup

1. **Navigate to data_processing directory:**
```bash
cd data_processing
```

2. **Install dependencies:**
```bash
uv sync
```

3. **Download spaCy model (for Phase 2):**
```bash
uv run python -m spacy download en_core_web_sm
```

4. **Configure environment:**
   - Copy and edit `.env` file with your credentials:
   - Neo4j connection (URI, username, password, database)
   - OpenAI API key
   - Processing parameters (chunk size, batch sizes, etc.)

## Usage

### Phase-Based CLI

The unified CLI provides clear phase separation:

```bash
# Run ALL phases (recommended for first-time setup)
uv run python run_ingest.py --all

# Or run phases individually:
uv run python run_ingest.py --clear            # Clear database
uv run python run_ingest.py --init-chunks      # Phase 1: PDF → chunks
uv run python run_ingest.py --extract-entities # Phase 2: Extract entities
uv run python run_ingest.py --augment          # Phase 3: Augment relationships
```

### Command-Line Options

#### Phase Selection

| Option | Description |
|--------|-------------|
| `--clear` | Clear database (requires confirmation) |
| `--init-chunks` | Phase 1: Ingest PDFs → chunks + embeddings |
| `--extract-entities` | Phase 2: Extract entities from existing chunks |
| `--augment` | Phase 3: Augment relationships from existing entities |
| `--all` | Run all phases: clear → init-chunks → extract-entities → augment |

#### Phase 1 Options

| Option | Description | Example |
|--------|-------------|---------|
| `--max-docs N` | Process only N documents | `--max-docs 10` |
| `--file NAME` | Process specific PDF file | `--file "report.pdf"` |
| `--chunk-size-chars N` | Override chunk size | `--chunk-size-chars 2000` |

#### General Options

| Option | Description |
|--------|-------------|
| `--dry-run` | Test run without database writes |
| `--yes`, `-y` | Skip confirmation prompts |

### Workflow Examples

**1. First-time setup (all phases):**
```bash
# Clear database and run all phases on 1 document (test)
uv run python run_ingest.py --all --max-docs 1

# Process all PDFs through all phases
uv run python run_ingest.py --all
```

**2. Phase 1 only (PDF ingestion):**
```bash
# Process all PDFs to chunks
uv run python run_ingest.py --init-chunks

# Process specific file
uv run python run_ingest.py --init-chunks --file "CFR-2024-title30-vol1.pdf"

# Process limited number
uv run python run_ingest.py --init-chunks --max-docs 5
```

**3. Phase 2 only (add entities to existing chunks):**
```bash
# Extract entities from all existing chunks
uv run python run_ingest.py --extract-entities
```

**4. Phase 3 only (add relationships to existing entities):**
```bash
# Augment relationships (respects MAX_AUGMENTATION_CHUNKS in .env)
uv run python run_ingest.py --augment
```

**5. Combined phases:**
```bash
# Phase 1 + 2: Ingest PDFs and extract entities
uv run python run_ingest.py --init-chunks --extract-entities

# Phase 2 + 3: Extract entities and augment relationships
uv run python run_ingest.py --extract-entities --augment

# Clear DB, then run Phase 1 and 2
uv run python run_ingest.py --clear --init-chunks --extract-entities
```

**6. Clear database:**
```bash
# Requires confirmation
uv run python run_ingest.py --clear

# Skip confirmation
uv run python run_ingest.py --clear --yes
```

**7. Test before committing:**
```bash
# Dry run to verify everything works
uv run python run_ingest.py --init-chunks --dry-run --max-docs 1
```

## Processing Pipeline

### Phase 1: PDF → Chunks + Embeddings

**What it does:**
1. Parses PDFs with Docling (extracts text with page numbers)
2. Creates Document nodes in Neo4j
3. Chunks text (~3000 characters per chunk)
4. Generates embeddings (OpenAI text-embedding-3-large)
5. Creates Chunk nodes with HAS_CHUNK relationships
6. Creates vector index for similarity search

**Processing time:**
- Small PDF (50 pages): ~30 seconds
- Large PDF (720 pages): ~5-10 minutes

### Phase 2: Entity Extraction

**What it does:**
1. Fetches all chunks from Neo4j
2. Extracts legal entities using hybrid approach:
   - spaCy EntityRuler (token-based patterns)
   - spaCy PhraseMatcher (dictionary matching)
   - Refined regex patterns
3. Normalizes and deduplicates entities
4. Creates Entity nodes and HAS_ENTITY relationships

**Entity types extracted:**
Optimized for the Legal domain, more specifically  
- `LEGAL_SECTION` - §75.1714-1, 30 CFR 57.4361(a)
- `CFR_PART` - Part 75, Part 75.1714
- `SUBPART` - Subpart D, Subpart AC
- `APPENDIX` - Appendix A, Appendix B-1
- `CFR_TITLE` - Title 30
- `STANDARD` - ISO 9001, ASTM D4318

**Processing time:**
- 1000 chunks: ~2-5 minutes

### Phase 3: Relationship Augmentation

**What it does:**
1. Fetches entities from Neo4j (limited by MAX_AUGMENTATION_CHUNKS)
2. Uses GPT-5-mini to detect semantic relationships
3. Creates RELATED_TO edges with:
   - `type`: "PART_OF", "REFERENCES", "SUPERSEDES", "RELATES_TO"
   - `confidence`: 0.6-1.0 (threshold at 0.6)
   - `source`: "llm"
   - `reason`: Natural language explanation

**Processing time:**
- 100 chunks: ~3-5 minutes
- **Cost**: ~$0.10-1.00 per 100 chunks (GPT-5-mini API)

⚠️ **Cost Control**: Set `MAX_AUGMENTATION_CHUNKS` in `.env` to limit API costs

## Verification

### Check Neo4j Data

**Phase 1 verification:**
```cypher
// Count documents and chunks
MATCH (d:Document)
RETURN d.title, count{(d)-[:HAS_CHUNK]->()} as chunks

// Verify embeddings exist
MATCH (c:Chunk)
WHERE c.embedding IS NOT NULL
RETURN count(c) as chunks_with_embeddings

// Check vector index
SHOW INDEXES
```

**Phase 2 verification:**
```cypher
// Count entities by type
MATCH (e:Entity)
RETURN e.type, count(e) as count, collect(e.name)[0..5] as sample
ORDER BY count DESC

// Check HAS_ENTITY relationships
MATCH (c:Chunk)-[:HAS_ENTITY]->(e:Entity)
RETURN e.name, e.type, count(c) as chunks
ORDER BY chunks DESC
LIMIT 20
```

**Phase 3 verification:**
```cypher
// Count relationships by type
MATCH ()-[r:RELATED_TO]-()
RETURN r.type as relationship_type, count(r) as count
ORDER BY count DESC

// View sample relationships
MATCH (source:Entity)-[r:RELATED_TO]->(target:Entity)
RETURN source.name, r.type, target.name, r.confidence, r.reason
LIMIT 10

// High confidence relationships
MATCH (source:Entity)-[r:RELATED_TO]->(target:Entity)
WHERE r.confidence >= 0.8
RETURN source.name, r.type, target.name, r.confidence, r.reason
ORDER BY r.confidence DESC
LIMIT 20
```

## Data Migrations

### Page Number Adjustment Script

**Purpose:** One-time migration to adjust `Chunk.page` values to match document pagination conventions.

**Transformation Rules:**
- Pages 1-10 → Roman numerals (i, ii, iii, iv, v, vi, vii, viii, ix, x)
- Pages >10 → Subtract 10, display as strings ("1", "2", "3", ...)

**Why?** CFR documents have preliminary pages (i-x) followed by numbered content. This script adjusts the stored page numbers to match the actual document structure for accurate citations.

**Usage:**

```bash
# Preview changes (safe, no writes)
uv run python page_num_adjust.py --dry-run

# Execute on all documents with confirmation
uv run python page_num_adjust.py

# Execute on specific document without prompt
uv run python page_num_adjust.py --document-id cfr_2024_title30_vol2 --yes

# View help
uv run python page_num_adjust.py --help
```

**CLI Options:**
- `--dry-run`: Preview changes without executing updates
- `--yes`, `-y`: Skip confirmation prompt
- `--document-id <id>`: Limit scope to a single document
- `--force`: Allow execution on production environment

**Safety Features:**
- Environment validation (blocks production without `--force`)
- Preview with counts and sample mappings
- Idempotent execution (won't break on re-runs)
- Comprehensive logging to `monitoring_logs/`
- Post-update verification queries

**Workflow:**

1. **Test on single document first:**
   ```bash
   uv run python page_num_adjust.py --document-id cfr_2024_title30_vol2 --yes
   ```

2. **Verify in Neo4j Browser:**
   ```cypher
   MATCH (c:Chunk {document_id: 'cfr_2024_title30_vol2'})
   RETURN c.page
   ORDER BY c.chunk_id
   LIMIT 20
   ```

3. **If successful, run on all documents:**
   ```bash
   uv run python page_num_adjust.py --yes
   ```

4. **Refresh backend schema cache** (see Backend Integration below)

**Important Notes:**
- This script does NOT modify `chunk_id` or relationships
- Page values change from integers to strings
- Backend already handles page as strings for display (citations like "p.i", "p.v", "p.1")
- Assumes all documents have exactly 10 preliminary roman pages

---

## Backend Integration

After ingestion (or after running migrations), refresh the backend schema cache:

```bash
cd ../backend
uv run python -m app.schema.refresh_schema
```

This updates:
- `schema_cache/kg_schema.json`
- `kg_schema_structural_summary.json`

Then test queries via:
- Backend API: `http://localhost:8000/query`
- Streamlit frontend: `http://localhost:8501`

## Configuration

### Environment Variables (.env)

Key settings in `data_processing/.env`:

```bash
# Neo4j Connection (auto-selected based on APP_ENV)
APP_ENV='development'  # or 'test'
NEO4J_URI_DEV='bolt://172.20.80.1:7687'
NEO4J_USERNAME_DEV='neo4j'
NEO4J_PASSWORD_DEV='your-password'
NEO4J_DATABASE_DEV='neo4j'

# OpenAI API
OPENAI_API_KEY='sk-...'
OPENAI_MODEL_MINI='gpt-5-mini'

# Processing Parameters
CHUNK_SIZE_CHARS=3000
MAX_DOCS=0  # 0 = process all
EMBEDDING_BATCH_SIZE=64
NEO4J_BATCH_SIZE=500

# Phase 2: Entity Extraction
EXTRACT_ENTITIES=false
SPACY_MODEL=en_core_web_sm
ENTITY_TYPES=LEGAL_SECTION,CFR_PART,SUBPART,APPENDIX,CFR_TITLE,STANDARD

# Phase 3: Relationship Augmentation
AUGMENT_RELATIONSHIPS=false
MAX_AUGMENTATION_CHUNKS=0  # 0 = unlimited (use with caution!)

# Logging
VERBOSE=true
MONITORING_ENABLED=true
```

### Cost Control for Phase 3

⚠️ **Important**: Phase 3 uses GPT-5-mini API calls which incur costs.

**Configure in `.env`:**
```bash
# Process only 100 chunks (safer for testing)
MAX_AUGMENTATION_CHUNKS=100

# Process all chunks (expensive, use with caution!)
MAX_AUGMENTATION_CHUNKS=0

# Process 500 chunks (medium-sized job)
MAX_AUGMENTATION_CHUNKS=500
```

## Troubleshooting

### Issue: "No pages extracted from PDF"

**Cause**: Some PDFs don't populate Docling's provenance metadata.

**Solution**: The system falls back to text export (works but loses page granularity).

**Verify**: Check logs for "Extracted X pages using provenance metadata" vs "No page provenance found"

### Issue: Embeddings API rate limit errors

**Solution**: Adjust rate limiting in `.env`:
```bash
OPENAI_REQUESTS_PER_MINUTE=4500  # Lower if hitting limits
EMBEDDING_BATCH_SIZE=32  # Reduce batch size
```

### Issue: Neo4j connection timeout

**Solution**: Verify Neo4j is running:
```bash
# Check Neo4j status
neo4j status

# Or check connection manually
curl http://localhost:7474
```

### Issue: Vector index not created

**Solution**: Create manually:
```cypher
CREATE VECTOR INDEX chunk_embedding_index IF NOT EXISTS
FOR (n:Chunk)
ON n.embedding
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 3072,
    `vector.similarity_function`: 'cosine'
  }
}
```

### Issue: spaCy model not found

**Solution**: Download the model:
```bash
uv run python -m spacy download en_core_web_sm
```

### Issue: Phase 3 relationship augmentation errors

**Common errors:**
- **"Unsupported parameter: max_tokens"** - Fixed in v2.0 (uses max_completion_tokens)
- **"Unsupported value: temperature"** - Fixed in v2.0 (removed temperature parameter)

**Solution**: Ensure you're using the latest version of entities/augment.py

## File Structure

```
data_processing/
├── README.md                 # This file
├── .env.example              # Environment configuration
├── pyproject.toml            # Dependencies
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

## Development

### Running Tests

```bash
# Unit tests for entity patterns
uv run pytest entities/evaluation/test_patterns.py -v

# Integration test with sample PDF
uv run python run_ingest.py --init-chunks --file "sample.pdf" --dry-run --max-docs 1
```

### CLI Help

```bash
# View all available commands
uv run python run_ingest.py --help
```

## Performance Metrics

**Phase 1 (PDF → Chunks):**
- Small PDF (50 pages): ~28 seconds total
- Large PDF (720 pages): ~5-10 minutes total

**Breakdown:**
- Docling conversion: ~20-60% of time
- Embedding generation: ~30-40% of time
- Neo4j writes: ~5-10% of time

**Phase 2 (Entity Extraction):**
- 1000 chunks: ~2-5 minutes
- Adds ~20-30% to Phase 1 processing time

**Phase 3 (Relationship Augmentation):**
- 100 chunks: ~3-5 minutes
- Cost: ~$0.10-1.00 per 100 chunks (GPT-5-mini)

## Support

For issues or questions:
1. Check logs in `monitoring_logs/`
2. Verify `.env` configuration
3. Test with `--dry-run` first
4. Review documentation:
   - Docling: https://github.com/docling-project/docling
   - spaCy: https://spacy.io/usage/rule-based-matching
   - Neo4j: https://neo4j.com/docs/

---

**Quick Reference:**

```bash
# Full pipeline (fresh start)
uv run python run_ingest.py --all --max-docs 1

# Phase 1: Ingest PDFs
uv run python run_ingest.py --init-chunks

# Phase 2: Extract entities
uv run python run_ingest.py --extract-entities

# Phase 3: Augment relationships
uv run python run_ingest.py --augment

# Clear database
uv run python run_ingest.py --clear

# Page number migration (run after ingestion if needed)
uv run python page_num_adjust.py --dry-run  # Preview
uv run python page_num_adjust.py --yes       # Execute
```
