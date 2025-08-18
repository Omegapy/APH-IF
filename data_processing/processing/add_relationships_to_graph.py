# -------------------------------------------------------------------------
# File: add_relationships_to_graph.py
# Project: APH-IF
# Author: Alexander Ricciardi
# Date: 2025-08-12
# File Path: data_processing/processing/add_relationships_to_graph.py
# ------------------------------------------------------------------------

# --- Module Objective ---
#   Provide programmatic utilities to discover and add higher-level
#   relationships between graph nodes (entities/documents), including
#   preconditions, taxonomy, candidate generation, evidence collection,
#   LLM labeling, and MERGE back to Neo4j.
#
# Entity Relationship Types:
# - `RELATED_TO`: General semantic relationship
# - `PART_OF`: Hierarchical containment
# - `REGULATES`: Regulatory relationship
# - `DEFINES`: Definition relationship
# - `REFERENCES`: Citation or reference
# - `CONFLICTS_WITH`: Contradictory relationship
# 
# Document Relationship Types:
# - `SIMILAR_TO`: Content similarity
# - `SUPERSEDES`: Version replacement
# - `AMENDS`: Modification relationship
# - `REFERENCES`: Citation relationship
# - `IMPLEMENTS`: Implementation relationship
# -------------------------------------------------------------------------

# --- Module Contents Overview ---
# - Class: GraphRelationshipAdder
# - Data Classes: RelationInput, RelationOutput, EdgePropSpec, EdgeTypeSpec
# - Functions: stream_entity_similarity, stream_doc_similarity,
#              fetch_entity_evidence, fetch_document_evidence,
#              label_with_llm, label_with_llm_stub,
#              merge_entity_relations, merge_document_relations
# -------------------------------------------------------------------------

# --- Dependencies / Imports ---
# - Standard Library: os, sys, dataclasses, typing, enum
# - Third-Party: neo4j (driver)
# - Local Project Modules: none (used by wrappers in processing/)
# -------------------------------------------------------------------------

# --- Usage / Integration ---
# See the module docstring for an end-to-end usage example demonstrating
# preconditions, candidate generation, LLM labeling, and MERGE.
# -------------------------------------------------------------------------

# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - APH-IF  
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG – Intelligent Fusion (APH-IF)
# -------------------------------------------------------------------------


# =========================================================================
# Imports
# =========================================================================
# Standard library imports
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

from neo4j import GraphDatabase, basic_auth

# Import monitoring components
try:
    from common.monitored_openai import MonitoredOpenAIClient
    from common.api_monitor import LogLevel
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

# =========================================================================
# Global Constants / Variables
# =========================================================================
# (No module-level constants required; taxonomy is defined below)

# ----------------------------
# Utilities / Logging
# ----------------------------

def info(msg: str) -> None:
    print(f"[INFO] {msg}")

def warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)

def error(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)

def debug(msg: str) -> None:
    """Print debug message if verbose mode is enabled."""
    if os.getenv("VERBOSE", "false").lower() == "true":
        print(f"[DEBUG] {msg}")

# ----------------------------
# Configuration
# ----------------------------

# --------------------------------------------------------------------------------- Neo4jConfig
@dataclass
class Neo4jConfig:
    uri: str
    user: str
    password: str

    @staticmethod
    def from_env() -> "Neo4jConfig":
        # Use centralized environment variables managed by set_environment.py
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD")
        return Neo4jConfig(uri=uri, user=user, password=password)

# --------------------------------------------------------------------------------- end Neo4jConfig

# =========================================================================
# Class Definitions
# =========================================================================

# ------------------------------------------------------------------------- class GraphRelationshipAdder
class GraphRelationshipAdder:
    """Utility for relationship augmentation with graph precondition checks.

    Responsibilities:
        - Provide lightweight Neo4j access helpers
        - Validate cluster capabilities (connectivity, APOC)
        - Ensure minimal schema constraints
        - Inspect base topology and vector index presence
    """

    # -------------------
    # --- Constructor ---
    # -------------------

    # --------------------------------------------------------------------------------- __init__()
    def __init__(self, cfg: Neo4jConfig):
        self.cfg = cfg
        self.driver = GraphDatabase.driver(cfg.uri, auth=basic_auth(cfg.user, cfg.password))
    # --------------------------------------------------------------------------------- end __init__()

    # ---------- low-level helpers ----------
    # --------------------------------------------------------------------------------- run_cypher()
    def run_cypher(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return materialized result rows.

        Args:
            query: Cypher text
            params: Parameter map

        Returns:
            List[Dict[str, Any]]: Result rows as dictionaries.
        """
        with self.driver.session() as session:
            return list(session.run(query, params or {}))
    # --------------------------------------------------------------------------------- end run_cypher()

    # --------------------------------------------------------------------------------- single_value()
    def single_value(self, query: str, default: Any = None) -> Any:
        """Execute a Cypher query and extract the first value from the first row."""
        recs = self.run_cypher(query)
        if not recs:
            return default
        first = recs[0]
        return list(first.values())[0] if first else default
    # --------------------------------------------------------------------------------- end single_value()

    # ---------- checks ----------

    # --------------------------------------------------------------------------------- check_connectivity()
    def check_connectivity(self) -> None:
        info("Checking Neo4j connectivity...")
        try:
            v = self.single_value("CALL dbms.components() YIELD name, versions RETURN head(versions)", default=None)
            if v:
                info(f"Connected to Neo4j, version: {v}")
            else:
                warn("Could not detect Neo4j version (proceeding).")
        except Exception as e:
            error(f"Failed to connect to Neo4j: {e}")
            raise
    # --------------------------------------------------------------------------------- end check_connectivity()

    # --------------------------------------------------------------------------------- check_apoc()
    def check_apoc(self) -> None:
        info("Checking APOC availability (optional, recommended)...")
        try:
            v = self.single_value("RETURN apoc.version()", default=None)
            if v:
                info(f"APOC detected, version: {v}")
            else:
                warn("APOC not detected. Some convenience procedures may be unavailable.")
        except Exception as e:
            warn(f"APOC check failed (likely not installed): {e}")
    # --------------------------------------------------------------------------------- end check_apoc()

    # --------------------------------------------------------------------------------- count_nodes()
    def count_nodes(self, label: str) -> int:
        """Return total number of nodes with the provided label."""
        return int(self.single_value(f"MATCH (n:{label}) RETURN count(n)", default=0))
    # --------------------------------------------------------------------------------- end count_nodes()

    # --------------------------------------------------------------------------------- relationship_exists()
    def relationship_exists(self, rel_type: str, lhs_label: Optional[str] = None, rhs_label: Optional[str] = None) -> bool:
        """Check for the existence of at least one relationship of a given type."""
        lhs = f":{lhs_label}" if lhs_label else ""
        rhs = f":{rhs_label}" if rhs_label else ""
        q = f"MATCH ({lhs})-[:{rel_type}]->({rhs}) RETURN count(*) LIMIT 1"
        c = int(self.single_value(q, default=0))
        return c > 0
    # --------------------------------------------------------------------------------- end relationship_exists()

    # --------------------------------------------------------------------------------- ensure_constraints()
    def ensure_constraints(self) -> None:
        info("Ensuring minimal schema constraints (idempotent)...")
        # Entity(name, type) UNIQUE (composite)
        try:
            self.run_cypher(
                """
                CREATE CONSTRAINT entity_name_type_unique IF NOT EXISTS
                FOR (e:Entity) REQUIRE (e.name, e.type) IS UNIQUE
                """
            )
            info("Ensured UNIQUE constraint on :Entity(name,type)")
        except Exception as e:
            warn(f"Could not create UNIQUE constraint for :Entity(name,type): {e}")

        # Detect if :Chunk has 'id' property; if so, create uniqueness
        try:
            has_chunk_id = bool(self.single_value(
                "CALL db.schema.nodeTypeProperties() YIELD nodeLabels, propertyName "
                "WHERE 'Chunk' IN nodeLabels AND propertyName='id' "
                "RETURN 1 LIMIT 1", default=0))
            if has_chunk_id:
                self.run_cypher(
                    """
                    CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
                    FOR (c:Chunk) REQUIRE (c.id) IS UNIQUE
                    """
                )
                info("Ensured UNIQUE constraint on :Chunk(id)")
            else:
                warn("Skipped :Chunk(id) UNIQUE (property 'id' not detected).")
        except Exception as e:
            warn(f"Could not evaluate/create :Chunk(id) UNIQUE: {e}")
    # --------------------------------------------------------------------------------- end ensure_constraints()

    # --------------------------------------------------------------------------------- ensure_document_unique_if_present()
    def ensure_document_unique_if_present(self) -> None:
        # Detect if :Document has 'doc_id' property; if so, create uniqueness
        try:
            has_doc_id = bool(self.single_value(
                "CALL db.schema.nodeTypeProperties() YIELD nodeLabels, propertyName "
                "WHERE 'Document' IN nodeLabels AND propertyName='doc_id' "
                "RETURN 1 LIMIT 1", default=0))
            if has_doc_id:
                self.run_cypher(
                    """
                    CREATE CONSTRAINT document_doc_id_unique IF NOT EXISTS
                    FOR (d:Document) REQUIRE (d.doc_id) IS UNIQUE
                    """
                )
                info("Ensured UNIQUE constraint on :Document(doc_id)")
            else:
                warn("Skipped :Document(doc_id) UNIQUE (property 'doc_id' not detected).")
        except Exception as e:
            warn(f"Could not evaluate/create :Document(id) UNIQUE: {e}")
    # --------------------------------------------------------------------------------- end ensure_document_unique_if_present()

    # --------------------------------------------------------------------------------- check_vector_index()
    def check_vector_index(self) -> None:
        info("Checking for a vector index on :Chunk(embedding)...")
        try:
            rows = self.run_cypher(
                "SHOW INDEXES YIELD name, type, entityType, labelsOrTypes, properties RETURN name, type, entityType, labelsOrTypes, properties"
            )
        except Exception:
            # Fallback for older servers
            try:
                rows = self.run_cypher(
                    "CALL db.indexes() YIELD name, type, entityType, labelsOrTypes, properties RETURN name, type, entityType, labelsOrTypes, properties"
                )
            except Exception:
                rows = []

        found = False
        for r in rows:
            entity_type = r.get("entityType")
            labels = r.get("labelsOrTypes") or []
            props = r.get("properties") or []
            idx_type = r.get("type", "")
            if entity_type == "NODE" and "Chunk" in labels and "embedding" in props and "VECTOR" in str(idx_type).upper():
                found = True
                info(f"Detected vector index: {r.get('name')} ({idx_type}) on :Chunk(embedding)")
                break
        if not found:
            warn("Vector index on :Chunk(embedding) not found. Similarity search will be slow until created.")
    # --------------------------------------------------------------------------------- end check_vector_index()

    # --------------------------------------------------------------------------------- check_base_topology()
    def check_base_topology(self) -> None:
        info("Checking base labels and relationships...")
        counts = {
            "Document": self.count_nodes("Document"),
            "Chunk": self.count_nodes("Chunk"),
            "Entity": self.count_nodes("Entity"),
        }
        info(f"Counts -> Document: {counts['Document']}, Chunk: {counts['Chunk']}, Entity: {counts['Entity']}")
        if counts["Document"] == 0 or counts["Chunk"] == 0:
            warn("Expected nonzero :Document and :Chunk nodes. Did you run initial_graph_build.py?")
        if counts["Entity"] == 0:
            warn("No :Entity nodes found. If entity extraction was disabled, later steps will skip Entity↔Entity edges.")

        # Relationship spot checks (existence only)
        pairs = [
            ("HAS_CHUNK", "Document", "Chunk"),
            ("PART_OF", "Chunk", "Document"),
            ("HAS_ENTITY", "Chunk", "Entity"),
            ("PART_OF", "Entity", "Chunk"),
        ]
        for rel, lhs, rhs in pairs:
            exists = self.relationship_exists(rel, lhs, rhs)
            info(f"Rel check {lhs}-[:{rel}]->{rhs}: {'OK' if exists else 'MISSING'}")
            if not exists:
                warn(f"Expected relationship pattern {lhs}-[:{rel}]->{rhs} not found.")
    # --------------------------------------------------------------------------------- end check_base_topology()

    # ---------- public entry point ----------

    # --------------------------------------------------------------------------------- run_preconditions()
    def run_preconditions(self) -> None:
        """Run all preconditions and log findings before augmentation."""
        info("--- Step 0: Preconditions ---")
        self.check_connectivity()
        self.check_apoc()
        self.check_base_topology()
        self.ensure_constraints()
        self.ensure_document_unique_if_present()
        self.check_vector_index()
        info("Precondition checks completed.")
    # --------------------------------------------------------------------------------- end run_preconditions()

    # --------------------------------------------------------------------------------- close()
    def close(self) -> None:
        """Close the underlying driver if open."""
        try:
            self.driver.close()
        except Exception:
            pass
    # --------------------------------------------------------------------------------- end close()

 # ------------------------------------------------------------------------- end class GraphRelationshipAdder

# =========================================================================
# Standalone Function Definitions
# =========================================================================
 # ----------------------------
 # Edge taxonomy
 # ----------------------------

from enum import Enum

# --------------------------------------------------------------------------------- Direction
class Direction(str, Enum):
    DIRECTED = "DIRECTED"
    UNDIRECTED = "UNDIRECTED"  # logical symmetry; storage may still be directed
# --------------------------------------------------------------------------------- end Direction

# --------------------------------------------------------------------------------- ValueType
class ValueType(str, Enum):
    STRING = "STRING"
    FLOAT = "FLOAT"
    INT = "INT"
    BOOLEAN = "BOOLEAN"
    STRING_LIST = "STRING_LIST"
# --------------------------------------------------------------------------------- end ValueType

# --------------------------------------------------------------------------------- EdgePropSpec
@dataclass(frozen=True)
class EdgePropSpec:
    name: str
    vtype: ValueType
    required: bool = False
    default: Any = None
    description: str = ""
# --------------------------------------------------------------------------------- end EdgePropSpec

# --------------------------------------------------------------------------------- EdgeTypeSpec
@dataclass(frozen=True)
class EdgeTypeSpec:
    """Defines a relationship type and its allowed properties."""
    type_name: str
    direction: Direction
    props: Tuple[EdgePropSpec, ...] = field(default_factory=tuple)
    # Optional vocabulary constraint for a property (e.g., rel_type)
    vocab_for: Optional[str] = None
    vocab_values: Tuple[str, ...] = field(default_factory=tuple)
# --------------------------------------------------------------------------------- end EdgeTypeSpec

# --------------------------------------------------------------------------------- ENTITY_REL_VOCAB
# ---- Controlled vocabularies ----
ENTITY_REL_VOCAB: Tuple[str, ...] = (
    "APPLIES_TO",
    "CITES",
    "REFUTES",
    "CLARIFIES",
    "RELATED_TO",
)
# --------------------------------------------------------------------------------- end ENTITY_REL_VOCAB

# --------------------------------------------------------------------------------- DOCUMENT_REL_VOCAB
DOCUMENT_REL_VOCAB: Tuple[str, ...] = (
    "CITES",
    "REFUTES",
    "SUPPORTS",
    "CLARIFIES",
    "RELATED_TO",
)
# --------------------------------------------------------------------------------- end DOCUMENT_REL_VOCAB

# --------------------------------------------------------------------------------- ENTITY_EDGE_TYPES
# ---- Edge type specifications ----
ENTITY_EDGE_TYPES: Tuple[EdgeTypeSpec, ...] = (
    EdgeTypeSpec(
        type_name="RELATED_TO",
        direction=Direction.DIRECTED,  # stored directed; treat as symmetric in queries
        props=(
            EdgePropSpec("rel_type", ValueType.STRING, required=True, description="Controlled label from ENTITY_REL_VOCAB"),
            EdgePropSpec("score", ValueType.FLOAT, required=False, default=0.0, description="LLM or fusion score"),
            EdgePropSpec("source", ValueType.STRING, required=False, default="llm", description="llm|heuristic|human"),
            EdgePropSpec("evidence_chunks", ValueType.STRING_LIST, required=False, default=tuple(), description="Chunk IDs supporting the edge"),
            EdgePropSpec("justification", ValueType.STRING, required=False, default="", description="Short explanation"),
            EdgePropSpec("created_at", ValueType.STRING, required=False, description="Datetime string"),
        ),
        vocab_for="rel_type",
        vocab_values=ENTITY_REL_VOCAB,
    ),
)
# --------------------------------------------------------------------------------- end ENTITY_EDGE_TYPES

# --------------------------------------------------------------------------------- DOCUMENT_EDGE_TYPES
DOCUMENT_EDGE_TYPES: Tuple[EdgeTypeSpec, ...] = (
    EdgeTypeSpec(
        type_name="SIMILAR_TO",
        direction=Direction.DIRECTED,  # kNN writes directed edges
        props=(
            EdgePropSpec("score", ValueType.FLOAT, required=True, description="Similarity (e.g., cosine)"),
            EdgePropSpec("method", ValueType.STRING, required=False, default="knn", description="knn|nodesim|manual"),
            EdgePropSpec("created_at", ValueType.STRING, required=False, description="Datetime string"),
        ),
    ),
    EdgeTypeSpec(
        type_name="RELATIONSHIP",
        direction=Direction.DIRECTED,
        props=(
            EdgePropSpec("rel_type", ValueType.STRING, required=True, description="Controlled label from DOCUMENT_REL_VOCAB"),
            EdgePropSpec("score", ValueType.FLOAT, required=False, default=0.0, description="Confidence/quality score"),
            EdgePropSpec("source", ValueType.STRING, required=False, default="llm", description="llm|heuristic|human"),
            EdgePropSpec("evidence_chunks", ValueType.STRING_LIST, required=False, default=tuple(), description="Chunk IDs supporting the edge"),
            EdgePropSpec("justification", ValueType.STRING, required=False, default="", description="Short explanation"),
            EdgePropSpec("created_at", ValueType.STRING, required=False, description="Datetime string"),
        ),
        vocab_for="rel_type",
        vocab_values=DOCUMENT_REL_VOCAB,
    ),
)
# --------------------------------------------------------------------------------- end DOCUMENT_EDGE_TYPES

# --------------------------------------------------------------------------------- TAXONOMY
# Unified taxonomy object
# Extending the taxonomy:
# - Add new EdgeTypeSpec entries below and reference them by domain.
# - If a property requires a controlled vocabulary, set `vocab_for` and
#   enumerate `vocab_values`.
TAXONOMY: Dict[str, Tuple[EdgeTypeSpec, ...]] = {
    "ENTITY": ENTITY_EDGE_TYPES,
    "DOCUMENT": DOCUMENT_EDGE_TYPES,
}
# --------------------------------------------------------------------------------- end TAXONOMY

# --------------------------------------------------------------------------------- print_taxonomy()
def print_taxonomy() -> None:
    info("=== Edge Taxonomy ===")
    for domain, specs in TAXONOMY.items():
        print(f"Domain: {domain}")
        for et in specs:
            print(f"  - Type: {et.type_name} ({et.direction})")
            if et.vocab_for:
                print(f"    Vocabulary for '{et.vocab_for}': {', '.join(et.vocab_values)}")
            for prop in et.props:
                req = "required" if prop.required else "optional"
                default_str = "" if prop.default in (None, (), []) else f" [default={prop.default}]"
                desc_str = "" if not prop.description else f" — {prop.description}"
                print(f"    * {prop.name}: {prop.vtype} ({req}){default_str}{desc_str}")

# --------------------------------------------------------------------------------- end print_taxonomy()
    
# ---- Validation helpers ----

# --------------------------------------------------------------------------------- _coerce_type()
def _coerce_type(value: Any, vtype: ValueType) -> Any:
    if value is None:
        return None
    try:
        if vtype == ValueType.STRING:
            return str(value)
        if vtype == ValueType.FLOAT:
            return float(value)
        if vtype == ValueType.INT:
            return int(value)
        if vtype == ValueType.BOOLEAN:
            return bool(value)
        if vtype == ValueType.STRING_LIST:
            if isinstance(value, (list, tuple)):
                return [str(v) for v in value]
            # allow comma-separated string
            return [s.strip() for s in str(value).split(",") if s.strip()]
    except Exception:
        return value  # leave as-is; validator will flag
    return value
# --------------------------------------------------------------------------------- end _coerce_type()

# --------------------------------------------------------------------------------- validate_edge_payload()
def validate_edge_payload(domain: str, type_name: str, payload: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Validate and normalize relationship properties against the taxonomy.

    Validation policy:
    - Enforces required properties and fills defaults when provided.
    - Unknown properties are allowed (warning emitted) to support forward-compat fields.
    - If a controlled vocabulary is specified (`vocab_for`), the value must match.

    Returns:
        Tuple[bool, List[str], Dict[str, Any]]: (ok, errors, normalized_props)
    """
    specs = TAXONOMY.get(domain.upper())
    if not specs:
        return False, [f"Unknown domain '{domain}'"], {}

    et: Optional[EdgeTypeSpec] = next((e for e in specs if e.type_name == type_name), None)
    if not et:
        return False, [f"Unknown edge type '{type_name}' for domain '{domain}'"], {}

    errors: List[str] = []
    normalized: Dict[str, Any] = {}

    # Required + known props
    known = {p.name: p for p in et.props}
    for name, spec in known.items():
        if name in payload and payload[name] is not None:
            normalized[name] = _coerce_type(payload[name], spec.vtype)
        elif spec.required:
            errors.append(f"Missing required property '{name}'")
        elif spec.default not in (None, (), []):
            normalized[name] = spec.default

    # Unknown props (warn but allow passthrough)
    for k, v in payload.items():
        if k not in known:
            warn(f"Unknown property '{k}' for edge type {type_name}; passing through")
            normalized[k] = v

    # Enforce vocabulary if any
    if et.vocab_for:
        key = et.vocab_for
        val = normalized.get(key)
        if val is None:
            errors.append(f"Missing required vocab property '{key}'")
        elif val not in et.vocab_values:
            errors.append(f"Invalid value '{val}' for '{key}'. Allowed: {', '.join(et.vocab_values)}")

    return len(errors) == 0, errors, normalized
# --------------------------------------------------------------------------------- end validate_edge_payload()

# ----------------------------
# Candidate generation
# ----------------------------

# -- 2B) Document embedding projection for KNN (requires :Document.doc_embedding) --
# --------------------------------------------------------------------------------- check_document_embedding_presence()
def check_document_embedding_presence(run_cypher_fn, sample: int = 50) -> float:
    """Estimate coverage of document embeddings for similarity operations.

    Returns a fraction in [0, 1] over a small sample, indicating whether
    downstream document similarity is feasible.
    """
    rows = run_cypher_fn(
        "MATCH (d:Document) WITH d LIMIT $n "
        "RETURN count(CASE WHEN d.doc_embedding IS NOT NULL THEN 1 END) AS have, count(*) AS total",
        {"n": sample}
    )
    if not rows:
        return 0.0
    row = rows[0]
    have, total = int(row.get("have", 0)), int(row.get("total", 0))
    return (have / total) if total else 0.0
# --------------------------------------------------------------------------------- end check_document_embedding_presence()

# ----------------------------
# Candidate generation implementations
# ----------------------------

# --------------------------------------------------------------------------------- stream_entity_similarity()
def stream_entity_similarity(run_cypher_fn,
                            similarity_cutoff: float = 0.2,
                            limit: int = 50000) -> List[Dict[str, Any]]:
    """Compute entity-to-entity Jaccard similarity over shared chunks.

    Query outline:
      1) Count co-occurring chunks for (e1, e2)
      2) Count degree for e1 and e2
      3) jaccard = co / (deg1 + deg2 - co)

    Returns:
      List of rows: {e1_id, e2_id, similarity}
    """
    q = (
        "MATCH (e1:Entity)<-[:HAS_ENTITY]-(c:Chunk)-[:HAS_ENTITY]->(e2:Entity) "
        "WHERE id(e1) < id(e2) "
        "WITH e1, e2, count(DISTINCT c) AS co "
        "MATCH (e1)<-[:HAS_ENTITY]-(c1:Chunk) "
        "WITH e1, e2, co, count(DISTINCT c1) AS deg1 "
        "MATCH (e2)<-[:HAS_ENTITY]-(c2:Chunk) "
        "WITH e1, e2, co, deg1, count(DISTINCT c2) AS deg2 "
        "WITH e1, e2, co, deg1, deg2, CASE WHEN (deg1 + deg2 - co) = 0 THEN 0.0 ELSE co * 1.0 / (deg1 + deg2 - co) END AS jaccard "
        "WHERE jaccard >= $cutoff "
        "RETURN id(e1) AS e1_id, id(e2) AS e2_id, jaccard AS similarity "
        "ORDER BY similarity DESC LIMIT $limit"
    )
    try:
        rows = run_cypher_fn(q, {"cutoff": similarity_cutoff, "limit": limit})
        info(f"Entity similarity: returned {len(rows)} pairs (cutoff={similarity_cutoff}).")
        return rows
    except Exception as e:
        warn(f"Entity similarity failed: {e}")
        return []
# --------------------------------------------------------------------------------- end stream_entity_similarity()


# --------------------------------------------------------------------------------- stream_doc_similarity()
def stream_doc_similarity(run_cypher_fn,
                          similarity_cutoff: float = 0.75,
                          limit: int = 50000) -> List[Dict[str, Any]]:
    """Compute document-to-document cosine similarity.

    Requirements:
      - `:Document.doc_embedding` vectors populated (Float[] of same dimension).

    Returns:
      List of rows: {d1_id, d2_id, similarity}. Caller may post-filter per-node top_k.
    """
    q = (
        "MATCH (d1:Document) WHERE d1.doc_embedding IS NOT NULL "
        "MATCH (d2:Document) WHERE d2.doc_embedding IS NOT NULL AND id(d2) > id(d1) "
        "WITH d1, d2, d1.doc_embedding AS v1, d2.doc_embedding AS v2, range(0, size(v1)-1) AS idxs "
        "WITH d1, d2, "
        "     reduce(dot = 0.0, i IN idxs | dot + coalesce(v1[i], 0.0) * coalesce(v2[i], 0.0)) AS dot, "
        "     sqrt(reduce(n1 = 0.0, x IN v1 | n1 + x*x)) AS n1, "
        "     sqrt(reduce(n2 = 0.0, y IN v2 | n2 + y*y)) AS n2 "
        "WITH d1, d2, CASE WHEN n1 = 0.0 OR n2 = 0.0 THEN 0.0 ELSE dot / (n1 * n2) END AS sim "
        "WHERE sim >= $cutoff "
        "RETURN id(d1) AS d1_id, id(d2) AS d2_id, sim AS similarity "
        "ORDER BY similarity DESC LIMIT $limit"
    )
    try:
        rows = run_cypher_fn(q, {"cutoff": similarity_cutoff, "limit": limit})
        info(f"Document similarity: returned {len(rows)} pairs (cutoff={similarity_cutoff}).")
        return rows
    except Exception as e:
        warn(f"Document similarity failed: {e}")
        return []
# --------------------------------------------------------------------------------- end stream_doc_similarity()

# ----------------------------
# LLM labeling + MERGE back into Neo4j
# ----------------------------

@dataclass
class RelationInput:
    domain: str                 # "ENTITY" | "DOCUMENT"
    subject_id: int             # Neo4j internal id()
    object_id: int
    evidence_chunks: List[Dict] # [{chunk_id, text}] small set for context
    seed_score: float = 0.0     # similarity score or co-occur score

@dataclass
class RelationOutput:
    domain: str
    type_name: str              # e.g., "RELATED_TO" or "RELATIONSHIP"
    subject_id: int
    object_id: int
    props: Dict[str, Any]       # validated properties (rel_type, score, source, evidence_chunks, ...)

# ---- Step 5: Retrieve evidence ----

# --------------------------------------------------------------------------------- fetch_entity_evidence()
def fetch_entity_evidence(run_cypher_fn, e1_id: int, e2_id: int, top_n: int = 6) -> List[Dict[str, Any]]:
    """Fetch up to top_n chunks that connect both entities via HAS_ENTITY."""
    q = (
        "MATCH (e1:Entity) WHERE id(e1)=$e1 "
        "MATCH (e2:Entity) WHERE id(e2)=$e2 "
        "MATCH (e1)<-[:HAS_ENTITY]-(c:Chunk)-[:HAS_ENTITY]->(e2) "
        "RETURN c.chunk_id AS chunk_id, c.text AS text, c.page AS page "
        "LIMIT $n"
    )
    rows = run_cypher_fn(q, {"e1": e1_id, "e2": e2_id, "n": top_n})
    if not rows:
        q2 = (
            "MATCH (e:Entity)<-[:HAS_ENTITY]-(c:Chunk) "
            "WHERE id(e) IN [$e1, $e2] "
            "RETURN c.chunk_id AS chunk_id, c.text AS text, c.page AS page "
            "LIMIT $n"
        )
        rows = run_cypher_fn(q2, {"e1": e1_id, "e2": e2_id, "n": top_n})
    return rows
# --------------------------------------------------------------------------------- end fetch_entity_evidence()

# --------------------------------------------------------------------------------- fetch_document_evidence()
def fetch_document_evidence(run_cypher_fn, d1_id: int, d2_id: int, per_doc: int = 5) -> List[Dict[str, Any]]:
    """Fetch a few chunks from each document as context."""
    q = (
        "MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk) "
        "WHERE id(d) = $d1 "
        "RETURN d.doc_id AS doc_id, c.chunk_id AS chunk_id, c.text AS text, c.page AS page "
        "LIMIT $n"
    )
    a = run_cypher_fn(q, {"d1": d1_id, "n": per_doc})
    q = (
        "MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk) "
        "WHERE id(d) = $d2 "
        "RETURN d.doc_id AS doc_id, c.chunk_id AS chunk_id, c.text AS text, c.page AS page "
        "LIMIT $n"
    )
    b = run_cypher_fn(q, {"d2": d2_id, "n": per_doc})
    return a + b
# --------------------------------------------------------------------------------- end fetch_document_evidence()

# ---- LLM labeling ----

# --------------------------------------------------------------------------------- label_with_llm()
def label_with_llm(batch: List[RelationInput], model: str = None) -> List[RelationOutput]:
    """Label relationships using OpenAI LLM (gpt-5-nano by default).

    Args:
        batch: List of RelationInput objects to process
        model: OpenAI model to use (defaults to gpt-5-nano)

    Returns:
        List of RelationOutput objects with LLM-generated labels
    """
    if not batch:
        return []

    # Use gpt-5-nano as default if no model specified
    model = model or os.getenv("OPENAI_MODEL_NANO", os.getenv("OPENAI_MODEL", "gpt-5-nano"))

    try:
        import openai

        # Use monitored client if available
        if MONITORING_AVAILABLE:
            from pathlib import Path
            monitoring_dir = Path("monitoring_logs")
            monitoring_dir.mkdir(exist_ok=True)
            log_level = LogLevel.DETAILED if os.getenv("VERBOSE", "false").lower() == "true" else LogLevel.STANDARD
            client = MonitoredOpenAIClient(
                api_key=os.getenv("OPENAI_API_KEY"),
                log_level=log_level,
                output_file=monitoring_dir / "openai_relationship_labeling.jsonl"
            )
        else:
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        outputs: List[RelationOutput] = []

        for item in batch:
            # Create prompt based on domain
            if item.domain.upper() == "ENTITY":
                prompt = f"""Analyze the relationship between two entities based on the evidence provided.

Entity 1 ID: {item.subject_id}
Entity 2 ID: {item.object_id}
Similarity Score: {item.seed_score}

Evidence chunks: {item.evidence_chunks[:3]}

Determine the most appropriate relationship type from: WORKS_FOR, LOCATED_IN, PART_OF, RELATED_TO, INFLUENCES, ASSOCIATED_WITH

Respond with JSON only:
{{"rel_type": "relationship_type", "score": confidence_0_to_1, "justification": "brief_explanation"}}"""
            else:
                prompt = f"""Analyze the relationship between two documents based on the evidence provided.

Document 1 ID: {item.subject_id}
Document 2 ID: {item.object_id}
Similarity Score: {item.seed_score}

Evidence chunks: {item.evidence_chunks[:3]}

Determine the most appropriate relationship type from: SIMILAR_TOPIC, REFERENCES, CONTRADICTS, EXTENDS, RELATED_TO

Respond with JSON only:
{{"rel_type": "relationship_type", "score": confidence_0_to_1, "justification": "brief_explanation"}}"""

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=150
                )

                import json
                result = json.loads(response.choices[0].message.content.strip())

                # Validate and create output
                domain = item.domain.upper()
                type_name = "RELATED_TO" if domain == "ENTITY" else "RELATIONSHIP"

                props = {
                    "rel_type": result.get("rel_type", "RELATED_TO"),
                    "score": min(1.0, max(0.0, float(result.get("score", item.seed_score)))),
                    "source": "llm",
                    "evidence_chunks": [e.get("chunk_id") for e in item.evidence_chunks][:8],
                    "justification": result.get("justification", "LLM-generated relationship")
                }

                ok, errs, validated_props = validate_edge_payload(domain, type_name, props)
                if ok:
                    outputs.append(RelationOutput(
                        domain=domain, type_name=type_name,
                        subject_id=item.subject_id, object_id=item.object_id,
                        props=validated_props
                    ))
                else:
                    warn(f"LLM output validation failed ({domain}): {errs}")
                    # Fallback to stub behavior
                    outputs.extend(label_with_llm_stub([item]))

            except Exception as e:
                warn(f"LLM call failed for {item.domain} relationship: {e}")
                # Fallback to stub behavior
                outputs.extend(label_with_llm_stub([item]))

        return outputs

    except Exception as e:
        warn(f"LLM labeling failed, falling back to stub: {e}")
        return label_with_llm_stub(batch)
# --------------------------------------------------------------------------------- end label_with_llm()

# --------------------------------------------------------------------------------- label_with_llm_stub()
def label_with_llm_stub(batch: List[RelationInput]) -> List[RelationOutput]:
    """Rule-based placeholder that assigns generic types.
    Replace with an actual LLM call that returns structured JSON.
    """
    outputs: List[RelationOutput] = []
    for item in batch:
        if item.domain.upper() == "ENTITY":
            ok, errs, props = validate_edge_payload(
                "ENTITY", "RELATED_TO",
                {"rel_type": "RELATED_TO", "score": max(0.3, float(item.seed_score)), "source": "stub",
                 "evidence_chunks": [e.get("chunk_id") for e in item.evidence_chunks][:8],
                 "justification": "Heuristic: entity co-occurrence or proximity."}
            )
            if not ok:
                warn(f"Validation failed (ENTITY): {errs}")
                continue
            outputs.append(RelationOutput(
                domain="ENTITY", type_name="RELATED_TO",
                subject_id=item.subject_id, object_id=item.object_id, props=props
            ))
        else:
            ok, errs, props = validate_edge_payload(
                "DOCUMENT", "RELATIONSHIP",
                {"rel_type": "RELATED_TO", "score": max(0.3, float(item.seed_score)), "source": "stub",
                 "evidence_chunks": [e.get('chunk_id') for e in item.evidence_chunks][:8],
                 "justification": "Heuristic: document similarity or shared topics."}
            )
            if not ok:
                warn(f"Validation failed (DOCUMENT): {errs}")
                continue
            outputs.append(RelationOutput(
                domain="DOCUMENT", type_name="RELATIONSHIP",
                subject_id=item.subject_id, object_id=item.object_id, props=props
            ))
    return outputs
# --------------------------------------------------------------------------------- end label_with_llm_stub()

# ---- MERGE relationships ----

# --------------------------------------------------------------------------------- merge_entity_relations()
def merge_entity_relations(run_cypher_fn, rels: List[RelationOutput]) -> int:
    """MERGE (:Entity)-[:RELATED_TO {...}]->(:Entity) with property upserts."""
    if not rels:
        return 0
    payload = [asdict(r) for r in rels if r.domain.upper() == "ENTITY" and r.type_name == "RELATED_TO"]
    if not payload:
        return 0
    q = (
        "UNWIND $rows AS r "
        "MATCH (s:Entity) WHERE id(s)=r.subject_id "
        "MATCH (o:Entity) WHERE id(o)=r.object_id "
        "MERGE (s)-[rel:RELATED_TO]->(o) "
        "ON CREATE SET rel += r.props, rel.created_at = datetime() "
        "ON MATCH  SET rel.score = CASE WHEN rel.score IS NULL OR r.props.score > rel.score THEN r.props.score ELSE rel.score END, "
        "              rel.rel_type = r.props.rel_type, "
        "              rel.source = r.props.source, "
        "              rel.justification = r.props.justification, "
        "              rel.evidence_chunks = CASE "
        "                   WHEN rel.evidence_chunks IS NOT NULL THEN rel.evidence_chunks + coalesce(r.props.evidence_chunks, []) "
        "                   ELSE coalesce(r.props.evidence_chunks, []) END"
    )
    try:
        run_cypher_fn(q, {"rows": payload})
        return len(payload)
    except Exception as e:
        warn(f"merge_entity_relations failed: {e}")
        return 0
# --------------------------------------------------------------------------------- end merge_entity_relations()

# --------------------------------------------------------------------------------- merge_document_relations()
def merge_document_relations(run_cypher_fn, rels: List[RelationOutput]) -> int:
    """MERGE (:Document)-[:RELATIONSHIP {...}]->(:Document)."""
    if not rels:
        return 0
    payload = [asdict(r) for r in rels if r.domain.upper() == "DOCUMENT" and r.type_name == "RELATIONSHIP"]
    if not payload:
        return 0
    q = (
        "UNWIND $rows AS r "
        "MATCH (a:Document) WHERE id(a)=r.subject_id "
        "MATCH (b:Document) WHERE id(b)=r.object_id "
        "MERGE (a)-[rel:RELATIONSHIP]->(b) "
        "ON CREATE SET rel += r.props, rel.created_at = datetime() "
        "ON MATCH  SET rel.score = CASE WHEN rel.score IS NULL OR r.props.score > rel.score THEN r.props.score ELSE rel.score END, "
        "              rel.rel_type = r.props.rel_type, "
        "              rel.source = r.props.source, "
        "              rel.justification = r.props.justification, "
        "              rel.evidence_chunks = CASE "
        "                   WHEN rel.evidence_chunks IS NOT NULL THEN rel.evidence_chunks + coalesce(r.props.evidence_chunks, []) "
        "                   ELSE coalesce(r.props.evidence_chunks, []) END"
    )
    try:
        run_cypher_fn(q, {"rows": payload})
        return len(payload)
    except Exception as e:
        warn(f"merge_document_relations failed: {e}")
        return 0
# --------------------------------------------------------------------------------- end merge_document_relations()

# =========================================================================
# End of File
# =========================================================================