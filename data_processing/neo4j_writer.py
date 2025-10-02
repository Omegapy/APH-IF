# -------------------------------------------------------------------------
# File: data_processing/neo4j_writer.py
# Author: Alexander Ricciardi
# Date: 2025-10-02
# [File Path] data_processing/neo4j_writer.py
# ------------------------------------------------------------------------
# Project: APH-IF — Advanced Parallel HybridRAG – Intelligent Fusion
#
# Module Functionality:
#   Neo4j writer utilities that support idempotent upserts for documents, chunks,
#   entities, and relationships, and ensure a backend-compatible vector index.
#
# Module Contents Overview:
# - Class: Neo4jWriter
#
# Dependencies / Imports:
# - Standard Library: logging, datetime, typing
# - Third-Party: neo4j, tqdm
#
# Usage / Integration:
#   Used by data_processing ingestion phases to persist graph data into Neo4j.
#
# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------
"""Neo4j writer for document, chunk, and entity operations.

Handles idempotent upserts of documents, chunks, and relationships.
Creates and manages vector index for backend compatibility.
"""

from __future__ import annotations

# __________________________________________________________________________
# Imports

import logging
from datetime import datetime
from typing import Optional

from neo4j import GraphDatabase, Driver
from tqdm import tqdm


logger = logging.getLogger(__name__)


# ____________________________________________________________________________
# Class Definitions
# ------------------------------------------------------------------------- class Neo4jWriter
# TODO: Regular class (manages external DB resources); not a dataclass candidate.
class Neo4jWriter:
    """Writer for Neo4j operations with batching and index management.

    Responsibilities:
        - Manage a Neo4j driver and session lifecycle.
        - Provide idempotent upserts for documents, chunks, entities, and relationships.
        - Ensure a backend-compatible vector index for embeddings.

    Attributes:
        driver: Underlying Neo4j Driver instance.
        database: Target Neo4j database name.
        batch_size: Batch size for batched write operations.
    """

    # -------------------------------------------------------------- __init__()
    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
        batch_size: int = 500,
    ) -> None:
        """Initialize Neo4j writer.

        Args:
            uri: Neo4j connection URI
            username: Neo4j username
            password: Neo4j password
            database: Neo4j database name
            batch_size: Number of records per batch write
        """
        self.driver: Driver = GraphDatabase.driver(uri, auth=(username, password))
        self.database = database
        self.batch_size = batch_size

        logger.info(f"Initialized Neo4j writer: {uri}, database={database}")

        # Test connection
        self._verify_connection()

    # -------------------------------------------------------------- close()
    def close(self) -> None:
        """Close the Neo4j driver connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

    # -------------------------------------------------------------- _verify_connection()
    def _verify_connection(self) -> None:
        """Verify Neo4j connection is working."""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 as test")
                result.single()
            logger.info("Neo4j connection verified")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    # -------------------------------------------------------------- ensure_vector_index()
    def ensure_vector_index(
        self,
        index_name: str = "chunk_embedding_index",
        node_label: str = "Chunk",
        property_name: str = "embedding",
        dimensions: int = 3072,
    ) -> None:
        """Ensure vector index exists for chunk embeddings.

        Args:
            index_name: Name of the vector index
            node_label: Node label to index
            property_name: Property containing embeddings
            dimensions: Vector dimensions
        """
        logger.info(f"Ensuring vector index: {index_name}")

        with self.driver.session(database=self.database) as session:
            # Check if index exists
            check_query = """
            SHOW INDEXES
            YIELD name, type
            WHERE name = $index_name AND type = 'VECTOR'
            RETURN count(*) as count
            """
            result = session.run(check_query, index_name=index_name)
            count = result.single()["count"]

            if count > 0:
                logger.info(f"Vector index '{index_name}' already exists")
                return

            # Create vector index
            # Note: Syntax may vary by Neo4j version
            create_query = f"""
            CREATE VECTOR INDEX {index_name} IF NOT EXISTS
            FOR (n:{node_label})
            ON n.{property_name}
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {dimensions},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
            """

            try:
                session.run(create_query)
                logger.info(f"Created vector index: {index_name}")
            except Exception as e:
                logger.warning(
                    f"Failed to create vector index (may already exist or "
                    f"syntax not supported): {e}"
                )
                # Try alternative syntax for older Neo4j versions
                try:
                    alt_query = f"""
                    CALL db.index.vector.createNodeIndex(
                        '{index_name}',
                        '{node_label}',
                        '{property_name}',
                        {dimensions},
                        'cosine'
                    )
                    """
                    session.run(alt_query)
                    logger.info(f"Created vector index using alternative syntax: {index_name}")
                except Exception as e2:
                    logger.error(f"Failed to create vector index with alternative syntax: {e2}")
                    raise

    # -------------------------------------------------------------- upsert_document()
    def upsert_document(
        self,
        document_id: str,
        title: str,
        filename: str,
    ) -> None:
        """Upsert a document node.

        Args:
            document_id: Unique document identifier
            title: Document title
            filename: Source filename
        """
        query = """
        MERGE (d:Document {document_id: $document_id})
        SET d.title = $title,
            d.filename = $filename,
            d.created_at = datetime($created_at),
            d.updated_at = datetime()
        RETURN d.document_id as id
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(
                query,
                document_id=document_id,
                title=title,
                filename=filename,
                created_at=datetime.utcnow().isoformat(),
            )
            result.single()

        logger.debug(f"Upserted document: {document_id}")

    # -------------------------------------------------------------- upsert_chunks_batch()
    def upsert_chunks_batch(
        self,
        chunks: list,
        show_progress: bool = True,
    ) -> int:
        """Upsert chunks in batches with relationships to documents.

        Args:
            chunks: List of ChunkRecord objects
            show_progress: Whether to show progress bar

        Returns:
            Number of chunks upserted
        """
        if not chunks:
            return 0

        logger.info(f"Upserting {len(chunks)} chunks in batches of {self.batch_size}")

        query = """
        UNWIND $chunks as chunk
        MERGE (c:Chunk {chunk_id: chunk.chunk_id})
        SET c.document_id = chunk.document_id,
            c.page = chunk.page,
            c.text = chunk.text,
            c.section = chunk.section,
            c.tokens = chunk.tokens,
            c.embedding = chunk.embedding,
            c.updated_at = datetime()

        WITH c, chunk
        MERGE (d:Document {document_id: chunk.document_id})
        MERGE (d)-[:HAS_CHUNK]->(c)

        RETURN count(c) as count
        """

        total_upserted = 0

        # Process in batches
        num_batches = (len(chunks) + self.batch_size - 1) // self.batch_size

        iterator = range(0, len(chunks), self.batch_size)
        if show_progress:
            iterator = tqdm(
                iterator,
                total=num_batches,
                desc="Upserting chunks",
                unit="batch",
            )

        for start_idx in iterator:
            end_idx = min(start_idx + self.batch_size, len(chunks))
            batch = chunks[start_idx:end_idx]

            # Convert chunks to dictionaries
            chunk_dicts = [
                {
                    "chunk_id": c.chunk_id,
                    "document_id": c.document_id,
                    "page": c.page,
                    "text": c.text,
                    "section": c.section,
                    "tokens": c.tokens,
                    "embedding": c.embedding,
                }
                for c in batch
            ]

            with self.driver.session(database=self.database) as session:
                result = session.run(query, chunks=chunk_dicts)
                batch_count = result.single()["count"]
                total_upserted += batch_count

        logger.info(f"Successfully upserted {total_upserted} chunks")
        return total_upserted

    # -------------------------------------------------------------- upsert_entities()
    def upsert_entities(
        self,
        chunk_id: str,
        entities: list,
    ) -> None:
        """Upsert entities and HAS_ENTITY relationships.

        Args:
            chunk_id: Chunk identifier
            entities: List of Entity objects
        """
        if not entities:
            return

        query = """
        UNWIND $entities AS entity
        MERGE (e:Entity {name: entity.name})
        ON CREATE SET
            e.type = entity.type,
            e.canonical_name = entity.canonical_name,
            e.occurrences = entity.occurrences
        ON MATCH SET
            e.occurrences = coalesce(e.occurrences, 0) + entity.occurrences

        WITH e, entity
        MATCH (c:Chunk {chunk_id: $chunk_id})
        MERGE (c)-[:HAS_ENTITY]->(e)
        """

        entity_dicts = [
            {
                "name": ent.name,
                "type": ent.type,
                "canonical_name": ent.canonical_name,
                "occurrences": ent.occurrences,
            }
            for ent in entities
        ]

        with self.driver.session(database=self.database) as session:
            session.run(query, chunk_id=chunk_id, entities=entity_dicts)

    # -------------------------------------------------------------- upsert_entities_batch()
    def upsert_entities_batch(
        self,
        chunk_entities: dict[str, list],
        show_progress: bool = True,
    ) -> int:
        """Batch upsert entities for multiple chunks.

        Args:
            chunk_entities: Dict mapping chunk_id -> list of entities
            show_progress: Show progress bar

        Returns:
            Total number of entities upserted
        """
        if not chunk_entities:
            return 0

        total_entities = sum(len(ents) for ents in chunk_entities.values())
        logger.info(f"Upserting {total_entities} entities for {len(chunk_entities)} chunks")

        iterator = chunk_entities.items()
        if show_progress:
            iterator = tqdm(iterator, desc="Upserting entities", unit="chunk")

        for chunk_id, entities in iterator:
            self.upsert_entities(chunk_id, entities)

        logger.info(f"Successfully upserted entities for {len(chunk_entities)} chunks")
        return total_entities

    # -------------------------------------------------------------- upsert_relationships()
    def upsert_relationships(
        self,
        relationships: list,
    ) -> int:
        """Upsert entity relationships.

        Args:
            relationships: List of EntityRelationship objects

        Returns:
            Number of relationships created
        """
        if not relationships:
            return 0

        query = """
        UNWIND $relationships AS rel
        MATCH (source:Entity {name: rel.source_entity})
        MATCH (target:Entity {name: rel.target_entity})
        MERGE (source)-[r:RELATED_TO {type: rel.relationship_type}]->(target)
        SET r.confidence = rel.confidence,
            r.source = rel.source,
            r.reason = rel.reason,
            r.updated_at = datetime()
        RETURN count(r) as count
        """

        relationship_dicts = [
            {
                "source_entity": rel.source_entity,
                "target_entity": rel.target_entity,
                "relationship_type": rel.relationship_type,
                "confidence": rel.confidence,
                "source": rel.source,
                "reason": rel.reason,
            }
            for rel in relationships
        ]

        with self.driver.session(database=self.database) as session:
            result = session.run(query, relationships=relationship_dicts)
            count = result.single()["count"]

        logger.info(f"Upserted {count} entity relationships")
        return count

    # -------------------------------------------------------------- upsert_relationships_batch()
    def upsert_relationships_batch(
        self,
        chunk_relationships: dict[str, list],
        show_progress: bool = True,
    ) -> int:
        """Batch upsert relationships for multiple chunks.

        Args:
            chunk_relationships: Dict mapping chunk_id -> list of relationships
            show_progress: Show progress bar

        Returns:
            Total number of relationships upserted
        """
        if not chunk_relationships:
            return 0

        # Flatten all relationships
        all_relationships = []
        for relationships in chunk_relationships.values():
            all_relationships.extend(relationships)

        if not all_relationships:
            return 0

        logger.info(
            f"Upserting {len(all_relationships)} relationships from "
            f"{len(chunk_relationships)} chunks"
        )

        # Process in batches
        batch_size = 100
        total_upserted = 0

        from tqdm import tqdm

        iterator = range(0, len(all_relationships), batch_size)
        if show_progress:
            iterator = tqdm(
                iterator,
                desc="Upserting relationships",
                unit="batch",
                total=(len(all_relationships) + batch_size - 1) // batch_size,
            )

        for start_idx in iterator:
            end_idx = min(start_idx + batch_size, len(all_relationships))
            batch = all_relationships[start_idx:end_idx]
            count = self.upsert_relationships(batch)
            total_upserted += count

        logger.info(
            f"Successfully upserted {total_upserted} entity relationships"
        )
        return total_upserted

    # -------------------------------------------------------------- clear_database()
    def clear_database(self, confirm: bool = False) -> None:
        """Clear all nodes and relationships from the database.

        Args:
            confirm: Must be True to actually clear the database

        Warning:
            This is a destructive operation!
        """
        if not confirm:
            logger.warning("Database clear called without confirmation - skipping")
            return

        logger.warning("Clearing entire Neo4j database!")

        query = """
        MATCH (n)
        DETACH DELETE n
        """

        with self.driver.session(database=self.database) as session:
            session.run(query)

        logger.info("Database cleared")

    # -------------------------------------------------------------- get_document_count()
    def get_document_count(self) -> int:
        """Get count of Document nodes.

        Returns:
            Number of documents in database
        """
        query = "MATCH (d:Document) RETURN count(d) as count"

        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            return result.single()["count"]

    # -------------------------------------------------------------- get_chunk_count()
    def get_chunk_count(self) -> int:
        """Get count of Chunk nodes.

        Returns:
            Number of chunks in database
        """
        query = "MATCH (c:Chunk) RETURN count(c) as count"

        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            return result.single()["count"]

    # -------------------------------------------------------------- __enter__()
    def __enter__(self):
        """Context manager entry."""
        return self

    # -------------------------------------------------------------- __exit__()
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

# ------------------------------------------------------------------------- end class Neo4jWriter

# __________________________________________________________________________
# End of File
#
