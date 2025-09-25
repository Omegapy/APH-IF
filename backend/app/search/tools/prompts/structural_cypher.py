# -------------------------------------------------------------------------
# File: structural_cypher.py
# Author: Alexander Ricciardi
# Date: 2025-09-15
# [File Path] backend/app/search/tools/prompts/structural_cypher.py
# ------------------------------------------------------------------------
# Project:
# Project: APH-IF
#
# Project description:
# Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF)
# is a novel Retrieval Augmented Generation (RAG) system that differs from
# traditional RAG approaches by performing semantic and traversal searches
# concurrently, rather than sequentially, and fusing the results using an LLM
# or an LRM to generate the final response.
# -------------------------------------------------------------------------

# --- Module Functionality ---
#   Token-aware prompt builders and templates for LLM structural Cypher generation and
#   narrative summarization. Provides utilities to assemble system/user prompts with
#   schema-aware context, token budgeting, and optional few-shot examples tailored to the
#   backend tools module.
# -------------------------------------------------------------------------

# --- Module Contents Overview ---
# - Constant: FEW_SHOT_EXAMPLES
# - Constant: SYSTEM_PROMPT_TEMPLATE
# - Constant: USER_PROMPT_TEMPLATE
# - Constant: EXAMPLES_SECTION_TEMPLATE
# - Class (dataclass): PromptComponents
# - Class (dataclass): SchemaSubset
# - Class: StructuralCypherPromptBuilder
# - Class (dataclass): NarrativePromptComponents
# - Class: StructuralNarrativePromptBuilder
# - Function: safe_lower
# - Factory: get_structural_cypher_prompt_builder
# - Factory: get_structural_narrative_prompt_builder
# -------------------------------------------------------------------------

# --- Dependencies / Imports ---
# - Standard Library: re (regex), logging (logger), dataclasses (dataclass), typing (hints)
# - Third-Party: (none)
# - Local Project Modules: integrated via higher-level tools in the backend
# --- Requirements ---
# - Python 3.12
# -------------------------------------------------------------------------

# --- Usage / Integration ---
# This module is used by the LLM structural tools to build token-conscious prompts for:
# - Generating Cypher queries from natural language with schema adherence
# - Composing narrative answers from numbered sources while preserving citations
# Factories provide configured builders consumed within the backend search tools.

# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------

"""
Token-Aware Prompt System for LLM Structural Cypher Generation.

Enhanced prompt builder that manages token budgets, provides few-shot examples,
and intelligently truncates structural schema summaries based on query relevance.
"""

# __________________________________________________________________________
# Imports

from __future__ import annotations

import re
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass

# __________________________________________________________________________
# Global Constants / Variables
logger = logging.getLogger(__name__)

# __________________________________________________________________________
# Standalone Function Definitions
#
# =========================================================================
# Utility Functions
# =========================================================================

# --------------------------------------------------------------------------------- safe_lower()
def safe_lower(value: Any) -> str:
    """Safely convert a value to a lowercase string.

    Provides a defensive lowercasing utility that tolerates `None` and non-string inputs
    without raising, which is useful when normalizing mixed user/schema inputs during
    prompt construction.

    Args:
        value: Any Python value to normalize (e.g., str, int, None).

    Returns:
        A lowercase string representation of the input. Returns an empty string when
        the input is `None`.

    Examples:
        >>> safe_lower("Chunk")
        'chunk'
        >>> safe_lower(123)
        '123'
        >>> safe_lower(None)
        ''
    """
    if isinstance(value, str):
        return value.lower()
    return str(value).lower() if value is not None else ""
# --------------------------------------------------------------------------------- end safe_lower()

# =========================================================================
# Few-Shot Examples for Common Graph Patterns
# =========================================================================

FEW_SHOT_EXAMPLES = [
    {
        "description": "Chunk/Entity keyword search (chunk-pivot, citations)",
        "user_query": "Find content about machine learning",
        "cypher": (
            "MATCH (c:Chunk)-[:HAS_ENTITY]->(e:Entity)\n"
            "WHERE toLower(c.text) CONTAINS 'machine learning' OR toLower(e.name) CONTAINS 'machine learning'\n"
            "RETURN c.text, c.chunk_id AS chunk_id, c.page AS page\n"
            "LIMIT 10"
        )
    },
    {
        "description": "Entities mentioned in a document (return chunks for citations)", 
        "user_query": "What entities are mentioned in document 'doc123'?",
        "cypher": (
            "MATCH (d:Document {doc_id: 'doc123'})-[:HAS_CHUNK]->(c:Chunk)-[:HAS_ENTITY]->(e:Entity)\n"
            "RETURN c.text, c.chunk_id AS chunk_id, c.page AS page\n"
            "LIMIT 20"
        )
    },
    {
        "description": "Entity→Entity expansion to discover related chunks (up to 3 hops)",
        "user_query": "Find content related to entities connected to 'artificial intelligence'",
        "cypher": (
            "MATCH (e1:Entity {name: 'artificial intelligence'})-[:RELATED_TO*0..3]->(e2:Entity)<-[:HAS_ENTITY]-(c:Chunk)\n"
            "RETURN c.text, c.chunk_id AS chunk_id, c.page AS page\n"
            "LIMIT 15"
        )
    },
    {
        "description": "Chunk search with citation data",
        "user_query": "Find text chunks about machine learning",
        "cypher": (
            "MATCH (c:Chunk) WHERE toLower(c.text) CONTAINS 'machine learning'\n"
            "RETURN c.text, c.chunk_id AS chunk_id, c.page AS page\n"
            "LIMIT 10"
        )
    },
    {
        "description": "Chunks for a specific entity type or name",
        "user_query": "Show chunks that mention regulations",
        "cypher": (
            "MATCH (c:Chunk)-[:HAS_ENTITY]->(e:Entity)\n"
            "WHERE toLower(e.name) CONTAINS 'regulation' OR toLower(e.type) = 'regulation'\n"
            "RETURN c.text, c.chunk_id AS chunk_id, c.page AS page\n"
            "LIMIT 25"
        )
    },
    {
        "description": "Recall-first: phrase + stems with 0..1 hop expansion and UNION",
        "user_query": "Explain environmental monitoring procedures",
        "cypher": (
            "CALL {\n"
            "  MATCH (c:Chunk)-[:HAS_ENTITY]->(e:Entity)\n"
            "  WHERE toLower(c.text) CONTAINS 'environmental monitoring'\n"
            "     OR toLower(c.text) CONTAINS 'environmental'\n"
            "     OR toLower(c.text) CONTAINS 'monitor'\n"
            "     OR toLower(c.text) CONTAINS 'procedur'\n"
            "     OR toLower(e.name)  CONTAINS 'environmental monitoring'\n"
            "     OR toLower(e.name)  CONTAINS 'environmental'\n"
            "     OR toLower(e.name)  CONTAINS 'monitor'\n"
            "     OR toLower(e.name)  CONTAINS 'procedur'\n"
            "  RETURN DISTINCT c.text, c.chunk_id AS chunk_id, c.page AS page\n"
            "  UNION\n"
            "  MATCH (c1:Chunk)-[:HAS_ENTITY]->(e:Entity)\n"
            "  WHERE toLower(c1.text) CONTAINS 'environmental monitoring'\n"
            "     OR toLower(c1.text) CONTAINS 'environmental'\n"
            "     OR toLower(c1.text) CONTAINS 'monitor'\n"
            "     OR toLower(c1.text) CONTAINS 'procedur'\n"
            "     OR toLower(e.name)  CONTAINS 'environmental monitoring'\n"
            "     OR toLower(e.name)  CONTAINS 'environmental'\n"
            "     OR toLower(e.name)  CONTAINS 'monitor'\n"
            "     OR toLower(e.name)  CONTAINS 'procedur'\n"
            "  MATCH (e)-[:RELATED_TO*0..1]->(e2:Entity)\n"
            "  MATCH (c2:Chunk)-[:HAS_ENTITY]->(:Entity)\n"
            "  WHERE toLower(c2.text) CONTAINS toLower(e.name)\n"
            "     OR toLower(c2.text) CONTAINS toLower(e2.name)\n"
            "  RETURN DISTINCT c2.text AS `c.text`, c2.chunk_id AS chunk_id, c2.page AS page\n"
            "}\n"
            "RETURN `c.text`, chunk_id, page\n"
            "LIMIT 25"
        )
    }
]

# =========================================================================
# System Prompts
# =========================================================================

SYSTEM_PROMPT_TEMPLATE = """You are an expert Neo4j Cypher query generator that converts natural language questions into precise, read-only Cypher queries.

STRICT REQUIREMENTS:
1. Generate ONLY read-only queries using MATCH, RETURN, WHERE, ORDER BY, WITH
2. NEVER use: CREATE, MERGE, DELETE, SET, REMOVE, CALL, LOAD CSV, UNWIND (with writes)
3. ALWAYS include LIMIT clause with maximum {max_results} results
4. ALWAYS include explicit RETURN clause (no wildcards like RETURN *)
5. Use ONLY the exact labels, relationship types, and properties provided in the schema
6. Ensure relationship patterns respect maximum hop limit of {max_hops}

CITATION REQUIREMENTS (MANDATORY):
- When returning Chunk rows you MUST return: c.text, c.chunk_id AS chunk_id, c.page AS page
- Acceptable page variants: page, page_number, page_num (all map to citations)
- Examples:
  - MATCH (c:Chunk) WHERE toLower(c.text) CONTAINS 'keyword' RETURN c.text, c.chunk_id AS chunk_id, c.page AS page LIMIT 10
  - MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk) RETURN c.text, c.chunk_id AS chunk_id, c.page AS page LIMIT 15

RECALL-FIRST MATCHING (IMPORTANT):
- Do NOT rely only on multi-word phrases (e.g., 'monitoring procedures').
- ALWAYS include single-word stem variants to improve recall (e.g., 'monitor', 'procedur').
- Apply both phrase and stem terms to c.text and to e.name when using (c:Chunk)-[:HAS_ENTITY]->(e:Entity).
- Prefer broad stems when the domain term has common morphology (monitor/monitoring, procedure/procedures → 'procedur').

ENTITY-FIRST QUERY PLANNING (MANDATORY):
- Intent: Elevate Chunk nodes as first-class signals in retrieval and query planning.
- Use the cached structural schema summary to understand node types, key properties (especially on Chunk), and relationships. Use ONLY schema elements provided.
- Entities live in Chunks, and every Chunk has Entities.
- Documents are secondary; the citation is in Chunk identifiers. Citations MUST use Chunk.chunk_id and Chunk.page.
- Chunks are the primary pivot for Cypher. Combine c.chunk_id, c.page, c.text with Chunk↔Entity edges to:
  - find Entities from Chunks,
  - find Chunks from Entities, and
  - (optionally) traverse Entity↔Entity relationships to discover additional relevant nodes.
- Querying principle: Start at Chunk, fan out to Entity, optionally expand across Entity↔Entity relations (respecting {max_hops}), then return back to Chunk for citation-grade results.

EXAMPLE PATTERN (adapt labels to the provided schema; respect {max_hops}):
```cypher
// From prompt-derived terms -> Entities -> related Entities
MATCH (c:Chunk)-[:HAS_ENTITY]->(e:Entity)
WHERE toLower(c.text) CONTAINS $term OR toLower(e.name) CONTAINS $term
WITH DISTINCT e, c
// Use non-optional RELATED_TO expansion when you choose to expand
MATCH (e)-[:RELATED_TO*0..{max_hops}]->(e2:Entity)
WITH DISTINCT c
RETURN c.text, c.chunk_id AS chunk_id, c.page AS page
LIMIT {max_results}
```

EXPANSION & UNION PATTERN (RECOMMENDED FOR RECALL):
- It is often effective to combine two branches with a UNION and apply a single LIMIT at the end:
  - Branch A: direct matches by c.text and e.name using both phrase and stem terms
  - Branch B: 0..1 hop expansion via RELATED_TO, then match chunks again by e.name/e2.name
- Use DISTINCT to avoid duplicates; place the final LIMIT after the UNION to cap total rows.

SCHEMA ADHERENCE:
- Only use node labels that exist in the provided schema
- Only use relationship types that exist in the provided schema
- Only use property names that exist in the provided schema
- If uncertain about schema elements, prefer simpler queries

OUTPUT FORMAT:
- Return ONLY the Cypher query inside ```cypher``` code fences
- No explanations, comments, or additional text
- Ensure query syntax is valid and executable"""

# =========================================================================
# User Prompt Templates  
# =========================================================================

USER_PROMPT_TEMPLATE = """Convert this natural language question into a Cypher query:

QUESTION: {user_query}

KNOWLEDGE GRAPH SCHEMA (use only these elements):
{schema_summary}

CONSTRAINTS:
- Maximum results: {max_results}
- Maximum relationship hops: {max_hops}
- Start at Chunk, pivot to Entity, optionally expand Entity↔Entity up to {max_hops}, then return Chunk rows with citation fields (c.text, c.chunk_id AS chunk_id, c.page AS page).
- Use ONLY labels/relationships/properties from schema above.

{examples_section}

Generate the Cypher query:"""

EXAMPLES_SECTION_TEMPLATE = """EXAMPLES OF VALID PATTERNS:
{examples}

"""

# ____________________________________________________________________________
# Class Definitions
#
# ------------------------------------------------------------------------- class PromptComponents
@dataclass
class PromptComponents:
    """Container for prompt building components."""
    system_prompt: str
    user_prompt: str
    total_estimated_tokens: int
    schema_elements_used: Dict[str, int]  # counts of labels, relationships, properties
    truncation_applied: bool
# ------------------------------------------------------------------------- end class PromptComponents

# ------------------------------------------------------------------------- class SchemaSubset
@dataclass 
class SchemaSubset:
    """Subset of schema elements for token-efficient prompting."""
    node_labels: List[Dict[str, Any]]
    relationship_types: List[Dict[str, Any]]
    node_properties: List[Dict[str, Any]]
    relationship_properties: List[Dict[str, Any]]
    estimated_tokens: int
# ------------------------------------------------------------------------- end class SchemaSubset

# ------------------------------------------------------------------------- class StructuralCypherPromptBuilder
class StructuralCypherPromptBuilder:
    """
    Token-aware prompt builder for LLM structural Cypher generation.
    
    Features:
    - Token budget management with intelligent truncation
    - Query-relevant schema subset selection
    - Configurable few-shot examples
    - Schema element prioritization based on query keywords
    """
    
    # ______________________
    # Constructor 
    # 
    # --------------------------------------------------------------------------------- __init__()
    def __init__(
        self,
        token_budget: int = 3500,
        examples_enabled: bool = True,
        max_examples: int = 3
    ):
        """Initialize prompt builder with token budget management.

        Args:
            token_budget: Maximum tokens allowed for entire prompt.
            examples_enabled: Whether to include few-shot examples.
            max_examples: Maximum number of examples to include.
        """
        self.token_budget = token_budget
        self.examples_enabled = examples_enabled
        self.max_examples = max_examples
        self.logger = logging.getLogger(__name__)
        
        # Token estimation factors (rough approximations)
        self.tokens_per_char = 0.25  # Conservative estimate for GPT tokenization
        self.system_prompt_base_tokens = 150  # Base system prompt tokens
        self.user_prompt_base_tokens = 50    # Base user prompt tokens
        self.example_tokens_each = 60        # Tokens per example
    # --------------------------------------------------------------------------------- end __init__()
        
    # =========================================================================
    # Public API: Prompt Building
    # =========================================================================
    # -------------------------------------------------------------- build_prompt()
    def build_prompt(
        self,
        user_query: str,
        structural_summary: Dict[str, Any],
        max_results: int = 50,
        max_hops: int = 3
    ) -> PromptComponents:
        """
        Build complete prompt with token budget management.
        
        Args:
            user_query: User's natural language query
            structural_summary: Complete structural schema summary
            max_results: Maximum results constraint
            max_hops: Maximum relationship hops constraint
            
        Returns:
            PromptComponents with system and user prompts
        """
        try:
            # Extract query keywords for schema relevance scoring
            query_keywords = self._extract_query_keywords(user_query)
            
            # Calculate available tokens for schema content
            available_tokens = self._calculate_available_tokens(user_query)
            
            # Select relevant schema subset within token budget
            schema_subset = self._select_relevant_schema_subset(
                structural_summary,
                query_keywords,
                available_tokens
            )
            
            # Build system prompt
            system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
                max_results=max_results,
                max_hops=max_hops
            )
            
            # Build schema summary text
            schema_summary_text = self._format_schema_subset(schema_subset)
            
            # Build examples section if enabled
            examples_section = ""
            if self.examples_enabled:
                examples_section = self._build_examples_section(query_keywords)
            
            # Build user prompt
            user_prompt = USER_PROMPT_TEMPLATE.format(
                user_query=user_query,
                schema_summary=schema_summary_text,
                max_results=max_results,
                max_hops=max_hops,
                examples_section=examples_section
            )
            
            # Calculate total tokens
            total_tokens = self._estimate_total_tokens(system_prompt, user_prompt)
            
            # Track schema elements used
            schema_elements_used = {
                "node_labels": len(schema_subset.node_labels),
                "relationship_types": len(schema_subset.relationship_types),
                "node_properties": len(schema_subset.node_properties),
                "relationship_properties": len(schema_subset.relationship_properties)
            }
            
            self.logger.info(f"Prompt built: {total_tokens} tokens, "
                           f"schema elements: {sum(schema_elements_used.values())}")
            
            return PromptComponents(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                total_estimated_tokens=total_tokens,
                schema_elements_used=schema_elements_used,
                truncation_applied=schema_subset.estimated_tokens < self._estimate_full_schema_tokens(structural_summary)
            )
            
        except Exception as e:
            self.logger.error(f"Error building prompt: {e}")
            # Fallback to minimal prompt
            return self._build_fallback_prompt(user_query, max_results, max_hops)
    # -------------------------------------------------------------- end build_prompt()
 
    # ______________________
    # Internal/Private Methods (single leading underscore convention)
    #
    # =========================================================================
    # Query Analysis Utilities
    # =========================================================================
    # -------------------------------------------------------------- _extract_query_keywords()
    def _extract_query_keywords(self, user_query: str) -> List[str]:
            """Extract relevant keywords from user query for schema matching."""
            # Simple keyword extraction - lowercase, remove common words
            common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'what', 'how', 'where', 'when', 'why', 'who', 'find', 'show', 'get', 'list'}
            words = re.findall(r'\b\w+\b', user_query.lower())
            keywords = [w for w in words if len(w) > 2 and w not in common_words]
            return keywords[:10]  # Limit to top 10 keywords
    # -------------------------------------------------------------- end _extract_query_keywords()
        
    # =========================================================================
    # Token Budgeting and Estimation
    # =========================================================================
    # -------------------------------------------------------------- _calculate_available_tokens()
    def _calculate_available_tokens(self, user_query: str) -> int:
            """Calculate tokens available for schema content."""
            base_tokens = (
                self.system_prompt_base_tokens +
                self.user_prompt_base_tokens +
                int(len(user_query) * self.tokens_per_char)
            )
            
            if self.examples_enabled:
                base_tokens += min(self.max_examples, len(FEW_SHOT_EXAMPLES)) * self.example_tokens_each
                
            available = self.token_budget - base_tokens
            return max(available, 200)  # Minimum 200 tokens for schema
    # -------------------------------------------------------------- end _calculate_available_tokens()
        
    # =========================================================================
    # Schema Selection and Scoring
    # =========================================================================
    # -------------------------------------------------------------- _select_relevant_schema_subset()
    def _select_relevant_schema_subset(
            self,
            structural_summary: Dict[str, Any],
            query_keywords: List[str],
            available_tokens: int
        ) -> SchemaSubset:
            """Select relevant schema elements within token budget."""
            
            # Score and sort schema elements by relevance
            node_labels = self._score_and_sort_elements(
                structural_summary.get("node_labels", []),
                query_keywords,
                lambda x: x.get("label", "")
            )
            
            relationship_types = self._score_and_sort_elements(
                structural_summary.get("relationship_types", []),
                query_keywords,
                lambda x: x.get("type", "")
            )
            
            node_properties = self._score_and_sort_elements(
                structural_summary.get("node_property_types", []),
                query_keywords,
                lambda x: x.get("property", "")
            )
            
            relationship_properties = self._score_and_sort_elements(
                structural_summary.get("relationship_property_types", []),
                query_keywords, 
                lambda x: x.get("property", "")
            )
            
            # Select elements within token budget
            selected_labels = []
            selected_rels = []
            selected_node_props = []
            selected_rel_props = []
            current_tokens = 0
            
            # Prioritize labels and relationships first
            for label in node_labels:
                label_tokens = self._estimate_element_tokens(label)
                if current_tokens + label_tokens <= available_tokens * 0.4:  # 40% for labels
                    selected_labels.append(label)
                    current_tokens += label_tokens
            
            for rel in relationship_types:
                rel_tokens = self._estimate_element_tokens(rel)
                if current_tokens + rel_tokens <= available_tokens * 0.7:  # 70% for labels+rels
                    selected_rels.append(rel)
                    current_tokens += rel_tokens
            
            # Add properties with remaining budget
            remaining_tokens = available_tokens - current_tokens
            for prop in node_properties[:20]:  # Limit to top 20 most relevant
                prop_tokens = self._estimate_element_tokens(prop)
                if current_tokens + prop_tokens <= available_tokens:
                    selected_node_props.append(prop)
                    current_tokens += prop_tokens
            
            for prop in relationship_properties[:10]:  # Limit to top 10
                prop_tokens = self._estimate_element_tokens(prop)
                if current_tokens + prop_tokens <= available_tokens:
                    selected_rel_props.append(prop)
                    current_tokens += prop_tokens
            
            return SchemaSubset(
                node_labels=selected_labels,
                relationship_types=selected_rels,
                node_properties=selected_node_props,
                relationship_properties=selected_rel_props,
                estimated_tokens=current_tokens
            )
    # -------------------------------------------------------------- end _select_relevant_schema_subset()
        
    # -------------------------------------------------------------- _score_and_sort_elements()
    def _score_and_sort_elements(
            self,
            elements: List[Dict[str, Any]],
            keywords: List[str],
            name_extractor: Callable[[Dict[str, Any]], str]
        ) -> List[Dict[str, Any]]:
            """Score schema elements by relevance to query keywords."""
            scored_elements = []
            
            for element in elements:
                name = safe_lower(name_extractor(element))
                score = sum(1 for keyword in keywords if keyword in name)
                # Add base score to ensure some elements are always included
                score += 0.1
                scored_elements.append((score, element))
            
            # Sort by score descending
            scored_elements.sort(key=lambda x: x[0], reverse=True)
            return [element for score, element in scored_elements]
    # -------------------------------------------------------------- end _score_and_sort_elements()
        
    # -------------------------------------------------------------- _estimate_element_tokens()
    def _estimate_element_tokens(self, element: Dict[str, Any]) -> int:
            """Estimate tokens for a single schema element."""
            element_str = str(element)
            return int(len(element_str) * self.tokens_per_char) + 2  # +2 for formatting
    # -------------------------------------------------------------- end _estimate_element_tokens()
        
    # -------------------------------------------------------------- _estimate_full_schema_tokens()
    def _estimate_full_schema_tokens(self, structural_summary: Dict[str, Any]) -> int:
            """Estimate tokens for complete schema summary."""
            full_str = str(structural_summary)
            return int(len(full_str) * self.tokens_per_char)
    # -------------------------------------------------------------- end _estimate_full_schema_tokens()
        
    # =========================================================================
    # Formatting Helpers
    # =========================================================================
    # -------------------------------------------------------------- _format_schema_subset()
    def _format_schema_subset(self, schema_subset: SchemaSubset) -> str:
            """Format schema subset into readable text for prompt."""
            lines = []
            
            # Node Labels
            if schema_subset.node_labels:
                lines.append("NODE LABELS:")
                for label_info in schema_subset.node_labels:
                    label = label_info.get("label", "Unknown")
                    lines.append(f"  - {label}")
            
            # Relationship Types
            if schema_subset.relationship_types:
                lines.append("\nRELATIONSHIP TYPES:")
                for rel_info in schema_subset.relationship_types:
                    rel_type = rel_info.get("type", "Unknown")
                    lines.append(f"  - {rel_type}")
            
            # Node Properties (top most relevant)
            if schema_subset.node_properties:
                lines.append("\nNODE PROPERTIES (key examples):")
                for prop_info in schema_subset.node_properties[:10]:
                    prop_name = prop_info.get("property", "Unknown")
                    labels = prop_info.get("labels", []) or []  # Handle None labels
                    # Filter out None values and ensure strings
                    safe_labels = [str(label) for label in labels[:3] if label is not None]
                    label_list = ", ".join(safe_labels)
                    lines.append(f"  - {prop_name} (on: {label_list})")
            
            # Relationship Properties
            if schema_subset.relationship_properties:
                lines.append("\nRELATIONSHIP PROPERTIES:")
                for prop_info in schema_subset.relationship_properties[:5]:
                    prop_name = prop_info.get("property", "Unknown")
                    lines.append(f"  - {prop_name}")
            
            return "\n".join(lines)
    # -------------------------------------------------------------- end _format_schema_subset()
        
    # =========================================================================
    # Examples Section Builder
    # =========================================================================
    # -------------------------------------------------------------- _build_examples_section()
    def _build_examples_section(self, query_keywords: List[str]) -> str:
            """Build examples section with most relevant few-shot examples."""
            if not self.examples_enabled:
                return ""
            
            # Select most relevant examples based on keywords
            relevant_examples = []
            for example in FEW_SHOT_EXAMPLES:
                relevance = sum(1 for keyword in query_keywords 
                            if keyword in safe_lower(example.get("user_query", "")) or 
                                keyword in safe_lower(example.get("cypher", "")))
                relevant_examples.append((relevance, example))
            
            # Sort by relevance and take top N
            relevant_examples.sort(key=lambda x: x[0], reverse=True)
            selected_examples = [ex for _, ex in relevant_examples[:self.max_examples]]
            
            if not selected_examples:
                selected_examples = FEW_SHOT_EXAMPLES[:self.max_examples]
            
            # Format examples
            example_lines = []
            for i, example in enumerate(selected_examples, 1):
                example_lines.append(f"{i}. Query: \"{example['user_query']}\"")
                example_lines.append(f"   Cypher: {example['cypher']}")
                example_lines.append("")
            
            return EXAMPLES_SECTION_TEMPLATE.format(examples="\n".join(example_lines))
    # -------------------------------------------------------------- end _build_examples_section()
        
    # -------------------------------------------------------------- _estimate_total_tokens()
    def _estimate_total_tokens(self, system_prompt: str, user_prompt: str) -> int:
            """Estimate total tokens for complete prompt."""
            total_chars = len(system_prompt) + len(user_prompt)
            return int(total_chars * self.tokens_per_char)
    # -------------------------------------------------------------- end _estimate_total_tokens()
        
    # =========================================================================
    # Fallback Prompt Construction
    # =========================================================================
    # -------------------------------------------------------------- _build_fallback_prompt()
    def _build_fallback_prompt(self, user_query: str, max_results: int, max_hops: int) -> PromptComponents:
            """Build minimal fallback prompt when normal building fails."""
            system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
                max_results=max_results,
                max_hops=max_hops
            )
            
            user_prompt = f"""Convert this question into a Cypher query:
    QUESTION: {user_query}

    Use standard graph patterns and include LIMIT {max_results}.
    Generate the Cypher query:"""
            
            return PromptComponents(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                total_estimated_tokens=self._estimate_total_tokens(system_prompt, user_prompt),
                schema_elements_used={"fallback": True},
                truncation_applied=True
            )
    # -------------------------------------------------------------- end _build_fallback_prompt()
    
# ------------------------------------------------------------------------- end class StructuralCypherPromptBuilder

# =========================================================================
# Narrative Summarization Prompt System
# =========================================================================

NARRATIVE_SYSTEM_PROMPT = """You are an expert knowledge synthesizer that composes authoritative answers from pre-numbered sources.

CRITICAL CITATION REQUIREMENTS:
- PRESERVE the given source numbers [n] EXACTLY as provided - NEVER invent or renumber
- Place [n] immediately after the statements they support
- Each [n] must correspond to a real source provided in the context
- If unsure about a citation, omit it rather than guess
- Do NOT add a References section - this will be added programmatically

DOMAIN-SPECIFIC CITATION ENHANCEMENT:
When sources contain recognizable domain identifiers, include them inline with citations:

• Legal/Regulatory: Include specific sections/parts next to [n]
  - Example: "Emergency evacuation procedures are required under §75.1502 [3] and refuge alternatives must meet §75.1505 [7]"
  - Look for: §XX.XXXX, Part XX.XXXX, Title XX CFR

• Academic/Scholarly: Include structural references next to [n]
  - Example: "As shown in Figure 4.1 [2], the correlation is significant according to Vol. 2, Chapter 3 [1]"
  - Look for: Vol. X, Chapter Y, Figure X.X, Table X, Section X.X

• Technical/Standards: Include standard identifiers next to [n]
  - Example: "The protocol follows ISO 9001:2015 [1] and complies with ASTM D2582-07 [2]"
  - Look for: ISO XXXX:YYYY, ASTM XXXXX-XX, IEEE XXX

• Business/Policy: Include policy IDs/procedure codes next to [n]
  - Example: "According to Policy 2.4 [1], the procedure outlined in SOP-17 [2] must be followed"
  - Look for: Policy X.X, Procedure XXX, SOP-XX, Guidelines X.X

• Medical/Clinical: Include protocol/guideline names next to [n]  
  - Example: "Treatment follows Protocol 4.2 [1] and NICE NG12 guidelines [3]"
  - Look for: Protocol X.X, NICE XXXX, Clinical Guidelines X.X, ICD-XX

FORBIDDEN ACTIONS:
- Do not create new citation numbers beyond those provided
- Do not add References, Bibliography, or Sources sections
- Do not renumber existing citations
- Do not include meta-commentary about sources or citations

COMPOSITION GUIDELINES:
1. Write natural, flowing prose that directly answers the user's query
2. Integrate information from multiple sources seamlessly
3. Use domain-specific terminology appropriately
4. Maintain professional, authoritative tone
5. Place citations [n] immediately after specific claims they support
6. Include domain identifiers when present in sources, positioned naturally with [n]"""

NARRATIVE_USER_PROMPT_TEMPLATE = """Query: {user_query}

Sources (pre-numbered for citation):
{numbered_sources}

{schema_section}

Compose a comprehensive answer using the provided sources. Use inline citations [n] exactly as numbered above."""

# ------------------------------------------------------------------------- class NarrativePromptComponents
@dataclass
class NarrativePromptComponents:
    """Components for narrative summarization prompt."""
    system_prompt: str
    user_prompt: str
    total_estimated_tokens: int
    schema_elements_used: int
    sources_included: int
    truncation_applied: bool

# ------------------------------------------------------------------------- class StructuralNarrativePromptBuilder
class StructuralNarrativePromptBuilder:
    """Builds prompts for LLM narrative composition from numbered sources."""
    
    # ______________________
    # Constructor 
    # 
    # --------------------------------------------------------------------------------- __init__()
    def __init__(
        self,
        token_budget: int = 3500,
        preserve_numbers: bool = True
    ):
        """
        Initialize narrative prompt builder.
        
        Args:
            token_budget: Maximum tokens for prompt (input only)
            preserve_numbers: Require LLM to preserve provided numbers
        """
        self.token_budget = token_budget
        self.preserve_numbers = preserve_numbers
        self.logger = logging.getLogger(__name__)
        
        # Token estimation factors
        self.tokens_per_char = 0.25
        self.system_prompt_tokens = int(len(NARRATIVE_SYSTEM_PROMPT) * self.tokens_per_char)
        self.base_user_tokens = 100  # Base user prompt tokens
    # --------------------------------------------------------------------------------- end __init__()
    
    # -------------------------------------------------------------- build_narrative_prompt()
    def build_narrative_prompt(
        self,
        user_query: str,
        numbered_sources: List[Dict[str, Any]],
        schema_text: Optional[str] = None
    ) -> NarrativePromptComponents:
        """Build complete narrative prompt with token management."""
        
        # Format sources (no excerpt truncation - use source selection instead)
        sources_text = self._format_numbered_sources(numbered_sources)
        
        # Apply token budget through schema truncation or source selection
        schema_section = ""
        if schema_text:
            schema_section = self._fit_schema_to_budget(
                schema_text, sources_text, user_query
            )
        
        user_prompt = NARRATIVE_USER_PROMPT_TEMPLATE.format(
            user_query=user_query,
            numbered_sources=sources_text,
            schema_section=schema_section
        )
        
        total_tokens = self._estimate_tokens(NARRATIVE_SYSTEM_PROMPT + user_prompt)
        
        return NarrativePromptComponents(
            system_prompt=NARRATIVE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            total_estimated_tokens=total_tokens,
            schema_elements_used=len(schema_text.split()) if schema_text else 0,
            sources_included=len(numbered_sources),
            truncation_applied=len(schema_section) < len(schema_text or "")
        )
    # -------------------------------------------------------------- end build_narrative_prompt()
    
    # ______________________
    # Internal/Private Methods (single leading underscore convention)
    #
    # -------------------------------------------------------------- _format_numbered_sources()
    def _format_numbered_sources(self, numbered_sources: List[Dict[str, Any]]) -> str:
        """Format numbered sources for prompt with domain markers (no truncation of excerpts)."""
        source_lines = []
        
        for source in numbered_sources:
            n = source['n']
            chunk_id = source['chunk_id']
            page = source['page']
            excerpt = source['excerpt']  # Full excerpt, no truncation
            domain_markers = source.get('domain_markers', {})
            
            # Start with basic citation info
            source_lines.append(f"[{n}] {chunk_id}, p.{page}")
            
            # Add domain markers if present
            if domain_markers:
                marker_sections = []
                for domain, markers in domain_markers.items():
                    if markers:  # Only add if markers exist for this domain
                        domain_name = self._format_domain_name(domain)
                        markers_str = ", ".join(markers)
                        marker_sections.append(f"{domain_name} markers: {markers_str}")
                
                if marker_sections:
                    source_lines.append("\n".join(marker_sections))
            
            # Add excerpt content
            source_lines.append(f"Excerpt: {excerpt}")
            source_lines.append("")  # Empty line between sources
        
        return "\n".join(source_lines)
    # -------------------------------------------------------------- end _format_numbered_sources()
    
    # -------------------------------------------------------------- _format_domain_name()
    def _format_domain_name(self, domain: str) -> str:
        """Format domain name for display in prompts."""
        domain_names = {
            'legal': 'Legal/Regulatory',
            'academic': 'Academic',
            'technical': 'Technical/Standards',
            'business': 'Business/Policy',
            'medical': 'Medical/Clinical'
        }
        return domain_names.get(domain, domain.capitalize())
    # -------------------------------------------------------------- end _format_domain_name()
    
    # -------------------------------------------------------------- _fit_schema_to_budget()
    def _fit_schema_to_budget(
        self, 
        schema_text: str, 
        sources_text: str, 
        user_query: str
    ) -> str:
        """Fit schema text within token budget through intelligent truncation."""
        # Calculate used tokens
        user_query_tokens = int(len(user_query) * self.tokens_per_char)
        sources_tokens = int(len(sources_text) * self.tokens_per_char)
        base_tokens = self.system_prompt_tokens + self.base_user_tokens + user_query_tokens + sources_tokens
        
        # Available tokens for schema
        available_schema_tokens = max(self.token_budget - base_tokens, 100)  # Minimum 100 for schema
        available_chars = int(available_schema_tokens / self.tokens_per_char)
        
        if len(schema_text) <= available_chars:
            return f"\n\nSchema Context (domain terminology):\n{schema_text}"
        
        # Truncate schema intelligently - keep most relevant parts
        truncated_schema = self._truncate_schema_intelligently(schema_text, available_chars)
        return f"\n\nSchema Context (domain terminology):\n{truncated_schema}..."
    # -------------------------------------------------------------- end _fit_schema_to_budget()
    
    # -------------------------------------------------------------- _truncate_schema_intelligently()
    def _truncate_schema_intelligently(self, schema_text: str, max_chars: int) -> str:
        """Truncate schema keeping most important elements."""
        # Priority order: node labels, relationship types, then properties
        lines = schema_text.split('\n')
        important_lines = []
        char_count = 0
        
        # First pass: collect node labels and relationship types
        for line in lines:
            if ('NODE LABELS:' in line or 
                'RELATIONSHIP TYPES:' in line or 
                (line.strip().startswith('-') and 
                 ('PROPERTIES' not in line or char_count < max_chars * 0.7))):
                if char_count + len(line) <= max_chars:
                    important_lines.append(line)
                    char_count += len(line) + 1  # +1 for newline
                else:
                    break
        
        return '\n'.join(important_lines)
    # -------------------------------------------------------------- end _truncate_schema_intelligently()
    
    # -------------------------------------------------------------- _estimate_tokens()
    def _estimate_tokens(self, text: str) -> int:
        """Estimate tokens for text."""
        return int(len(text) * self.tokens_per_char)
    # -------------------------------------------------------------- end _estimate_tokens()

# ------------------------------------------------------------------------- end class StructuralNarrativePromptBuilder

# __________________________________________________________________________
# Standalone Function Definitions
#
# =========================================================================
# Factory Functions
# =========================================================================

# --------------------------------------------------------------------------------- get_structural_cypher_prompt_builder()
def get_structural_cypher_prompt_builder(
    token_budget: int = 3500,
    examples_enabled: bool = True
) -> StructuralCypherPromptBuilder:
    """Create a configured StructuralCypherPromptBuilder.

    This factory applies sensible defaults used across the backend tools module. It
    centralizes construction so callers don't need to know about token budgets or
    example toggles when generating Cypher prompts for the LLM structural path.

    Args:
        token_budget: Maximum tokens to allocate for the combined system+user prompt.
        examples_enabled: Include a compact few‑shot examples section when True.

    Returns:
        An initialized `StructuralCypherPromptBuilder` ready to build prompts.

    Examples:
        >>> builder = get_structural_cypher_prompt_builder(token_budget=3000)
        >>> isinstance(builder, StructuralCypherPromptBuilder)
        True
    """
    return StructuralCypherPromptBuilder(
        token_budget=token_budget,
        examples_enabled=examples_enabled
    )
# --------------------------------------------------------------------------------- end get_structural_cypher_prompt_builder()

# --------------------------------------------------------------------------------- get_structural_narrative_prompt_builder()
def get_structural_narrative_prompt_builder(
    token_budget: int = 3500,
    preserve_numbers: bool = True
) -> StructuralNarrativePromptBuilder:
    """Create a configured StructuralNarrativePromptBuilder.

    This factory mirrors the Cypher builder factory but for narrative composition.
    It enforces consistent token budgeting and citation‑number preservation across
    backend usage sites.

    Args:
        token_budget: Maximum tokens to allocate for the narrative prompt (input only).
        preserve_numbers: Instruct the LLM to keep citation numbers exactly as provided.

    Returns:
        An initialized `StructuralNarrativePromptBuilder` for narrative prompts.

    Examples:
        >>> nb = get_structural_narrative_prompt_builder(preserve_numbers=True)
        >>> isinstance(nb, StructuralNarrativePromptBuilder)
        True
    """
    return StructuralNarrativePromptBuilder(
        token_budget=token_budget,
        preserve_numbers=preserve_numbers
    )
# --------------------------------------------------------------------------------- end get_structural_narrative_prompt_builder()

# __________________________________________________________________________
# End of File
#
