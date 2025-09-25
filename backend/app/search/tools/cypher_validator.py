# -------------------------------------------------------------------------
# File: cypher_validator.py
# Author: Alexander Ricciardi
# Date: 2025-09-15
# [File Path] backend/app/search/tools/cypher_validator.py
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
#   Enhanced Cypher validation framework used by the LLM Structural Cypher engine. Performs
#   safety checks (no writes, forbidden calls), structural checks (RETURN, LIMIT, hop caps),
#   schema validation, and conservative smart rewrites with actionable reporting.
# -------------------------------------------------------------------------

# --- Module Contents Overview ---
# - Enum: ValidationSeverity
# - Dataclass: ValidationIssue
# - Dataclass: FixApplied
# - Dataclass: ValidationReport
# - Dataclass: ParsedCypher
# - Class: CypherParser (query parsing utilities)
# - Class: CypherValidator (validation workflow and fixes)
# - Function: get_cypher_validator (factory)
# -------------------------------------------------------------------------

# --- Dependencies / Imports ---
# - Standard Library: re, logging, enum, dataclasses, typing
# - Local Project Modules: schema_models.CompleteKGSchema
# --- Requirements ---
# - Python 3.12
# -------------------------------------------------------------------------

# --- Usage / Integration ---
# - The structural engine composes Cypher then calls CypherValidator.validate() to enforce safety
#   and structure, optionally applying conservative rewrites before execution via the schema layer.

# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------

"""
Enhanced Cypher Validation Framework for APH-IF LLM Structural Cypher Generator.

Provides comprehensive validation with safety checks, schema validation against a cached
`CompleteKGSchema`, conservative smart rewrites, and actionable error reporting used by the
LLM Structural Cypher engine.
"""

from __future__ import annotations

# __________________________________________________________________________
# Imports

import re
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ...schema.schema_models import CompleteKGSchema

logger = logging.getLogger(__name__)

# __________________________________________________________________________
# Global Constants / Variables

# =========================================================================
# Validation Models
# =========================================================================

# ------------------------------------------------------------------------- class ValidationSeverity
@dataclass
# ____________________________________________________________________________
# Class Definitions

# ------------------------------------------------------------------------- class ValidationSeverity
class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    ERROR = "error"          # Blocking issues that prevent execution
    WARNING = "warning"      # Issues that should be fixed but don't block
    INFO = "info"           # Informational notices
# ------------------------------------------------------------------------- end class ValidationSeverity

# ------------------------------------------------------------------------- class ValidationIssue
@dataclass
class ValidationIssue:
    """Individual validation issue with details.

    Attributes:
        severity: Severity level indicating error, warning, or info.
        code: Machine-friendly issue code.
        message: Human-readable description of the issue.
        element: Optional problematic element (label, property, etc.).
        suggestion: Optional suggested fix that may resolve the issue.
        line_number: Optional line number in the query where the issue occurs.
    """
    severity: ValidationSeverity
    code: str
    message: str
    element: Optional[str] = None  # The problematic element (label, property, etc.)
    suggestion: Optional[str] = None  # Suggested fix
    line_number: Optional[int] = None  # Line in query where issue occurs
# ------------------------------------------------------------------------- class ValidationIssue

# ------------------------------------------------------------------------- class FixApplied
@dataclass
class FixApplied:
    """Details about a fix that was automatically applied.

    Attributes:
        description: Short description of the fix applied.
        original_element: The original problematic element.
        fixed_element: The element after applying the fix.
        fix_type: Type of fix (e.g., "injection", "removal", "replacement", "wrapping").
    """
    description: str
    original_element: str
    fixed_element: str
    fix_type: str  # "injection", "removal", "replacement", "wrapping"
# ------------------------------------------------------------------------- end class FixApplied

# ------------------------------------------------------------------------- class ValidationReport
@dataclass
class ValidationReport:
    """Comprehensive validation report with issues and fixes.

    Attributes:
        is_valid: Final validity flag after all checks and fixes.
        original_cypher: The original Cypher query provided for validation.
        safe_cypher: The safe Cypher after applying non-destructive fixes (if any).
        issues: List of collected validation issues.
        fixes_applied: List of applied fixes with details.
        fallback_recommended: Whether to recommend a fallback path.
        schema_elements_found: Summary of elements detected in the query.
    """
    is_valid: bool
    original_cypher: str
    safe_cypher: Optional[str]
    issues: List[ValidationIssue] = field(default_factory=list)
    fixes_applied: List[FixApplied] = field(default_factory=list)
    fallback_recommended: bool = False
    schema_elements_found: Dict[str, List[str]] = field(default_factory=dict)
    
    @property
    def has_errors(self) -> bool:
        """Check if report contains any error-level issues."""
        return any(issue.severity == ValidationSeverity.ERROR for issue in self.issues)
    
    @property
    def has_warnings(self) -> bool:
        """Check if report contains any warning-level issues."""
        return any(issue.severity == ValidationSeverity.WARNING for issue in self.issues)

# ------------------------------------------------------------------------- end class ValidationReport

# =========================================================================
# Cypher Query Parser
# =========================================================================

# ------------------------------------------------------------------------- class ParsedCypher
@dataclass
class ParsedCypher:
    """Parsed Cypher query components.

    Attributes:
        clauses: Ordered list of major clauses detected (e.g., MATCH, RETURN).
        node_labels: Set of node labels referenced in the query.
        relationship_types: Set of relationship types referenced.
        property_keys: Set of property keys referenced.
        has_limit: Whether a LIMIT clause is present.
        has_return: Whether a RETURN clause is present.
        has_explicit_return: Whether RETURN is explicit (not wildcard).
        max_hops: Maximum relationship traversal hops estimated.
        write_operations: Write operations found (e.g., CREATE, MERGE).
        function_calls: Detected function/procedure call patterns.
        contains_unwind: Whether UNWIND appears in the query.
    """
    clauses: List[str]
    node_labels: Set[str]
    relationship_types: Set[str]
    property_keys: Set[str]
    has_limit: bool
    has_return: bool
    has_explicit_return: bool  # Not just "RETURN *"
    max_hops: int
    write_operations: List[str]
    function_calls: List[str]
    contains_unwind: bool
# ------------------------------------------------------------------------- end class ParsedCypher

# ------------------------------------------------------------------------- class CypherParser
class CypherParser:
    """Parse Cypher queries to extract components and patterns."""
    
    # Regex patterns for parsing
    WRITE_OPERATIONS = [
        'CREATE', 'MERGE', 'DELETE', 'REMOVE', 'SET', 
        'DROP', 'ALTER', 'LOAD CSV', 'IMPORT'
    ]
    
    DANGEROUS_FUNCTIONS = [
        'apoc\\.',  # All APOC procedures
        'dbms\\.',  # Database management
        'db\\.',    # Database functions
        'call\\s+\\{',  # Subquery calls that might contain writes
    ]
    
    ALLOWED_FUNCTIONS = [
        'count', 'sum', 'avg', 'max', 'min', 'collect', 'distinct',
        'size', 'length', 'coalesce', 'head', 'tail', 'range',
        'toString', 'toInteger', 'toFloat', 'substring', 'split',
        'trim', 'upper', 'lower', 'replace'
    ]
    
    # ______________________
    # Constructor 
    # 
    # --------------------------------------------------------------------------------- __init__()
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    # --------------------------------------------------------------------------------- end __init__()
    
    # -------------------------------------------------------------- parse()
    def parse(self, cypher: str) -> ParsedCypher:
        """Parse a Cypher query into structured components.

        Args:
            cypher: Raw Cypher text to analyze.

        Returns:
            ParsedCypher: Structured representation of clauses, labels, relationships, properties,
            as well as structural flags useful for downstream validation.
        """
        try:
            # Normalize query
            normalized = self._normalize_cypher(cypher)
            
            # Extract components
            clauses = self._extract_clauses(normalized)
            node_labels = self._extract_node_labels(normalized)
            relationship_types = self._extract_relationship_types(normalized)
            property_keys = self._extract_property_keys(normalized)
            
            # Analyze query structure
            has_limit = self._has_limit_clause(normalized)
            has_return = self._has_return_clause(normalized)
            has_explicit_return = self._has_explicit_return(normalized)
            max_hops = self._calculate_max_hops(normalized)
            write_operations = self._find_write_operations(normalized)
            function_calls = self._find_function_calls(normalized)
            contains_unwind = 'UNWIND' in normalized.upper()
            
            return ParsedCypher(
                clauses=clauses,
                node_labels=node_labels,
                relationship_types=relationship_types,
                property_keys=property_keys,
                has_limit=has_limit,
                has_return=has_return,
                has_explicit_return=has_explicit_return,
                max_hops=max_hops,
                write_operations=write_operations,
                function_calls=function_calls,
                contains_unwind=contains_unwind
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing Cypher: {e}")
            # Return empty parsed result on failure
            return ParsedCypher(
                clauses=[],
                node_labels=set(),
                relationship_types=set(),
                property_keys=set(),
                has_limit=False,
                has_return=False,
                has_explicit_return=False,
                max_hops=0,
                write_operations=[],
                function_calls=[],
                contains_unwind=False
            )
    # -------------------------------------------------------------- end parse()
    
    # ______________________
    # Internal/Private Methods (single leading underscore convention)
    #
    # -------------------------------------------------------------- _normalize_cypher()
    def _normalize_cypher(self, cypher: str) -> str:
        """Normalize Cypher query for parsing."""
        # Remove code fences if present
        cypher = re.sub(r'```cypher\s*\n?', '', cypher, flags=re.IGNORECASE)
        cypher = re.sub(r'```\s*\n?', '', cypher)
        
        # Remove comments
        cypher = re.sub(r'//.*$', '', cypher, flags=re.MULTILINE)
        cypher = re.sub(r'/\*.*?\*/', '', cypher, flags=re.DOTALL)
        
        # Normalize whitespace
        cypher = ' '.join(cypher.split())
        
        return cypher.strip()
    
    # -------------------------------------------------------------- _extract_clauses()
    def _extract_clauses(self, cypher: str) -> List[str]:
        """Extract major Cypher clauses."""
        clause_pattern = r'\b(MATCH|RETURN|WHERE|WITH|ORDER\s+BY|LIMIT|SKIP|CREATE|MERGE|DELETE|SET|REMOVE|CALL|UNWIND|UNION)\b'
        clauses = re.findall(clause_pattern, cypher, re.IGNORECASE)
        return [clause.upper() for clause in clauses]
    
    # -------------------------------------------------------------- _extract_node_labels()
    def _extract_node_labels(self, cypher: str) -> Set[str]:
        """Extract node labels from Cypher query."""
        # Pattern for node labels: (variable:Label) or (:Label)
        label_pattern = r'\(\w*:(\w+)(?:\s|\)|:)'
        labels = re.findall(label_pattern, cypher, re.IGNORECASE)
        return set(labels)
    
    # -------------------------------------------------------------- _extract_relationship_types()
    def _extract_relationship_types(self, cypher: str) -> Set[str]:
        """Extract relationship types from Cypher query."""
        # Pattern for relationship types: -[:REL_TYPE]-> or -[r:REL_TYPE]-
        rel_pattern = r'-\[\w*:(\w+)(?:\s|\]|\|)'
        relationships = re.findall(rel_pattern, cypher, re.IGNORECASE)
        return set(relationships)
    
    # -------------------------------------------------------------- _extract_property_keys()
    def _extract_property_keys(self, cypher: str) -> Set[str]:
        """Extract property keys from Cypher query."""
        # Pattern for properties: variable.property or {property: value}
        prop_patterns = [
            r'\w+\.(\w+)',  # variable.property
            r'\{\s*(\w+)\s*:',  # {property: value}
            r'(\w+)\s*CONTAINS',  # property CONTAINS
            r'(\w+)\s*STARTS\s+WITH',  # property STARTS WITH
            r'(\w+)\s*ENDS\s+WITH',  # property ENDS WITH
        ]
        
        properties = set()
        for pattern in prop_patterns:
            matches = re.findall(pattern, cypher, re.IGNORECASE)
            properties.update(matches)
        
        return properties
    
    # -------------------------------------------------------------- _has_limit_clause()
    def _has_limit_clause(self, cypher: str) -> bool:
        """Check if query has LIMIT clause."""
        return bool(re.search(r'\bLIMIT\s+\d+', cypher, re.IGNORECASE))
    
    # -------------------------------------------------------------- _has_return_clause()
    def _has_return_clause(self, cypher: str) -> bool:
        """Check if query has RETURN clause."""
        return bool(re.search(r'\bRETURN\b', cypher, re.IGNORECASE))
    
    # -------------------------------------------------------------- _has_explicit_return()
    def _has_explicit_return(self, cypher: str) -> bool:
        """Check if query has explicit RETURN (not just RETURN *)."""
        return bool(re.search(r'\bRETURN\s+(?!\*\s*(?:LIMIT|$))', cypher, re.IGNORECASE))
    
    # -------------------------------------------------------------- _calculate_max_hops()
    def _calculate_max_hops(self, cypher: str) -> int:
        """Calculate maximum relationship hops in query."""
        # Look for variable length patterns like [*1..3] or [*..5]
        hop_patterns = [
            r'\[\*(\d+)\.\.(\d+)\]',  # [*1..3]
            r'\[\*\.\.(\d+)\]',       # [*..5]  
            r'\[\*(\d+)\]',           # [*3]
        ]
        
        max_hops = 1  # Default single hop
        
        for pattern in hop_patterns:
            matches = re.findall(pattern, cypher)
            for match in matches:
                if isinstance(match, tuple):
                    # Range pattern [*1..3]
                    if len(match) == 2 and match[1]:
                        max_hops = max(max_hops, int(match[1]))
                    elif len(match) == 1:
                        max_hops = max(max_hops, int(match[0]))
                else:
                    # Single number pattern [*3] or [*..5]
                    max_hops = max(max_hops, int(match))
        
        # Count explicit relationship chains (rough estimate)
        chain_count = len(re.findall(r'-\[.*?\]-', cypher))
        if chain_count > max_hops:
            max_hops = chain_count
            
        return max_hops
    
    # -------------------------------------------------------------- _find_write_operations()
    def _find_write_operations(self, cypher: str) -> List[str]:
        """Find write operations in query."""
        found_operations = []
        for operation in self.WRITE_OPERATIONS:
            if re.search(rf'\b{operation}\b', cypher, re.IGNORECASE):
                found_operations.append(operation)
        return found_operations
    
    # -------------------------------------------------------------- _find_function_calls()
    def _find_function_calls(self, cypher: str) -> List[str]:
        """Find function calls in query."""
        found_functions = []
        for func_pattern in self.DANGEROUS_FUNCTIONS:
            if re.search(func_pattern, cypher, re.IGNORECASE):
                found_functions.append(func_pattern.replace('\\\\', '').replace('\\s+', ' '))
        return found_functions

# =========================================================================
# Enhanced Cypher Validator
# =========================================================================

# ------------------------------------------------------------------------- class CypherValidator
class CypherValidator:
    """
    Enhanced Cypher validator with safety checks and smart rewrites.
    
    Features:
    - Comprehensive safety validation (no writes, dangerous functions)
    - Schema validation against cached CompleteKGSchema
    - Smart rewrites for common issues
    - Actionable error reporting
    - Fallback recommendations
    """
    
    # ______________________
    # Constructor 
    # 
    # --------------------------------------------------------------------------------- __init__()
    def __init__(
        self,
        max_hops: int = 3,
        force_limit: int = 50,
        allow_call: bool = False
    ):
        """
        Initialize validator with safety constraints.
        
        Args:
            max_hops: Maximum allowed relationship hops
            force_limit: Default limit to inject if missing
            allow_call: Whether to allow CALL procedures (dangerous)
        """
        self.max_hops = max_hops
        self.force_limit = force_limit
        self.allow_call = allow_call
        self.parser = CypherParser()
        self.logger = logging.getLogger(__name__)
    # --------------------------------------------------------------------------------- end __init__()
    
    # =========================================================================
    # Functionality: Validation Workflow
    # =========================================================================
    # -------------------------------------------------------------- validate()
    def validate(
        self,
        cypher: str,
        schema: Optional[CompleteKGSchema] = None
    ) -> ValidationReport:
        """
        Validate Cypher query with comprehensive safety and schema checks.
        
        Args:
            cypher: Cypher query to validate
            schema: Complete knowledge graph schema for validation
            
        Returns:
            Comprehensive validation report
        """
        report = ValidationReport(
            is_valid=True,
            original_cypher=cypher,
            safe_cypher=cypher
        )
        
        try:
            # Parse query
            parsed = self.parser.parse(cypher)
            
            # Safety validation
            self._validate_safety(parsed, report)
            
            # Structural validation
            self._validate_structure(parsed, report)
            
            # Schema validation
            if schema:
                self._validate_schema(parsed, schema, report)
            
            # Apply fixes if possible
            if report.issues and not report.has_errors:
                report.safe_cypher = self._apply_fixes(cypher, parsed, report)
            
            # Determine if fallback is recommended
            # Use string comparison instead of enum comparison to avoid enum issues
            error_count = sum(1 for issue in report.issues if str(issue.severity.value).lower() == "error")
            unknown_elements = sum(
                len(elements) for elements in report.schema_elements_found.values()
                if any("unknown" in str(e).lower() for e in elements)
            )

            report.fallback_recommended = error_count > 0 or unknown_elements > 1
            # Only ERROR-level issues should make a query invalid, not warnings
            report.is_valid = error_count == 0
            
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            report.is_valid = False
            report.issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="VALIDATION_EXCEPTION",
                message=f"Validation failed: {str(e)}"
            ))
        
        return report
    # -------------------------------------------------------------- end validate()
    
    # ______________________
    # Internal/Private Methods (single leading underscore convention)
    #
    # -------------------------------------------------------------- _validate_safety()
    def _validate_safety(self, parsed: ParsedCypher, report: ValidationReport) -> None:
        """Validate query safety (no writes, dangerous functions)."""
        
        # Check for write operations
        if parsed.write_operations:
            for operation in parsed.write_operations:
                report.issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="WRITE_OPERATION_FORBIDDEN",
                    message=f"Write operation '{operation}' is forbidden in read-only queries",
                    element=operation,
                    suggestion="Remove write operations and use only MATCH, RETURN, WHERE, etc."
                ))
        
        # Check for dangerous function calls
        if parsed.function_calls:
            for func in parsed.function_calls:
                report.issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="DANGEROUS_FUNCTION_FORBIDDEN",
                    message=f"Dangerous function '{func}' is forbidden",
                    element=func,
                    suggestion="Use only allowed functions like count(), sum(), collect(), etc."
                ))
        
        # Check UNWIND usage (scrutinize but don't always block)
        if parsed.contains_unwind and parsed.write_operations:
            report.issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="UNWIND_WITH_WRITES",
                message="UNWIND combined with write operations is forbidden",
                suggestion="Remove write operations or use UNWIND only for read operations"
            ))
        elif parsed.contains_unwind:
            report.issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="UNWIND_USAGE",
                message="UNWIND detected - ensure it's used only for read operations",
                suggestion="Verify UNWIND is not modifying data"
            ))
        
        # Check for CALL procedures if not allowed
        if not self.allow_call and 'CALL' in parsed.clauses:
            report.issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="CALL_FORBIDDEN",
                message="CALL procedures are forbidden in this context",
                suggestion="Use direct Cypher patterns instead of procedure calls"
            ))
    
    # -------------------------------------------------------------- _validate_structure()
    def _validate_structure(self, parsed: ParsedCypher, report: ValidationReport) -> None:
        """Validate query structure (RETURN, LIMIT, hops)."""
        
        # Check for RETURN clause
        if not parsed.has_return:
            report.issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="MISSING_RETURN",
                message="Query must have an explicit RETURN clause",
                suggestion="Add RETURN clause specifying what to return"
            ))
        
        # Check for explicit RETURN (not just RETURN *)
        if parsed.has_return and not parsed.has_explicit_return:
            report.issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="WILDCARD_RETURN",
                message="Prefer explicit RETURN over RETURN *",
                suggestion="Specify exact fields to return instead of using RETURN *"
            ))
        
        # Check for LIMIT clause
        if not parsed.has_limit:
            report.issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="MISSING_LIMIT",
                message="Query should have a LIMIT clause to prevent large result sets",
                suggestion=f"Add LIMIT {self.force_limit} to limit results"
            ))
        
        # Check hop limit
        if parsed.max_hops > self.max_hops:
            report.issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="EXCESSIVE_HOPS",
                message=f"Query has {parsed.max_hops} hops, maximum recommended is {self.max_hops}",
                element=str(parsed.max_hops),
                suggestion=f"Reduce relationship traversal depth to {self.max_hops} or fewer"
            ))
    
    # -------------------------------------------------------------- _validate_schema()
    def _validate_schema(
        self,
        parsed: ParsedCypher,
        schema: CompleteKGSchema,
        report: ValidationReport
    ) -> None:
        """Validate query against knowledge graph schema."""
        
        # Get schema elements
        schema_node_labels = set(schema.get_node_labels())
        schema_relationship_types = set(schema.get_relationship_types())
        schema_properties = set(schema.get_all_properties())
        
        # Validate node labels
        unknown_labels = parsed.node_labels - schema_node_labels
        for label in unknown_labels:
            report.issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="UNKNOWN_NODE_LABEL",
                message=f"Node label '{label}' does not exist in the knowledge graph",
                element=label,
                suggestion=f"Use valid labels: {', '.join(sorted(schema_node_labels)[:5])}..."
            ))
        
        # Validate relationship types
        unknown_relationships = parsed.relationship_types - schema_relationship_types
        for rel in unknown_relationships:
            report.issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="UNKNOWN_RELATIONSHIP_TYPE",
                message=f"Relationship type '{rel}' does not exist in the knowledge graph",
                element=rel,
                suggestion=f"Use valid relationships: {', '.join(sorted(schema_relationship_types)[:5])}..."
            ))
        
        # Validate properties (warnings only, as they can be dropped)
        unknown_properties = parsed.property_keys - schema_properties
        for prop in unknown_properties:
            report.issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="UNKNOWN_PROPERTY",
                message=f"Property '{prop}' may not exist in the knowledge graph",
                element=prop,
                suggestion=f"Verify property exists or use: {', '.join(sorted(schema_properties)[:5])}..."
            ))
        
        # Store found schema elements
        report.schema_elements_found = {
            "node_labels": list(parsed.node_labels),
            "relationship_types": list(parsed.relationship_types),
            "properties": list(parsed.property_keys),
            "valid_labels": list(parsed.node_labels & schema_node_labels),
            "valid_relationships": list(parsed.relationship_types & schema_relationship_types),
            "valid_properties": list(parsed.property_keys & schema_properties)
        }
    
    # -------------------------------------------------------------- _apply_fixes()
    def _apply_fixes(self, original_cypher: str, parsed: ParsedCypher, report: ValidationReport) -> str:
        """Apply automatic fixes to query where possible."""
        fixed_cypher = original_cypher
        
        # Fix 1: Inject LIMIT if missing
        if not parsed.has_limit and not report.has_errors:
            fixed_cypher = self._inject_limit(fixed_cypher)
            report.fixes_applied.append(FixApplied(
                description="Injected LIMIT clause",
                original_element="(no LIMIT)",
                fixed_element=f"LIMIT {self.force_limit}",
                fix_type="injection"
            ))
        
        # Fix 2: Remove unknown properties from WHERE clauses (conservative approach)
        unknown_props = [issue.element for issue in report.issues 
                        if issue.code == "UNKNOWN_PROPERTY"]
        if unknown_props:
            for prop in unknown_props:
                # Simple removal of property conditions (very conservative)
                prop_pattern = rf'\s+AND\s+\w+\.{prop}\s*[=<>!]+\s*[^\\s]+'
                if re.search(prop_pattern, fixed_cypher, re.IGNORECASE):
                    fixed_cypher = re.sub(prop_pattern, '', fixed_cypher, flags=re.IGNORECASE)
                    report.fixes_applied.append(FixApplied(
                        description=f"Removed unknown property condition",
                        original_element=f"condition with {prop}",
                        fixed_element="(removed)",
                        fix_type="removal"
                    ))
        
        return fixed_cypher
    
    # -------------------------------------------------------------- _inject_limit()
    def _inject_limit(self, cypher: str) -> str:
        """Safely inject LIMIT clause into query."""
        # If query already ends with LIMIT, don't inject
        if re.search(r'\\bLIMIT\\s+\\d+\\s*$', cypher, re.IGNORECASE):
            return cypher
        
        # Simple injection at the end
        cypher = cypher.strip()
        if not cypher.endswith(';'):
            cypher += f" LIMIT {self.force_limit}"
        else:
            cypher = cypher[:-1] + f" LIMIT {self.force_limit};"
        
        return cypher
# ------------------------------------------------------------------------- end class CypherValidator

# __________________________________________________________________________
# Standalone Function Definitions
#
# =========================================================================
# Factory Function
# =========================================================================

# --------------------------------------------------------------------------------- get_cypher_validator()
def get_cypher_validator(
    max_hops: int = 3,
    force_limit: int = 50,
    allow_call: bool = False
) -> CypherValidator:
    """
    Get configured Cypher validator instance.
    
    Args:
        max_hops: Maximum allowed relationship hops
        force_limit: Default limit to inject if missing  
        allow_call: Whether to allow CALL procedures
        
    Returns:
        Configured CypherValidator
    """
    return CypherValidator(
        max_hops=max_hops,
        force_limit=force_limit,
        allow_call=allow_call
    )
# --------------------------------------------------------------------------------- end get_cypher_validator()

# __________________________________________________________________________
# End of File
