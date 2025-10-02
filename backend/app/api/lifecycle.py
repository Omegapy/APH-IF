# -------------------------------------------------------------------------
# File: lifecycle.py
# Author: Alexander Ricciardi
# Date: 2025-09-25
# [File Path] backend/app/api/lifecycle.py
# -------------------------------------------------------------------------
# Project: APH-IF
#
# Project description:
# Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF)
# is a novel Retrieval Augmented Generation (RAG) system that differs from
# traditional RAG approaches by performing semantic and traversal searches
# concurrently, rather than sequentially, and fusing the results using an LLM
# or an LRM to generate the final response.
# -------------------------------------------------------------------------
#
# --- Module Functionality ---
#   Registers FastAPI startup and shutdown handlers responsible for initializing
#   and tearing down shared resources (schema cache, LLM clients, CPU pool,
#   session state) across the APH-IF backend service.
# -------------------------------------------------------------------------
#
# --- Module Contents Overview ---
# - Function: register_startup
# - Function: register_shutdown
# - Constant: __all__ (module exports)
# -------------------------------------------------------------------------
#
# --- Dependencies / Imports ---
# - Standard Library: logging
# - Third-Party: fastapi (FastAPI)
# - Local Project Modules: app.core.config, app.api.state, app.schema,
#   app.search.tools, app.processing, app.core infrastructure clients
# --- Requirements ---
# - Python 3.12
# -------------------------------------------------------------------------
#
# --- Usage / Integration ---
# Imported by `backend/app/main.py` to attach lifecycle hooks during app
# initialization.
#
# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------

"""Lifecycle registration utilities for the APH-IF FastAPI service."""

# __________________________________________________________________________
# Imports

from __future__ import annotations

import logging

from fastapi import FastAPI

from ..core.config import settings
from . import state as api_state

logger = logging.getLogger("app.main")


# __________________________________________________________________________
# Lifecycle Registration
#
# -------------------------------------------------------------- register_startup()
def register_startup(app: FastAPI) -> None:
    """Attach the startup handler to the supplied FastAPI app.

    Args:
        app: FastAPI application instance receiving the startup handler.
    """

    async def on_startup() -> None:
        logger.info("=" * 70)
        logger.info("APH-IF Backend API Starting")
        logger.info("=" * 70)

        settings.log_configuration()

        logger.info("=" * 70)
        logger.info("🛡️ Configuration Safety Check")
        logger.info("=" * 70)

        logger.info("Environment Mode: %s", settings.environment_mode.value)
        logger.info("Neo4j Instance: %s", settings.get_neo4j_mode_name())

        safety_checks: list[str] = []
        if settings.is_production:
            logger.info("🔒 Production environment detected - verifying safety settings:")

            if not settings.llm_cypher_allow_call:
                safety_checks.append("✅ LLM_CYPHER_ALLOW_CALL is safely disabled")
            else:
                safety_checks.append("⚠️ LLM_CYPHER_ALLOW_CALL is enabled (risky)")

            if settings.llm_cypher_max_hops <= 5:
                safety_checks.append(
                    f"✅ LLM_CYPHER_MAX_HOPS is safely limited to {settings.llm_cypher_max_hops}"
                )
            else:
                safety_checks.append(
                    f"⚠️ LLM_CYPHER_MAX_HOPS is high ({settings.llm_cypher_max_hops})"
                )

            if settings.llm_cypher_force_limit <= 100:
                safety_checks.append(
                    f"✅ LLM_CYPHER_FORCE_LIMIT is safely set to {settings.llm_cypher_force_limit}"
                )
            else:
                safety_checks.append(
                    f"⚠️ LLM_CYPHER_FORCE_LIMIT is high ({settings.llm_cypher_force_limit})"
                )

            for cap_name, cap_value in (
                ("Semantic", settings.validated_semantic_confidence_cap),
                ("Traversal", settings.validated_traversal_confidence_cap),
                ("Fusion", settings.validated_fusion_confidence_cap),
            ):
                safety_checks.append(f"✅ {cap_name} confidence cap: {cap_value}")
        else:
            logger.info("🔧 Development/Testing environment - relaxed safety settings allowed")
            safety_checks.append("✅ Development mode active")

        for check in safety_checks:
            logger.info("  %s", check)

        warning_count = sum(1 for check in safety_checks if "⚠️" in check)
        if warning_count == 0:
            logger.info("✅ All production safety checks passed")
        else:
            logger.warning("⚠️ %s safety warning(s) detected - review configuration", warning_count)

        logger.info("=" * 70)

        try:
            from ..schema import get_schema_manager

            schema_manager = get_schema_manager()
            cache_info = schema_manager.get_cache_info()
            logger.info(
                "✅ Schema cache initialized: files_exist=%s, valid=%s",
                cache_info.get("cache_file_exists", False),
                cache_info.get("cache_valid", False),
            )
        except ImportError:
            logger.warning("⚠️ Schema module not available")
        except Exception as exc:  # noqa: BLE001 - log detailed failure
            logger.error("❌ Schema cache initialization failed: %s", exc)

        try:
            from ..core.async_llm_client import get_async_llm_client
            from ..core.cpu_pool import get_cpu_pool
            from ..processing.citation_processor import get_citation_processor

            _ = await get_async_llm_client()  # Initialize singleton
            logger.info("✅ Async LLM client initialized")

            _ = get_citation_processor()  # Initialize singleton
            logger.info("✅ Citation processor initialized")

            _ = get_cpu_pool()  # Initialize singleton
            logger.info("✅ CPU pool started")
        except Exception as exc:  # noqa: BLE001 - log initialization issues
            logger.warning("⚠️ Some optimization components failed to initialize: %s", exc)

        if settings.use_llm_structural_cypher:
            try:
                from ..schema import get_schema_manager

                schema_manager = get_schema_manager()

                logger.info("🔄 Warming up schema manager for LLM structural queries...")
                structural_summary = schema_manager.get_structural_summary()

                if structural_summary:
                    try:
                        if hasattr(structural_summary, "get_node_labels_list"):
                            node_count = len(structural_summary.get_node_labels_list())
                            rel_count = len(structural_summary.get_relationship_types_list())
                        else:
                            summary_dict = (
                                structural_summary.to_dict()
                                if hasattr(structural_summary, "to_dict")
                                else structural_summary
                            )
                            node_count = len(summary_dict.get("node_labels", []))
                            rel_count = len(summary_dict.get("relationship_types", []))

                        logger.info(
                            "✅ Schema manager warmed up: %s node labels, %s relationship types",
                            node_count,
                            rel_count,
                        )
                    except Exception as api_error:  # noqa: BLE001
                        logger.warning(
                            "⚠️ Schema API access issue (will adapt at runtime): %s",
                            api_error,
                        )
                        logger.info("✅ Schema manager warmed up for semantic search (basic mode)")
                else:
                    logger.warning(
                        "⚠️ Schema structural summary not available - will load on first request"
                    )
            except ImportError:
                logger.warning("⚠️ Schema manager not available for warmup")
            except Exception as exc:  # noqa: BLE001
                logger.warning("⚠️ Schema manager warmup failed (non-fatal): %s", exc)
        else:
            logger.info("📋 LLM Structural Cypher disabled - skipping schema warmup")

        logger.info("🔄 Initializing semantic search engine...")
        try:
            from ..search.tools.vector import get_vector_engine

            _ = get_vector_engine()
            logger.info("✅ Semantic search engine ready")

            if settings.semantic_use_structural_schema:
                try:
                    from ..schema import get_schema_manager

                    schema_manager = get_schema_manager()

                    logger.info("🔄 Warming up schema manager for semantic search...")
                    structural_summary = schema_manager.get_structural_summary()

                    if structural_summary:
                        try:
                            if hasattr(structural_summary, "get_node_labels_list"):
                                node_count = len(structural_summary.get_node_labels_list())
                                rel_count = len(
                                    structural_summary.get_relationship_types_list()
                                )
                            else:
                                summary_dict = (
                                    structural_summary.to_dict()
                                    if hasattr(structural_summary, "to_dict")
                                    else structural_summary
                                )
                                node_count = len(summary_dict.get("node_labels", []))
                                rel_count = len(summary_dict.get("relationship_types", []))

                            logger.info(
                                (
                                    "✅ Schema manager warmed up for semantic search:"
                                    " %s node labels, %s relationship types"
                                ),
                                node_count,
                                rel_count,
                            )
                        except Exception as api_error:  # noqa: BLE001
                            logger.warning(
                                "⚠️ Schema API access issue (will adapt at runtime): %s", api_error
                            )
                            logger.info(
                                "✅ Schema manager warmed up for semantic search (basic mode)"
                            )
                    else:
                        logger.warning(
                            "⚠️ Schema structural summary not available - "
                            "will load on first request"
                        )
                except ImportError:
                    logger.warning(
                        "⚠️ Schema manager not available for semantic search warmup"
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "⚠️ Schema manager warmup for semantic search failed (non-fatal): %s",
                        exc,
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning("⚠️ Semantic engine initialization failed: %s", exc)

        logger.info(
            "🚀 APH-IF Backend API started on %s:%s", settings.host, settings.port
        )
        logger.info("=" * 70)

    app.add_event_handler("startup", on_startup)

# -------------------------------------------------------------- end register_startup()

# -------------------------------------------------------------- register_shutdown()
def register_shutdown(app: FastAPI) -> None:
    """Attach the shutdown handler to the supplied FastAPI app.

    Args:
        app: FastAPI application instance receiving the shutdown handler.
    """

    async def on_shutdown() -> None:
        logger.info("APH-IF Backend API shutting down")

        api_state.active_sessions.clear()

        try:
            from ..core.async_llm_client import shutdown_async_llm_client
            from ..core.cpu_pool import shutdown_cpu_pool
            from ..processing.citation_processor import shutdown_citation_processor

            await shutdown_async_llm_client()
            logger.info("✅ Async LLM client shutdown")

            await shutdown_citation_processor()
            logger.info("✅ Citation processor shutdown")

            shutdown_cpu_pool()
            logger.info("✅ CPU pool shutdown")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Error shutting down optimization components: %s", exc
            )

        try:
            from ..schema import get_schema_manager

            schema_manager = get_schema_manager()
            schema_manager.shutdown_database_connections()
            logger.info("✅ Database connections closed")
        except ImportError:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error closing database connections: %s", exc)

        logger.info("APH-IF Backend API shutdown complete")

    app.add_event_handler("shutdown", on_shutdown)

# -------------------------------------------------------------- end register_shutdown()

# __________________________________________________________________________
# Module Exports
#
__all__ = ["register_shutdown", "register_startup"]

# __________________________________________________________________________
# End of File