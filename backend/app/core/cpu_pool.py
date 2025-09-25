# -------------------------------------------------------------------------
# File: core/cpu_pool.py
# Author: Alexander Ricciardi
# Date: 2025-09-24
# [File Path] backend/app/core/cpu_pool.py
# ------------------------------------------------------------------------
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
#   Manages a shared process pool for CPU-bound tasks, providing async helpers
#   for compression, serialization, hashing, and custom workloads across the
#   APH-IF backend service.
# -------------------------------------------------------------------------
# --- Module Contents Overview ---
# - Dataclass: CPUTaskResult
# - Class: CPUPoolManager
# - Function: get_cpu_pool
# - Function: shutdown_cpu_pool
# -------------------------------------------------------------------------
# --- Dependencies / Imports ---
# - Standard Library: asyncio, concurrent.futures, dataclasses, gzip, hashlib,
#   json, logging, time, typing
# - Third-Party: None
# - Local Project Modules: None
# --- Requirements ---
# - Python 3.12+
# -------------------------------------------------------------------------
# --- Usage / Integration ---
#   Imported by backend components requiring CPU-intensive operations to run in
#   a shared process pool. Not intended as a standalone entry point.
# -------------------------------------------------------------------------
# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------

"""
CPU pool utilities supporting asynchronous access to process-bound workloads.

Provides the `CPUPoolManager` abstraction along with module-level helpers for
reusing a global process pool across the APH-IF backend service.
"""

from __future__ import annotations

# __________________________________________________________________________
# Imports
import asyncio
import concurrent.futures
import gzip
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# __________________________________________________________________________
# Dataclass Definitions
#
# ------------------------------------------------------------------------- class CPUTaskResult
@dataclass(slots=True, kw_only=True)
class CPUTaskResult:
    """Encapsulate the outcome of a CPU pool task execution.

    Attributes:
        task_id: Unique identifier assigned to the task invocation.
        result: Raw result payload returned by the executed function.
        duration_ms: Runtime duration expressed in milliseconds.
        success: Flag indicating whether the task completed successfully.
        error: Optional error message when `success` is False.
    """

    # ______________________
    #  Instance Fields
    #
    task_id: str
    result: Any
    duration_ms: float
    success: bool
    error: Optional[str] = None

# ------------------------------------------------------------------------- end class CPUTaskResult

# __________________________________________________________________________
# Class Definitions
#
# TODO: Evaluate @dataclass suitability—CPUPoolManager manages external executors and caches
#       requiring explicit lifecycle control, so conversion is deferred.
# ------------------------------------------------------------------------- class CPUPoolManager
class CPUPoolManager:
    """Manage process pool resources for CPU-intensive workloads.

    Responsibilities:
        - Start and stop a shared `ProcessPoolExecutor`.
        - Provide async wrappers around synchronous operations.
        - Maintain cache and metrics for executed tasks.

    Attributes:
        num_workers: Number of process workers allocated to the pool.
    """

    # ______________________
    # Constructor
    #
    # -------------------------------------------------------------- __init__()
    def __init__(self, num_workers: int = 4) -> None:
        """Initialize pool state and supporting metrics.

        Args:
            num_workers: Number of worker processes to provision.

        Returns:
            None
        """
        self.num_workers = num_workers
        self._executor: Optional[concurrent.futures.ProcessPoolExecutor] = None
        self._task_cache: Dict[str, Any] = {}
        self._metrics: Dict[str, float | int] = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "avg_duration_ms": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        logger.info("CPU pool manager initialized with %s workers", num_workers)

    # -------------------------------------------------------------- end __init__()

    # ______________________
    # Public Methods
    #
    # -------------------------------------------------------------- start()
    def start(self) -> None:
        """Start the process pool executor if not already running.

        Returns:
            None
        """
        if not self._executor:
            self._executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.num_workers,
            )
            logger.info("Process pool started with %s workers", self.num_workers)

    # -------------------------------------------------------------- end start()

    # -------------------------------------------------------------- stop()
    def stop(self) -> None:
        """Stop the process pool executor and release resources.

        Returns:
            None
        """
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
            logger.info("Process pool stopped")

    # -------------------------------------------------------------- end stop()

    # -------------------------------------------------------------- compress_content()
    async def compress_content(
        self,
        content: str,
        level: int = 6,
    ) -> bytes:
        """Compress textual content using gzip within the process pool.

        Args:
            content: Text content to compress.
            level: Compression level in the inclusive range [1, 9].

        Returns:
            Compressed payload expressed as bytes.
        """
        if not self._executor:
            self.start()

        loop = asyncio.get_event_loop()
        compressed = await loop.run_in_executor(
            self._executor,
            self._compress_sync,
            content,
            level,
        )

        return compressed

    # -------------------------------------------------------------- end compress_content()

    # -------------------------------------------------------------- decompress_content()
    async def decompress_content(self, compressed: bytes) -> str:
        """Decompress gzip content using the process pool.

        Args:
            compressed: Gzip-compressed content as bytes.

        Returns:
            Decompressed textual content.
        """
        if not self._executor:
            self.start()

        loop = asyncio.get_event_loop()
        content = await loop.run_in_executor(
            self._executor,
            self._decompress_sync,
            compressed,
        )

        return content

    # -------------------------------------------------------------- end decompress_content()

    # -------------------------------------------------------------- serialize_json()
    async def serialize_json(
        self,
        data: Any,
        compress: bool = False,
    ) -> bytes:
        """Serialize data into JSON bytes optionally compressed with gzip.

        Args:
            data: Arbitrary payload to serialize.
            compress: Flag controlling whether gzip compression is applied.

        Returns:
            Serialized bytes representing the payload.
        """
        if not self._executor:
            self.start()

        loop = asyncio.get_event_loop()
        serialized = await loop.run_in_executor(
            self._executor,
            self._serialize_json_sync,
            data,
            compress,
        )

        return serialized

    # -------------------------------------------------------------- end serialize_json()

    # -------------------------------------------------------------- deserialize_json()
    async def deserialize_json(
        self,
        data: bytes,
        compressed: bool = False,
    ) -> Any:
        """Deserialize JSON bytes optionally compressed with gzip.

        Args:
            data: Serialized JSON payload as bytes.
            compressed: Indicates whether gzip compression was applied.

        Returns:
            Deserialized Python object.
        """
        if not self._executor:
            self.start()

        loop = asyncio.get_event_loop()
        deserialized = await loop.run_in_executor(
            self._executor,
            self._deserialize_json_sync,
            data,
            compressed,
        )

        return deserialized

    # -------------------------------------------------------------- end deserialize_json()

    # -------------------------------------------------------------- process_regex_batch()
    async def process_regex_batch(
        self,
        patterns: List[str],
        texts: List[str],
    ) -> List[List[str]]:
        """Process multiple regex patterns across multiple texts concurrently.

        Args:
            patterns: Collection of regex patterns to evaluate.
            texts: Text strings to scan for pattern matches.

        Returns:
            Ordered match results for each input text.
        """
        if not self._executor:
            self.start()

        loop = asyncio.get_event_loop()

        tasks = [
            loop.run_in_executor(
                self._executor,
                self._process_regex_sync,
                patterns,
                text,
            )
            for text in texts
        ]

        results = await asyncio.gather(*tasks)
        return results

    # -------------------------------------------------------------- end process_regex_batch()

    # -------------------------------------------------------------- calculate_hash()
    async def calculate_hash(
        self,
        content: str,
        algorithm: str = "sha256",
    ) -> str:
        """Compute a cryptographic hash of the supplied content.

        Args:
            content: Text payload to hash.
            algorithm: Hashing algorithm identifier (md5, sha256, sha512).

        Returns:
            Hexadecimal digest string for the supplied content.
        """
        if not self._executor:
            self.start()

        cache_key = f"{algorithm}:{len(content)}:{content[:100]}"
        if cache_key in self._task_cache:
            self._metrics["cache_hits"] += 1
            return self._task_cache[cache_key]

        self._metrics["cache_misses"] += 1

        loop = asyncio.get_event_loop()
        hash_result = await loop.run_in_executor(
            self._executor,
            self._calculate_hash_sync,
            content,
            algorithm,
        )

        self._task_cache[cache_key] = hash_result

        return hash_result

    # -------------------------------------------------------------- end calculate_hash()

    # -------------------------------------------------------------- run_cpu_task()
    async def run_cpu_task(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> CPUTaskResult:
        """Execute an arbitrary CPU-bound function within the pool.

        Args:
            func: Callable to execute in the process pool.
            *args: Positional arguments passed to the callable.
            **kwargs: Keyword arguments passed to the callable.

        Returns:
            Structured task result with runtime metadata.
        """
        if not self._executor:
            self.start()

        task_id = f"task_{time.time()}"
        start_time = time.time()

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                func,
                *args,
                **kwargs,
            )

            duration_ms = (time.time() - start_time) * 1000

            self._metrics["total_tasks"] += 1
            self._metrics["successful_tasks"] += 1
            self._update_avg_duration(duration_ms)

            return CPUTaskResult(
                task_id=task_id,
                result=result,
                duration_ms=duration_ms,
                success=True,
            )

        except Exception as error:  # noqa: BLE001
            duration_ms = (time.time() - start_time) * 1000

            self._metrics["total_tasks"] += 1
            self._metrics["failed_tasks"] += 1

            logger.error("CPU task %s failed: %s", task_id, error)

            return CPUTaskResult(
                task_id=task_id,
                result=None,
                duration_ms=duration_ms,
                success=False,
                error=str(error),
            )

    # -------------------------------------------------------------- end run_cpu_task()

    # -------------------------------------------------------------- get_metrics()
    def get_metrics(self) -> Dict[str, Any]:
        """Return a snapshot of accumulated pool metrics.

        Returns:
            Copy of the metrics dictionary describing task performance.
        """
        return self._metrics.copy()

    # -------------------------------------------------------------- end get_metrics()

    # -------------------------------------------------------------- clear_cache()
    def clear_cache(self) -> None:
        """Clear cached task results maintained by the pool.

        Returns:
            None
        """
        self._task_cache.clear()
        logger.info("CPU task cache cleared")

    # -------------------------------------------------------------- end clear_cache()

    # -------------------------------------------------------------- __enter__()
    def __enter__(self) -> "CPUPoolManager":
        """Enter the context manager by starting the pool.

        Returns:
            Active pool manager instance.
        """
        self.start()
        return self

    # -------------------------------------------------------------- end __enter__()

    # -------------------------------------------------------------- __exit__()
    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        """Exit the context manager by stopping the pool.

        Args:
            exc_type: Exception type raised within the context, if any.
            exc_val: Exception instance raised within the context, if any.
            exc_tb: Traceback associated with the raised exception.

        Returns:
            None
        """
        self.stop()

    # -------------------------------------------------------------- end __exit__()

    # ______________________
    # Internal/Private Methods (single leading underscore convention)
    #
    # -------------------------------------------------------------- _compress_sync()
    @staticmethod
    def _compress_sync(content: str, level: int) -> bytes:
        """Synchronously compress content using gzip.

        Args:
            content: Text content to compress.
            level: Compression level in the inclusive range [1, 9].

        Returns:
            Compressed bytes representing the input content.
        """
        return gzip.compress(content.encode("utf-8"), compresslevel=level)

    # -------------------------------------------------------------- end _compress_sync()

    # -------------------------------------------------------------- _decompress_sync()
    @staticmethod
    def _decompress_sync(compressed: bytes) -> str:
        """Synchronously decompress gzip payloads.

        Args:
            compressed: Gzip-compressed content as bytes.

        Returns:
            Decompressed text string.
        """
        return gzip.decompress(compressed).decode("utf-8")

    # -------------------------------------------------------------- end _decompress_sync()

    # -------------------------------------------------------------- _serialize_json_sync()
    @staticmethod
    def _serialize_json_sync(data: Any, compress: bool) -> bytes:
        """Synchronously serialize data to JSON and optionally compress.

        Args:
            data: Payload to serialize to JSON.
            compress: Indicates whether gzip compression is applied.

        Returns:
            Serialized bytes representing the payload.
        """
        json_str = json.dumps(data, default=str, separators=(",", ":"))
        json_bytes = json_str.encode("utf-8")

        if compress:
            return gzip.compress(json_bytes, compresslevel=6)
        return json_bytes

    # -------------------------------------------------------------- end _serialize_json_sync()

    # -------------------------------------------------------------- _deserialize_json_sync()
    @staticmethod
    def _deserialize_json_sync(data: bytes, compressed: bool) -> Any:
        """Synchronously deserialize JSON and optionally decompress.

        Args:
            data: Serialized JSON payload as bytes.
            compressed: Indicates whether gzip compression was applied.

        Returns:
            Deserialized Python object.
        """
        if compressed:
            data = gzip.decompress(data)

        json_str = data.decode("utf-8")
        return json.loads(json_str)

    # -------------------------------------------------------------- end _deserialize_json_sync()

    # -------------------------------------------------------------- _process_regex_sync()
    @staticmethod
    def _process_regex_sync(patterns: List[str], text: str) -> List[str]:
        """Synchronously evaluate regex patterns over the supplied text.

        Args:
            patterns: Regex patterns evaluated against the text.
            text: Text content to analyze.

        Returns:
            List of pattern matches found in the text.
        """
        import re

        matches: List[str] = []
        for pattern in patterns:
            try:
                compiled = re.compile(pattern, re.IGNORECASE)
                found = compiled.findall(text)
                matches.extend(found)
            except Exception as error:  # noqa: BLE001
                logger.debug("Regex error for pattern %s: %s", pattern, error)

        return matches

    # -------------------------------------------------------------- end _process_regex_sync()

    # -------------------------------------------------------------- _calculate_hash_sync()
    @staticmethod
    def _calculate_hash_sync(content: str, algorithm: str) -> str:
        """Synchronously compute a hash for the supplied content.

        Args:
            content: Text content to hash.
            algorithm: Hash algorithm identifier (md5, sha256, sha512).

        Returns:
            Hash digest as a hexadecimal string.
        """
        if algorithm == "md5":
            return hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()
        if algorithm == "sha256":
            return hashlib.sha256(content.encode()).hexdigest()
        if algorithm == "sha512":
            return hashlib.sha512(content.encode()).hexdigest()

        return hashlib.sha256(content.encode()).hexdigest()

    # -------------------------------------------------------------- end _calculate_hash_sync()

    # -------------------------------------------------------------- _update_avg_duration()
    def _update_avg_duration(self, duration_ms: float) -> None:
        """Update the running average of successful task durations.

        Args:
            duration_ms: Latest task duration in milliseconds.

        Returns:
            None
        """
        total = self._metrics["successful_tasks"]
        if total == 1:
            self._metrics["avg_duration_ms"] = duration_ms
        else:
            alpha = 0.1
            self._metrics["avg_duration_ms"] = (
                alpha * duration_ms
                + (1 - alpha) * self._metrics["avg_duration_ms"]
            )

    # -------------------------------------------------------------- end _update_avg_duration()

# ------------------------------------------------------------------------- end class CPUPoolManager

# __________________________________________________________________________
# Standalone Function Definitions
#
_cpu_pool: Optional[CPUPoolManager] = None


# -------------------------------------------------------------- get_cpu_pool()
def get_cpu_pool(num_workers: int = 4) -> CPUPoolManager:
    """Provide a lazily initialized global CPU pool manager.

    Args:
        num_workers: Optional worker count when instantiating the pool.

    Returns:
        Shared `CPUPoolManager` instance.
    """
    global _cpu_pool

    if _cpu_pool is None:
        _cpu_pool = CPUPoolManager(num_workers=num_workers)
        _cpu_pool.start()

    return _cpu_pool


# -------------------------------------------------------------- end get_cpu_pool()

# -------------------------------------------------------------- shutdown_cpu_pool()
def shutdown_cpu_pool() -> None:
    """Shut down the global CPU pool manager if it exists.

    Returns:
        None
    """
    global _cpu_pool

    if _cpu_pool:
        _cpu_pool.stop()
        _cpu_pool = None

# -------------------------------------------------------------- end shutdown_cpu_pool()

# __________________________________________________________________________
# End of File