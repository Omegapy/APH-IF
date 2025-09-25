# -------------------------------------------------------------------------
# File: core/async_llm_client.py
# Author: Alexander Ricciardi
# Date: 2025-09-24
# [File Path] backend/app/core/async_llm_client.py
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
#   Provides asynchronous OpenAI client management with batching, metrics,
#   and lifecycle helpers used by the APH-IF backend core module.
# -------------------------------------------------------------------------
# --- Module Contents Overview ---
# - Dataclass: FusionRequest
# - Class: AsyncLLMClient
# - Function: get_async_llm_client
# - Function: shutdown_async_llm_client
# -------------------------------------------------------------------------
# --- Dependencies / Imports ---
# - Standard Library: asyncio, dataclasses, datetime, logging, time, typing
# - Third-Party: openai.AsyncOpenAI, langchain_core.messages
# - Local Project Modules: backend.app.core.config
# --- Requirements ---
# - Python 3.12+
# -------------------------------------------------------------------------
# --- Usage / Integration ---
#   Imported by backend services requiring shared async GPT-5-mini access
#   for hybrid retrieval fusion steps. Not intended as a standalone entry point.
# -------------------------------------------------------------------------
# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------

"""
Asynchronous OpenAI client helpers for the APH-IF backend core.

Exposes the `AsyncLLMClient` batching manager along with global accessors used by
FastAPI services during hybrid retrieval fusion workflows.
"""

from __future__ import annotations

# __________________________________________________________________________
# Imports
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from langchain_core.messages import HumanMessage, SystemMessage
from openai import AsyncOpenAI

from .config import settings

logger = logging.getLogger(__name__)

# __________________________________________________________________________
# Dataclass Definitions
#
# ------------------------------------------------------------------------- class FusionRequest
@dataclass(slots=True, kw_only=True)
class FusionRequest:
    """Represent a queued fusion request awaiting LLM completion.

    Attributes:
        request_id: Unique identifier for traceability.
        messages: Ordered LangChain system and human messages to submit.
        callback: Future used for publishing completion results.
        priority: Higher values receive earlier processing.
        created_at: Epoch seconds when the request entered the queue.
    """
    # ______________________
    #  Instance Fields
    #
    request_id: str
    messages: List[Union[SystemMessage, HumanMessage]]
    callback: asyncio.Future[str]
    priority: int = 0
    created_at: float = field(default_factory=time.time)

    # ______________________
    # Getters (Property decorators are often preferred for simple getters)
    #
    # -------------------------------------------------------------- age_ms()
    @property
    def age_ms(self) -> float:
        """Age of the request in milliseconds."""
        return (time.time() - self.created_at) * 1000

    # -------------------------------------------------------------- end age_ms()

# ------------------------------------------------------------------------- end class FusionRequest

# __________________________________________________________________________
# Class Definitions
#
# TODO: Evaluate @dataclass(slots=True, kw_only=True) suitability; resource management lifecycle
#       currently depends on explicit start/stop semantics, so conversion is deferred.
# ------------------------------------------------------------------------- class AsyncLLMClient
class AsyncLLMClient:
    """Manage asynchronous OpenAI chat completions with batching and metrics.

    Responsibilities:
        - Maintain a background batch processor for queued requests.
        - Coordinate priority-aware batching for GPT-5-mini invocations.
        - Provide metrics describing latency and streaming reliability.

    Attributes:
        model: Target OpenAI model identifier.
        max_batch_size: Upper bound on simultaneous requests per batch.
        batch_timeout_ms: Maximum delay before dispatching a partial batch.
    """

    # ______________________
    # Constructor
    #
    # -------------------------------------------------------------- __init__()
    def __init__(
        self,
        model: Optional[str] = None,
        max_batch_size: int = 5,
        batch_timeout_ms: int = 100,
    ) -> None:
        """Initialize client state and prepare OpenAI transport.

        Args:
            model: Optional override for the OpenAI chat model.
            max_batch_size: Maximum number of requests per dispatched batch.
            batch_timeout_ms: Milliseconds to wait for additional requests.

        Returns:
            None
        """
        self.model = model or settings.openai_model_mini
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms

        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=240.0,
            max_retries=2,
        )

        self._request_queue: asyncio.Queue[FusionRequest] = asyncio.Queue()
        self._batch_processor_task: Optional[asyncio.Task] = None
        self._shutdown = False

        self._metrics: Dict[str, float | int] = {
            "total_requests": 0,
            "batched_requests": 0,
            "streaming_success": 0,
            "streaming_failures": 0,
            "avg_latency_ms": 0.0,
            "connection_reuse_count": 0,
        }

        logger.info("Async LLM client initialized with model: %s", self.model)

    # -------------------------------------------------------------- end __init__()

    # ______________________
    # Public Methods
    #
    # -------------------------------------------------------------- start()
    async def start(self) -> None:
        """Launch the background batch processor task.

        Returns:
            None
        """
        if not self._batch_processor_task:
            self._batch_processor_task = asyncio.create_task(
                self._batch_processor(),
                name="llm_batch_processor",
            )
            logger.info("LLM batch processor started")

    # -------------------------------------------------------------- end start()

    # -------------------------------------------------------------- stop()
    async def stop(self) -> None:
        """Stop the batch processor and clean up transports.

        Returns:
            None
        """
        self._shutdown = True

        if self._batch_processor_task:
            self._batch_processor_task.cancel()
            try:
                await self._batch_processor_task
            except asyncio.CancelledError:
                pass

        if hasattr(self._client, "_client") and hasattr(self._client._client, "close"):
            await self._client._client.close()

        logger.info("Async LLM client stopped")

    # -------------------------------------------------------------- end stop()

    # -------------------------------------------------------------- complete()
    async def complete(
        self,
        messages: List[Union[SystemMessage, HumanMessage]],
        priority: int = 0,
    ) -> str:
        """Submit messages for completion using the batching queue.

        Args:
            messages: LangChain messages to forward to OpenAI.
            priority: Relative importance for queue ordering.

        Returns:
            The response text from the language model.
        """
        future: asyncio.Future[str] = asyncio.Future()

        request = FusionRequest(
            request_id=f"req_{time.time()}",
            messages=messages,
            callback=future,
            priority=priority,
        )

        await self._request_queue.put(request)

        return await future

    # -------------------------------------------------------------- end complete()

    # -------------------------------------------------------------- stream_complete()
    async def stream_complete(
        self,
        messages: List[Union[SystemMessage, HumanMessage]],
    ) -> AsyncIterator[str]:
        """Yield a completion stream response for the provided messages.

        Args:
            messages: LangChain messages to forward to OpenAI.

        Yields:
            The full completion text once available.

        Note:
            Streaming currently returns a single chunk until streaming support is reintroduced.
        """
        result = await self.complete(messages)
        yield result

    # -------------------------------------------------------------- end stream_complete()

    # -------------------------------------------------------------- get_metrics()
    def get_metrics(self) -> Dict[str, Any]:
        """Return a snapshot of collected metrics.

        Returns:
            Dictionary containing counters and latency averages for this client.
        """
        return self._metrics.copy()

    # -------------------------------------------------------------- end get_metrics()

    # -------------------------------------------------------------- __aenter__()
    async def __aenter__(self) -> "AsyncLLMClient":
        """Enter the async context manager scope for the client.

        Returns:
            Active async client instance ready for use within the context block.
        """
        await self.start()
        return self

    # -------------------------------------------------------------- end __aenter__()

    # -------------------------------------------------------------- __aexit__()
    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        """Exit the async context manager scope and stop resources.

        Args:
            exc_type: Exception type raised within the context, if any.
            exc_val: Exception instance raised within the context, if any.
            exc_tb: Traceback associated with the raised exception.

        Returns:
            None
        """
        await self.stop()

    # -------------------------------------------------------------- end __aexit__()

    # ______________________
    # Internal/Private Methods (single leading underscore convention)
    #
    # -------------------------------------------------------------- _batch_processor()
    async def _batch_processor(self) -> None:
        """Continuously consume queued requests and dispatch batches.

        Returns:
            None
        """
        batch: List[FusionRequest] = []
        last_batch_time = time.time()

        while not self._shutdown:
            try:
                timeout = self.batch_timeout_ms / 1000.0
                remaining_timeout = timeout - (time.time() - last_batch_time)

                if remaining_timeout <= 0 or len(batch) >= self.max_batch_size:
                    if batch:
                        await self._process_batch(batch)
                        batch = []
                    last_batch_time = time.time()
                    continue

                try:
                    request = await asyncio.wait_for(
                        self._request_queue.get(),
                        timeout=remaining_timeout,
                    )
                    batch.append(request)
                except asyncio.TimeoutError:
                    if batch:
                        await self._process_batch(batch)
                        batch = []
                    last_batch_time = time.time()
            except Exception as error:  # noqa: BLE001
                logger.error("Batch processor error: %s", error)
                await asyncio.sleep(0.1)

    # -------------------------------------------------------------- end _batch_processor()

    # -------------------------------------------------------------- _process_batch()
    async def _process_batch(self, batch: List[FusionRequest]) -> None:
        """Invoke processing concurrently for all requests in a batch.

        Args:
            batch: Requests gathered for a single OpenAI invocation cycle.

        Returns:
            None
        """
        self._metrics["batched_requests"] += len(batch)

        batch.sort(key=lambda item: (-item.priority, item.created_at))

        tasks = [
            self._process_single_request(request)
            for request in batch
        ]

        await asyncio.gather(*tasks, return_exceptions=True)

    # -------------------------------------------------------------- end _process_batch()

    # -------------------------------------------------------------- _process_single_request()
    async def _process_single_request(self, request: FusionRequest) -> None:
        """Execute a single completion request and publish results.

        Args:
            request: Fusion request assembled by the batching queue.

        Returns:
            None
        """
        start_time = time.time()

        try:
            messages = [
                {
                    "role": "system" if isinstance(message, SystemMessage) else "user",
                    "content": message.content,
                }
                for message in request.messages
            ]

            params = {
                "model": self.model,
                "messages": messages,
                "max_completion_tokens": 6000,
                "stream": False,
            }

            response = await self._client.chat.completions.create(**params)

            content = response.choices[0].message.content
            request.callback.set_result(content)

            latency = (time.time() - start_time) * 1000
            self._update_latency(latency)
            self._metrics["total_requests"] += 1
        except Exception as error:  # noqa: BLE001
            logger.error("Request processing error: %s", error)
            request.callback.set_exception(error)

    # -------------------------------------------------------------- end _process_single_request()

    # -------------------------------------------------------------- _update_latency()
    def _update_latency(self, latency_ms: float) -> None:
        """Update the exponential moving average for latency samples.

        Args:
            latency_ms: Measured request latency in milliseconds.

        Returns:
            None
        """
        total_requests = self._metrics["total_requests"]
        if total_requests == 0:
            self._metrics["avg_latency_ms"] = latency_ms
        else:
            alpha = 0.1
            self._metrics["avg_latency_ms"] = (
                alpha * latency_ms
                + (1 - alpha) * self._metrics["avg_latency_ms"]
            )

    # -------------------------------------------------------------- end _update_latency()

# ------------------------------------------------------------------------- end class AsyncLLMClient

# __________________________________________________________________________
# Standalone Function Definitions
#
_async_llm_client: Optional[AsyncLLMClient] = None


# -------------------------------------------------------------- get_async_llm_client()
async def get_async_llm_client() -> AsyncLLMClient:
    """Provide a singleton async LLM client instance.

    Returns:
        The lazily initialized async LLM client shared across the backend.
    """
    global _async_llm_client

    if _async_llm_client is None:
        _async_llm_client = AsyncLLMClient()
        await _async_llm_client.start()

    return _async_llm_client


# -------------------------------------------------------------- end get_async_llm_client()

# -------------------------------------------------------------- shutdown_async_llm_client()
async def shutdown_async_llm_client() -> None:
    """Release the singleton async LLM client resources.

    Returns:
        None
    """
    global _async_llm_client

    if _async_llm_client:
        await _async_llm_client.stop()
        _async_llm_client = None

# -------------------------------------------------------------- end shutdown_async_llm_client()

# __________________________________________________________________________
# End of File
#