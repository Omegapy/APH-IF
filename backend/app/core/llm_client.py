# -------------------------------------------------------------------------
# File: core/llm_client.py
# Author: Alexander Ricciardi
# Date: 2025-09-24
# [File Path] backend/app/core/llm_client.py
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
#   Provides OpenAI client wrappers, rate limiting, token tracking, and fusion
#   helpers used by the APH-IF backend to interact with GPT-5 models.
# -------------------------------------------------------------------------

# --- Module Contents Overview ---
# - Dataclass: TokenUsage
# - Class: RateLimiter
# - Class: OpenAIClient
# - Functions: get_openai_client, openai_client, validate_client_setup
# -------------------------------------------------------------------------

# --- Dependencies / Imports ---
# - Standard Library: asyncio, contextlib.asynccontextmanager, dataclasses,
#   datetime, logging, time, typing
# - Third-Party: openai.OpenAI, openai.AsyncOpenAI, openai.types.chat.ChatCompletion
# - Local Project Modules: backend.app.core.config, backend.app.monitoring.timing_collector
# --- Requirements ---
# - Python 3.12+
# -------------------------------------------------------------------------

# --- Usage / Integration ---
#   Instantiate via get_openai_client() or openai_client context manager for
#   consistent GPT-5 access with rate limiting and monitoring instrumentation.
# -------------------------------------------------------------------------

# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------

"""
LLM client utilities enabling GPT-5 integrations with rate limiting,
monitoring, and advanced prompt handling for the APH-IF backend.
"""

# __________________________________________________________________________
# Imports
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import openai
from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion

from ..monitoring.timing_collector import get_timing_collector
from .config import EFFECTIVE_RPM, EFFECTIVE_TPM, EMBEDDING_CONFIG, LLM_MODEL_CONFIG, settings

logger = logging.getLogger(__name__)

# __________________________________________________________________________
# Class Definitions

# ------------------------------------------------------------------------- class TokenUsage
@dataclass(slots=True, kw_only=True)
class TokenUsage:
    """Track token usage statistics for rate limiting purposes.

    Attributes:
        prompt_tokens: Tokens consumed by the prompt portion of a request.
        completion_tokens: Tokens generated in the completion.
        total_tokens: Combined prompt and completion tokens.
        timestamp: Time at which the usage was recorded.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    # ______________________
    # Getters (Property decorators are often preferred for simple getters)
    #
    # --------------------------------------------------------------------------------- age_seconds()
    @property
    def age_seconds(self) -> float:
        """Compute the age of the usage record in seconds."""
        return (datetime.now() - self.timestamp).total_seconds()
    # --------------------------------------------------------------------------------- end age_seconds()

# ------------------------------------------------------------------------- end class TokenUsage

# ------------------------------------------------------------------------- class RateLimiter
class RateLimiter:
    """Enforce OpenAI Tier 3 rate limits for requests and tokens.

    Attributes:
        max_requests_per_minute: Allowed request throughput per minute.
        max_tokens_per_minute: Allowed token throughput per minute.
        request_timestamps: Sliding window of request timestamps.
        token_usage_history: Rolling collection of token usage records.
        _lock: Asyncio lock protecting shared state.
    """

    # ______________________
    # Constructor
    #
    # --------------------------------------------------------------------------------- __init__()
    def __init__(
        self,
        max_requests_per_minute: int = 0,
        max_tokens_per_minute: int = 0,
    ) -> None:
        """Initialize the rate limiter using effective RPM/TPM defaults."""
        self.max_requests_per_minute = max_requests_per_minute or EFFECTIVE_RPM
        self.max_tokens_per_minute = max_tokens_per_minute or EFFECTIVE_TPM

        self.request_timestamps: List[datetime] = []
        self.token_usage_history: List[TokenUsage] = []
        self._lock = asyncio.Lock()

        logger.info(
            "Rate limiter initialized: %s RPM, %s TPM",
            self.max_requests_per_minute,
            f"{self.max_tokens_per_minute:,}",
        )
    # --------------------------------------------------------------------------------- end __init__()

    # ______________________
    # Internal/Private Methods (single leading underscore convention)
    #
    # --------------------------------------------------------------------------------- _cleanup_windows()
    def _cleanup_windows(self) -> None:
        """Remove usage entries older than one minute from sliding windows."""
        cutoff = datetime.now() - timedelta(minutes=1)
        self.request_timestamps = [ts for ts in self.request_timestamps if ts > cutoff]
        self.token_usage_history = [usage for usage in self.token_usage_history if usage.timestamp > cutoff]
    # --------------------------------------------------------------------------------- end _cleanup_windows()

    # ______________________
    # Setters / Mutators
    #
    # --------------------------------------------------------------------------------- can_make_request()
    async def can_make_request(self, estimated_tokens: int = 0) -> bool:
        """Determine whether a new request fits within rate limits."""
        async with self._lock:
            self._cleanup_windows()

            if len(self.request_timestamps) >= self.max_requests_per_minute:
                logger.warning("RPM limit would be exceeded")
                return False

            current_tokens = sum(usage.total_tokens for usage in self.token_usage_history)
            if current_tokens + estimated_tokens > self.max_tokens_per_minute:
                logger.warning(
                    "TPM limit would be exceeded: %s tokens",
                    f"{current_tokens + estimated_tokens:,}",
                )
                return False

            return True
    # --------------------------------------------------------------------------------- end can_make_request()

    # --------------------------------------------------------------------------------- record_request()
    async def record_request(self, token_usage: Optional[TokenUsage] = None) -> None:
        """Record completion of a request for RPM/TPM accounting."""
        async with self._lock:
            now = datetime.now()
            self.request_timestamps.append(now)

            if token_usage:
                token_usage.timestamp = now
                self.token_usage_history.append(token_usage)
    # --------------------------------------------------------------------------------- end record_request()

    # --------------------------------------------------------------------------------- wait_if_needed()
    async def wait_if_needed(self, estimated_tokens: int = 0) -> None:
        """Sleep if needed to avoid future rate limit violations."""
        if not await self.can_make_request(estimated_tokens):
            wait_time = 5.0

            if self.request_timestamps:
                oldest_request = min(self.request_timestamps)
                request_wait = 61 - (datetime.now() - oldest_request).total_seconds()
                wait_time = max(wait_time, request_wait)

            if self.token_usage_history and estimated_tokens > 0:
                oldest_usage = min(self.token_usage_history, key=lambda usage: usage.timestamp)
                token_wait = 61 - oldest_usage.age_seconds
                wait_time = max(wait_time, token_wait)

            logger.info("Rate limit protection: waiting %.1f seconds", wait_time)
            await asyncio.sleep(wait_time)
    # --------------------------------------------------------------------------------- end wait_if_needed()

# ------------------------------------------------------------------------- end class RateLimiter

# ------------------------------------------------------------------------- class OpenAIClient
class OpenAIClient:
    """Manage GPT-5 interactions with rate limiting and monitoring hooks.

    Attributes:
        api_key: API key used for OpenAI requests.
        model: Identifier of the language model in use.
        sync_client: Synchronous OpenAI client instance.
        async_client: Asynchronous OpenAI client instance.
        rate_limiter: RateLimiter enforcing RPM and TPM constraints.
    """

    # ______________________
    # Constructor
    #
    # --------------------------------------------------------------------------------- __init__()
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-5-mini",
        rate_limiter: Optional[RateLimiter] = None,
    ) -> None:
        """Instantiate the client with optional overrides for API key and model."""
        self.api_key = api_key or settings.openai_api_key
        self.model = model
        self.sync_client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)
        self.rate_limiter = rate_limiter or RateLimiter()

        logger.info("OpenAI client initialized with model: %s", self.model)
    # --------------------------------------------------------------------------------- end __init__()

    # ______________________
    # Internal/Private Methods (single leading underscore convention)
    #
    # --------------------------------------------------------------------------------- _prepare_messages()
    def _prepare_messages(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Combine optional system prompt with provided user/assistant messages."""
        prepared: List[Dict[str, str]] = []
        if system_prompt:
            prepared.append({"role": "system", "content": system_prompt})
        prepared.extend(messages)
        return prepared
    # --------------------------------------------------------------------------------- end _prepare_messages()

    # --------------------------------------------------------------------------------- _prepare_gpt5_parameters()
    def _prepare_gpt5_parameters(
        self,
        verbosity: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        grammar: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Construct parameter payload honoring model capabilities and overrides."""
        params: Dict[str, Any] = {"model": self.model, **kwargs}
        model_config = LLM_MODEL_CONFIG

        if model_config.get("supports_temperature_control", True):
            params["temperature"] = kwargs.get("temperature", model_config["temperature"])
        else:
            params["temperature"] = model_config["temperature"]

        token_param = model_config.get("parameter_name_tokens", "max_tokens")
        params[token_param] = model_config["max_tokens"]

        if grammar:
            params["response_format"] = {"type": "grammar", "grammar": grammar}

        return params
    # --------------------------------------------------------------------------------- end _prepare_gpt5_parameters()

    # ______________________
    # Setters / Mutators
    #
    # --------------------------------------------------------------------------------- chat_completion()
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        verbosity: str = "medium",
        reasoning_effort: str = "medium",
        grammar: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> ChatCompletion:
        """Generate a GPT-5 chat completion with timing and rate limiting."""
        timing_collector = get_timing_collector()

        async with timing_collector.measure(
            "llm_api_request",
            {
                "model": LLM_MODEL_CONFIG["model"],
                "message_count": len(messages),
                "has_system_prompt": system_prompt is not None,
                "verbosity": verbosity,
                "reasoning_effort": reasoning_effort,
                "has_tools": tools is not None,
            },
        ) as llm_timer:
            async with timing_collector.measure("token_encoding") as encoding_timer:
                prepared_messages = self._prepare_messages(messages, system_prompt)
                estimated_tokens = sum(len(msg["content"]) // 4 for msg in prepared_messages)
                encoding_timer.add_metadata(
                    {
                        "estimated_tokens": estimated_tokens,
                        "message_count": len(prepared_messages),
                    }
                )

            async with timing_collector.measure("rate_limit_wait") as rate_limit_timer:
                rate_limit_start = time.perf_counter()
                await self.rate_limiter.wait_if_needed(estimated_tokens)
                rate_limit_wait_ms = (time.perf_counter() - rate_limit_start) * 1000
                rate_limit_timer.add_metadata(
                    {
                        "wait_time_ms": rate_limit_wait_ms,
                        "estimated_tokens": estimated_tokens,
                    }
                )

            async with timing_collector.measure("parameter_preparation") as param_timer:
                params = self._prepare_gpt5_parameters(
                    verbosity=verbosity,
                    reasoning_effort=reasoning_effort,
                    grammar=grammar,
                    **kwargs,
                )
                params["messages"] = prepared_messages
                if tools:
                    params["tools"] = tools
                    params["tool_choice"] = "auto"
                param_timer.add_metadata(
                    {"parameter_count": len(params), "has_tools": tools is not None}
                )

            try:
                logger.debug("Making OpenAI request with %s messages", len(prepared_messages))
                async with timing_collector.measure("llm_network_request") as network_timer:
                    network_start = time.perf_counter()
                    response = await self.async_client.chat.completions.create(**params)
                    network_time_ms = (time.perf_counter() - network_start) * 1000
                    network_timer.add_metadata(
                        {"network_time_ms": network_time_ms, "response_received": True}
                    )

                async with timing_collector.measure("token_decoding") as decoding_timer:
                    if response.usage:
                        usage = TokenUsage(
                            prompt_tokens=response.usage.prompt_tokens,
                            completion_tokens=response.usage.completion_tokens,
                            total_tokens=response.usage.total_tokens,
                        )
                        await self.rate_limiter.record_request(usage)
                        decoding_timer.add_metadata(
                            {
                                "prompt_tokens": usage.prompt_tokens,
                                "completion_tokens": usage.completion_tokens,
                                "total_tokens": usage.total_tokens,
                            }
                        )
                    else:
                        decoding_timer.add_metadata({"no_usage_data": True})

                llm_timer.add_metadata(
                    {
                        "success": True,
                        "rate_limit_wait_ms": rate_limit_wait_ms,
                        "network_time_ms": network_time_ms,
                        "final_token_count": response.usage.total_tokens
                        if response.usage
                        else estimated_tokens,
                        "model_used": response.model if hasattr(response, "model") else LLM_MODEL_CONFIG["model"],
                    }
                )

                return response

            except openai.RateLimitError as exc:
                logger.warning("Rate limit hit despite protection: %s", exc)
                async with timing_collector.measure("llm_retry") as retry_timer:
                    await asyncio.sleep(10)
                    retry_start = time.perf_counter()
                    response = await self.async_client.chat.completions.create(**params)
                    retry_time_ms = (time.perf_counter() - retry_start) * 1000
                    retry_timer.add_metadata(
                        {
                            "retry_reason": "rate_limit",
                            "retry_delay_ms": 10000,
                            "retry_time_ms": retry_time_ms,
                        }
                    )
                llm_timer.add_metadata(
                    {
                        "success": True,
                        "required_retry": True,
                        "retry_reason": "rate_limit",
                    }
                )
                return response

            except openai.APIError as exc:
                logger.error("OpenAI API error: %s", exc)
                llm_timer.add_metadata(
                    {"success": False, "error_type": "api_error", "error_message": str(exc)}
                )
                raise

            except Exception as exc:
                logger.error("Unexpected error in chat completion: %s", exc)
                raise
    # --------------------------------------------------------------------------------- end chat_completion()

    # --------------------------------------------------------------------------------- text_to_cypher()
    async def text_to_cypher(
        self,
        user_query: str,
        schema_info: Dict[str, Any],
        examples: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Convert natural language user queries to Cypher statements."""
        system_prompt = """You are an expert at converting natural language questions into Neo4j Cypher queries.

Knowledge Graph Schema:
- Node Labels: {', '.join(schema_info.get('node_labels', []))}
- Relationship Types: {', '.join(schema_info.get('relationship_types', []))}
- Key Properties: {', '.join(schema_info.get('property_keys', []))}

Instructions:
1. Generate only valid Cypher syntax
2. Use the exact node labels and relationship types from the schema
3. Return only the Cypher query, no explanations
4. Focus on the user's intent and return relevant information
"""

        if examples:
            system_prompt += "\n\nExample queries:\n"
            for example in examples:
                system_prompt += f"Question: {example.get('question', '')}\n"
                system_prompt += f"Cypher: {example.get('cypher', '')}\n\n"

        messages = [{"role": "user", "content": f"Convert this to Cypher: {user_query}"}]

        response = await self.chat_completion(messages=messages, system_prompt=system_prompt)
        return response.choices[0].message.content.strip()  # type: ignore
    # --------------------------------------------------------------------------------- end text_to_cypher()

    # --------------------------------------------------------------------------------- semantic_search_query_embedding()
    async def semantic_search_query_embedding(self, query: str) -> List[float]:
        """Generate an embedding vector for a semantic search query."""
        try:
            embedding_config = EMBEDDING_CONFIG
            response = await self.async_client.embeddings.create(
                model=embedding_config["model"],
                input=query,
            )
            embedding = response.data[0].embedding
            logger.debug("Generated embedding for query: %s dimensions", len(embedding))
            return embedding
        except Exception as exc:
            logger.error("Error generating query embedding: %s", exc)
            raise
    # --------------------------------------------------------------------------------- end semantic_search_query_embedding()

    # --------------------------------------------------------------------------------- fusion_analysis()
    async def fusion_analysis(
        self,
        user_query: str,
        traversal_results: List[Dict[str, Any]],
        semantic_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Fuse traversal and semantic results into a combined analysis."""
        system_prompt = """You are an expert at analyzing and fusing results from both graph traversal (GraphRAG) and semantic search (VectorRAG) to provide comprehensive answers.

Your task:
1. Analyze both sets of results for relevance to the user's query
2. Identify complementary information between the two approaches
3. Synthesize a comprehensive response that leverages both graph structure and semantic similarity
4. Highlight any conflicts or gaps in the information
5. Provide confidence scores for different aspects of your answer"""

        traversal_summary = f"Graph Traversal Results ({len(traversal_results)} items):\n"
        for index, result in enumerate(traversal_results[:5], 1):
            traversal_summary += f"{index}. {result}\n"

        semantic_summary = f"Semantic Search Results ({len(semantic_results)} items):\n"
        for index, result in enumerate(semantic_results[:5], 1):
            semantic_summary += f"{index}. {result}\n"

        messages = [
            {
                "role": "user",
                "content": (
                    f"User Query: {user_query}\n\n{traversal_summary}\n\n"
                    f"{semantic_summary}\n\nPlease provide a comprehensive analysis and response that fuses insights from both approaches."
                ),
            }
        ]

        response = await self.chat_completion(messages=messages, system_prompt=system_prompt)

        return {
            "fused_response": response.choices[0].message.content,
            "traversal_count": len(traversal_results),
            "semantic_count": len(semantic_results),
            "confidence": "medium",
            "reasoning_tokens": response.usage.completion_tokens if response.usage else 0,
        }
    # --------------------------------------------------------------------------------- end fusion_analysis()

# ------------------------------------------------------------------------- end class OpenAIClient

# __________________________________________________________________________
# Global Constants / Variables
#

# --------------------------------------------------------------------------------- _client
# Global client instance
_client: Optional[OpenAIClient] = None

# __________________________________________________________________________
# Global Constants / Variables
#

# __________________________________________________________________________
# Standalone Function Definitions
#

# --------------------------------------------------------------------------------- get_openai_client()
def get_openai_client() -> OpenAIClient:
    """Get or create the global OpenAI client instance.

    Returns:
        OpenAIClient: Singleton OpenAI client configured with project defaults.
    """
    global _client
    if _client is None:
        _client = OpenAIClient(model=settings.openai_model_mini)
    return _client
# --------------------------------------------------------------------------------- end get_openai_client()

# --------------------------------------------------------------------------------- openai_client()
@asynccontextmanager
async def openai_client():
    """Provide a managed OpenAI client instance within an async context.

    Yields:
        OpenAIClient: Initialized client ready for API operations.
    """
    client = get_openai_client()
    try:
        yield client
    finally:
        pass  # Cleanup if needed
# --------------------------------------------------------------------------------- end openai_client()

# --------------------------------------------------------------------------------- validate_client_setup()
async def validate_client_setup() -> Dict[str, Any]:
    """Validate LLM client setup for environment scripts.

    Returns:
        Dict[str, Any]: Validation metadata describing client readiness,
        rate limiter configuration, embedding capability, and any encountered
        errors.
    """
    validation = {
        "client_initialized": False,
        "api_connection": False,
        "rate_limiter": False,
        "embedding_model": False,
        "model_info": {},
        "errors": []
    }

    try:
        client = get_openai_client()
        validation["client_initialized"] = True
        validation["model_info"]["primary"] = client.model

        # Test minimal API call
        await client.chat_completion(
            messages=[{"role": "user", "content": "OK"}]
        )
        validation["api_connection"] = True

        # Check rate limiter
        if hasattr(client, 'rate_limiter') and client.rate_limiter:
            validation["rate_limiter"] = True
            validation["model_info"]["rate_limits"] = {
                "rpm": client.rate_limiter.max_requests_per_minute,
                "tpm": client.rate_limiter.max_tokens_per_minute
            }

        # Test embedding
        embedding = await client.semantic_search_query_embedding("test")
        if len(embedding) > 0:
            validation["embedding_model"] = True
            validation["model_info"]["embedding_dims"] = len(embedding)

    except Exception as e:
        validation["errors"].append(str(e))

    return validation
# --------------------------------------------------------------------------------- end validate_client_setup()

# __________________________________________________________________________
# End of File
#