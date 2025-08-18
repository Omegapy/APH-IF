"""
Monitored OpenAI Client

This module provides a wrapper around the OpenAI client that automatically
monitors all API calls, tracks usage, timing, and errors.

Usage:
    from common.monitored_openai import MonitoredOpenAIClient
    
    # Create monitored client
    client = MonitoredOpenAIClient(api_key="your-api-key")
    
    # Use exactly like regular OpenAI client - monitoring is automatic
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    
    # Get monitoring statistics
    stats = client.get_monitoring_stats()
    print(f"Total API calls: {stats['total_calls']}")
    print(f"Total tokens used: {stats['total_tokens_used']}")
"""

import os
import json
import time
import asyncio
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime, timedelta
import threading

try:
    import openai
    from openai import OpenAI
    from openai.types.chat import ChatCompletion
except ImportError:
    raise ImportError("OpenAI library not installed. Install with: pip install openai")

from .api_monitor import get_llm_monitor, LogLevel, APICallTracker


class RateLimiter:
    """
    Rate limiter for OpenAI API calls based on official OpenAI cookbook recommendations.

    Handles both requests per minute (RPM) and tokens per minute (TPM) limits.
    Implements exponential backoff for rate limit errors.
    """

    def __init__(self, requests_per_minute: int = 3, tokens_per_minute: int = 40000):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.request_times = []
        self.token_usage = []
        self._lock = threading.Lock()

        # Get limits from environment if available
        env_rpm = os.getenv('OPENAI_REQUESTS_PER_MINUTE')
        env_tpm = os.getenv('OPENAI_TOKENS_PER_MINUTE')

        if env_rpm:
            self.requests_per_minute = int(env_rpm)
        if env_tpm:
            self.tokens_per_minute = int(env_tpm)

    def wait_if_needed(self, estimated_tokens: int = 1000):
        """
        Wait if necessary to respect rate limits.

        Args:
            estimated_tokens: Estimated tokens for the upcoming request
        """
        with self._lock:
            now = datetime.now()

            # Clean old entries (older than 1 minute)
            cutoff = now - timedelta(minutes=1)
            self.request_times = [t for t in self.request_times if t > cutoff]
            self.token_usage = [(t, tokens) for t, tokens in self.token_usage if t > cutoff]

            # Check request rate limit
            if len(self.request_times) >= self.requests_per_minute:
                # Need to wait until the oldest request is more than 1 minute old
                oldest_request = min(self.request_times)
                wait_until = oldest_request + timedelta(minutes=1, seconds=1)  # Add 1 second buffer
                wait_time = (wait_until - now).total_seconds()

                if wait_time > 0:
                    print(f"⏳ Rate limit: waiting {wait_time:.1f}s (requests: {len(self.request_times)}/{self.requests_per_minute})")
                    time.sleep(wait_time)

            # Check token rate limit
            current_tokens = sum(tokens for _, tokens in self.token_usage)
            if current_tokens + estimated_tokens > self.tokens_per_minute:
                # Wait until enough tokens are available
                # Find when enough tokens will be freed up
                sorted_usage = sorted(self.token_usage, key=lambda x: x[0])
                tokens_needed = (current_tokens + estimated_tokens) - self.tokens_per_minute
                tokens_freed = 0

                for usage_time, tokens in sorted_usage:
                    tokens_freed += tokens
                    if tokens_freed >= tokens_needed:
                        wait_until = usage_time + timedelta(minutes=1, seconds=1)
                        wait_time = (wait_until - now).total_seconds()

                        if wait_time > 0:
                            print(f"⏳ Token limit: waiting {wait_time:.1f}s (tokens: {current_tokens + estimated_tokens}/{self.tokens_per_minute})")
                            time.sleep(wait_time)
                        break

            # Record this request
            self.request_times.append(now)

    def record_usage(self, tokens_used: int):
        """Record actual token usage after API call."""
        with self._lock:
            self.token_usage.append((datetime.now(), tokens_used))

    def handle_rate_limit_error(self, error, attempt: int = 1):
        """
        Handle rate limit errors with exponential backoff.

        Args:
            error: The rate limit error
            attempt: Current attempt number

        Returns:
            Wait time in seconds
        """
        # Extract wait time from error message if available
        error_str = str(error)

        # Default exponential backoff: 2^attempt seconds, max 60 seconds
        base_wait = min(2 ** attempt, 60)

        # Add jitter to avoid thundering herd
        import random
        jitter = random.uniform(0.1, 0.3) * base_wait
        wait_time = base_wait + jitter

        print(f"🚫 Rate limit error (attempt {attempt}): {error_str}")
        print(f"⏳ Waiting {wait_time:.1f}s before retry...")

        time.sleep(wait_time)
        return wait_time


class MonitoredChatCompletions:
    """Monitored wrapper for OpenAI chat completions."""
    
    def __init__(self, original_completions, monitor):
        self._original = original_completions
        self._monitor = monitor
    
    def create(self, **kwargs) -> ChatCompletion:
        """Create chat completion with monitoring and rate limiting."""
        model = kwargs.get('model', 'unknown')
        messages = kwargs.get('messages', [])
        operation = f"chat_completion_{model}"

        # Estimate tokens for rate limiting
        estimated_tokens = self._estimate_tokens(messages)

        # Apply rate limiting
        if hasattr(self._monitor, '_rate_limiter'):
            self._monitor._rate_limiter.wait_if_needed(estimated_tokens)

        # Add timeout to prevent freezing
        if 'timeout' not in kwargs:
            kwargs['timeout'] = 120  # 2 minute timeout

        print(f"🔄 API Call: {model} | Est. tokens: {estimated_tokens:,} | Timeout: {kwargs['timeout']}s")

        with self._monitor.track_openai_completion(operation, messages, model, **kwargs) as tracker:
            max_retries = 3
            for attempt in range(1, max_retries + 1):
                try:
                    start_time = time.time()
                    print(f"   Attempt {attempt}/{max_retries} - Sending request...")

                    response = self._original.create(**kwargs)

                    duration = time.time() - start_time
                    print(f"✅ Response received in {duration:.1f}s")

                    # Log response details
                    if hasattr(response, 'usage'):
                        usage = response.usage
                        print(f"   Tokens: {usage.prompt_tokens:,} prompt + {usage.completion_tokens:,} completion = {usage.total_tokens:,} total")

                    if hasattr(response, 'choices') and response.choices:
                        content_preview = response.choices[0].message.content[:100] if response.choices[0].message.content else "No content"
                        print(f"   Response preview: {content_preview}...")

                    tracker.record_response(response)

                    # Record actual token usage
                    if hasattr(response, 'usage') and hasattr(self._monitor, '_rate_limiter'):
                        self._monitor._rate_limiter.record_usage(response.usage.total_tokens)

                    return response

                except Exception as e:
                    duration = time.time() - start_time
                    error_msg = str(e)
                    print(f"❌ API Error after {duration:.1f}s: {error_msg}")

                    # Check if it's a rate limit error
                    if "rate limit" in error_msg.lower() and attempt < max_retries:
                        print(f"🔄 Rate limit detected, retrying...")
                        if hasattr(self._monitor, '_rate_limiter'):
                            self._monitor._rate_limiter.handle_rate_limit_error(e, attempt)
                        continue
                    elif "timeout" in error_msg.lower() and attempt < max_retries:
                        print(f"⏰ Timeout detected, retrying with longer timeout...")
                        kwargs['timeout'] = kwargs.get('timeout', 120) * 1.5  # Increase timeout
                        continue
                    else:
                        print(f"💥 Fatal error on attempt {attempt}: {error_msg}")
                        # Error will be automatically recorded by the tracker context manager
                        raise

    def _estimate_tokens(self, messages: List[Dict]) -> int:
        """Estimate token count for messages."""
        # Rough estimation: 1 token ≈ 4 characters for English text
        total_chars = 0
        for message in messages:
            if isinstance(message, dict) and 'content' in message:
                total_chars += len(str(message['content']))

        # Add some overhead for system tokens, formatting, etc.
        estimated_tokens = int(total_chars / 4 * 1.3)
        return max(estimated_tokens, 100)  # Minimum estimate


class MonitoredChat:
    """Monitored wrapper for OpenAI chat API."""
    
    def __init__(self, original_chat, monitor):
        self._original = original_chat
        self._monitor = monitor
        self.completions = MonitoredChatCompletions(original_chat.completions, monitor)


class MonitoredEmbeddings:
    """Monitored wrapper for OpenAI embeddings API."""
    
    def __init__(self, original_embeddings, monitor):
        self._original = original_embeddings
        self._monitor = monitor
    
    def create(self, **kwargs):
        """Create embeddings with monitoring and rate limiting."""
        model = kwargs.get('model', 'unknown')
        input_data = kwargs.get('input', [])
        operation = f"embeddings_{model}"

        # Calculate input size and estimate tokens
        if isinstance(input_data, str):
            input_size = len(input_data)
            estimated_tokens = int(input_size / 4)  # Rough estimate
        elif isinstance(input_data, list):
            input_size = sum(len(str(item)) for item in input_data)
            estimated_tokens = int(input_size / 4)
        else:
            input_size = len(str(input_data))
            estimated_tokens = int(input_size / 4)

        # Apply rate limiting
        if hasattr(self._monitor, '_rate_limiter'):
            self._monitor._rate_limiter.wait_if_needed(estimated_tokens)

        # Add timeout to prevent freezing
        if 'timeout' not in kwargs:
            kwargs['timeout'] = 60  # 1 minute timeout for embeddings

        print(f"🔄 Embeddings: {model} | Est. tokens: {estimated_tokens:,} | Items: {len(input_data) if isinstance(input_data, list) else 1}")

        with self._monitor.track_request(operation) as tracker:
            tracker.record_request({"input": input_data, "model": model}, model)

            max_retries = 3
            for attempt in range(1, max_retries + 1):
                try:
                    start_time = time.time()
                    print(f"   Attempt {attempt}/{max_retries} - Sending embeddings request...")

                    response = self._original.create(**kwargs)

                    duration = time.time() - start_time
                    print(f"✅ Embeddings received in {duration:.1f}s")

                    # Record usage information
                    if hasattr(response, 'usage'):
                        usage = response.usage
                        print(f"   Tokens: {usage.total_tokens:,} total")
                        tracker.metrics.total_tokens = usage.total_tokens
                        tracker.metrics.prompt_tokens = usage.prompt_tokens

                        # Record actual token usage for rate limiting
                        if hasattr(self._monitor, '_rate_limiter'):
                            self._monitor._rate_limiter.record_usage(usage.total_tokens)

                    # Record response size (number of embeddings)
                    if hasattr(response, 'data'):
                        tracker.metrics.response_size = len(response.data)
                        print(f"   Embeddings generated: {len(response.data)}")

                    tracker.record_response(response)
                    return response

                except Exception as e:
                    duration = time.time() - start_time
                    error_msg = str(e)
                    print(f"❌ Embeddings Error after {duration:.1f}s: {error_msg}")

                    # Check if it's a rate limit error
                    if "rate limit" in error_msg.lower() and attempt < max_retries:
                        print(f"🔄 Rate limit detected, retrying...")
                        if hasattr(self._monitor, '_rate_limiter'):
                            self._monitor._rate_limiter.handle_rate_limit_error(e, attempt)
                        continue
                    elif "timeout" in error_msg.lower() and attempt < max_retries:
                        print(f"⏰ Timeout detected, retrying with longer timeout...")
                        kwargs['timeout'] = kwargs.get('timeout', 60) * 1.5
                        continue
                    else:
                        print(f"💥 Fatal embeddings error on attempt {attempt}: {error_msg}")
                        # Error will be automatically recorded by the tracker context manager
                        raise


class MonitoredOpenAIClient:
    """
    Monitored wrapper for OpenAI client that automatically tracks all API calls.
    
    This class provides the same interface as the standard OpenAI client but
    adds comprehensive monitoring of all API interactions.
    """
    
    def __init__(self, api_key: Optional[str] = None,
                 log_level: LogLevel = LogLevel.STANDARD,
                 output_file: Optional[Path] = None,
                 enable_rate_limiting: bool = True,
                 **kwargs):
        """
        Initialize monitored OpenAI client.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            log_level: Monitoring detail level
            output_file: Optional file to write monitoring data
            enable_rate_limiting: Whether to enable automatic rate limiting
            **kwargs: Additional arguments passed to OpenAI client
        """
        # Initialize the original OpenAI client
        if api_key is None:
            api_key = os.getenv('OPENAI_API_KEY')

        if not api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")

        self._client = OpenAI(api_key=api_key, **kwargs)
        self._monitor = get_llm_monitor(log_level, output_file)

        # Add rate limiter if enabled
        if enable_rate_limiting:
            self._monitor._rate_limiter = RateLimiter()
            print(f"🛡️  Rate limiting enabled: {self._monitor._rate_limiter.requests_per_minute} req/min, {self._monitor._rate_limiter.tokens_per_minute} tokens/min")

        # Create monitored wrappers
        self.chat = MonitoredChat(self._client.chat, self._monitor)
        self.embeddings = MonitoredEmbeddings(self._client.embeddings, self._monitor)

        # Pass through other attributes
        for attr in ['models', 'files', 'fine_tuning', 'images', 'audio', 'moderations']:
            if hasattr(self._client, attr):
                setattr(self, attr, getattr(self._client, attr))
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get comprehensive monitoring statistics."""
        return self._monitor.get_metrics_summary()
    
    def get_operation_stats(self, operation: str) -> Dict[str, Any]:
        """Get statistics for a specific operation."""
        return self._monitor.get_metrics_summary(operation=operation)
    
    def export_monitoring_data(self, output_path: Path, format: str = "json"):
        """Export monitoring data to file."""
        self._monitor.export_metrics(output_path, format)
    
    def clear_monitoring_data(self):
        """Clear all monitoring data."""
        self._monitor.clear_metrics()
    
    def set_log_level(self, log_level: LogLevel):
        """Change monitoring log level."""
        self._monitor.log_level = log_level


def create_monitored_client(api_key: Optional[str] = None,
                          log_level: LogLevel = LogLevel.STANDARD,
                          output_dir: Optional[Path] = None) -> MonitoredOpenAIClient:
    """
    Create a monitored OpenAI client with optional output directory.
    
    Args:
        api_key: OpenAI API key
        log_level: Monitoring detail level
        output_dir: Directory to write monitoring files
    
    Returns:
        MonitoredOpenAIClient instance
    """
    output_file = None
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "openai_api_calls.jsonl"
    
    return MonitoredOpenAIClient(api_key=api_key, log_level=log_level, output_file=output_file)


# Convenience function for backward compatibility
def get_monitored_openai_client(**kwargs) -> MonitoredOpenAIClient:
    """Get a monitored OpenAI client with default settings."""
    return create_monitored_client(**kwargs)


class OpenAIUsageTracker:
    """Utility class for tracking OpenAI usage across sessions."""
    
    def __init__(self, usage_file: Optional[Path] = None):
        self.usage_file = usage_file or Path("openai_usage.json")
        self.session_usage = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_cost_estimate": 0.0,
            "by_model": {}
        }
    
    def load_usage_history(self) -> Dict[str, Any]:
        """Load usage history from file."""
        if self.usage_file.exists():
            try:
                with open(self.usage_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def save_usage_history(self, usage_data: Dict[str, Any]):
        """Save usage history to file."""
        try:
            with open(self.usage_file, 'w') as f:
                json.dump(usage_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save usage history: {e}")
    
    def estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost based on current OpenAI pricing (approximate)."""
        # Pricing as of 2024 (subject to change)
        pricing = {
            "gpt-4": {"prompt": 0.03, "completion": 0.06},
            "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
            "gpt-3.5-turbo": {"prompt": 0.0015, "completion": 0.002},
            "text-embedding-3-large": {"prompt": 0.00013, "completion": 0},
            "text-embedding-3-small": {"prompt": 0.00002, "completion": 0},
        }
        
        # Default pricing for unknown models
        default_pricing = {"prompt": 0.01, "completion": 0.03}
        
        model_pricing = pricing.get(model, default_pricing)
        
        prompt_cost = (prompt_tokens / 1000) * model_pricing["prompt"]
        completion_cost = (completion_tokens / 1000) * model_pricing["completion"]
        
        return prompt_cost + completion_cost
    
    def track_usage(self, model: str, prompt_tokens: int, completion_tokens: int):
        """Track usage for a single API call."""
        total_tokens = prompt_tokens + completion_tokens
        cost = self.estimate_cost(model, prompt_tokens, completion_tokens)
        
        self.session_usage["total_requests"] += 1
        self.session_usage["total_tokens"] += total_tokens
        self.session_usage["total_cost_estimate"] += cost
        
        if model not in self.session_usage["by_model"]:
            self.session_usage["by_model"][model] = {
                "requests": 0,
                "tokens": 0,
                "cost_estimate": 0.0
            }
        
        self.session_usage["by_model"][model]["requests"] += 1
        self.session_usage["by_model"][model]["tokens"] += total_tokens
        self.session_usage["by_model"][model]["cost_estimate"] += cost
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session usage."""
        return self.session_usage.copy()
    
    def reset_session(self):
        """Reset session usage tracking."""
        self.session_usage = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_cost_estimate": 0.0,
            "by_model": {}
        }
