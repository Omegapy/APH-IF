# =========================================================================
# File: llm.py
# Project: APH-IF Technology Framework
#          Advanced Parallel Hybrid - Intelligent Fusion System
# Author: Alexander Ricciardi
# Date: 2025-08-05
# File Path: backend/llm.py
# =========================================================================

"""
Large Language Model Client for APH-IF

Provides unified interface for multiple LLM providers including OpenAI,
Google Gemini, and other language models. Handles API calls, response
processing, and error handling for intelligent fusion operations.
"""

import logging
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

# Local imports
from .config import get_config
from .circuit_breaker import get_circuit_breaker_manager

# =========================================================================
# LLM Provider Enumeration
# =========================================================================
class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    GEMINI = "gemini"
    MOCK = "mock"

# =========================================================================
# LLM Client
# =========================================================================
class LLMClient:
    """
    Unified LLM client supporting multiple providers
    
    Provides consistent interface for generating responses from different
    language model providers with fault tolerance and circuit breaker protection.
    """
    
    def __init__(self, provider: str = "openai"):
        self.config = get_config()
        self.logger = logging.getLogger("llm_client")
        
        # Provider configuration
        self.provider = LLMProvider(provider)
        self.api_client = None
        
        # Circuit breaker for fault tolerance
        self.cb_manager = get_circuit_breaker_manager()
        self.circuit_breaker = self.cb_manager.create_circuit_breaker(
            f"llm_{provider}",
            failure_threshold=3,
            recovery_timeout=60,
            timeout=30
        )
        
        # Performance tracking
        self.total_requests = 0
        self.total_tokens = 0
        self.total_response_time = 0.0
        
        self.initialized = False
    
    async def initialize(self):
        """Initialize the LLM client"""
        try:
            if self.provider == LLMProvider.OPENAI:
                await self._initialize_openai()
            elif self.provider == LLMProvider.GEMINI:
                await self._initialize_gemini()
            else:
                await self._initialize_mock()
            
            self.initialized = True
            self.logger.info(f"LLM client initialized for provider: {self.provider.value}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM client: {e}")
            # Fallback to mock provider
            self.provider = LLMProvider.MOCK
            await self._initialize_mock()
            self.initialized = True
            self.logger.warning("Fallback to mock LLM provider")
    
    async def generate_response(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.1,
        system_message: Optional[str] = None
    ) -> str:
        """
        Generate response from LLM
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum response tokens
            temperature: Response randomness (0.0 to 1.0)
            system_message: Optional system message
            
        Returns:
            Generated response text
        """
        if not self.initialized:
            await self.initialize()
        
        start_time = asyncio.get_event_loop().time()
        self.total_requests += 1
        
        try:
            # Use circuit breaker for fault tolerance
            response = await self.circuit_breaker.call_async(
                self._generate_response_internal,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                system_message=system_message
            )
            
            response_time = asyncio.get_event_loop().time() - start_time
            self.total_response_time += response_time
            
            # Estimate token count (rough approximation)
            estimated_tokens = len(response.split()) * 1.3
            self.total_tokens += estimated_tokens
            
            self.logger.info(
                f"Generated response: {len(response)} chars, "
                f"~{estimated_tokens:.0f} tokens in {response_time:.2f}s"
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"LLM response generation failed: {e}")
            return f"Error generating response: {str(e)}"
    
    async def _generate_response_internal(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        system_message: Optional[str]
    ) -> str:
        """Internal response generation method"""
        
        if self.provider == LLMProvider.OPENAI:
            return await self._generate_openai_response(
                prompt, max_tokens, temperature, system_message
            )
        elif self.provider == LLMProvider.GEMINI:
            return await self._generate_gemini_response(
                prompt, max_tokens, temperature, system_message
            )
        else:
            return await self._generate_mock_response(
                prompt, max_tokens, temperature, system_message
            )
    
    async def _initialize_openai(self):
        """Initialize OpenAI client"""
        try:
            # In production, initialize actual OpenAI client
            # import openai
            # self.api_client = openai.AsyncOpenAI(
            #     api_key=self.config.llm.openai_api_key
            # )
            
            if not self.config.llm.openai_api_key:
                raise ValueError("OpenAI API key not configured")
            
            self.logger.info("OpenAI client initialized (mock)")
            
        except Exception as e:
            self.logger.error(f"OpenAI initialization failed: {e}")
            raise
    
    async def _initialize_gemini(self):
        """Initialize Google Gemini client"""
        try:
            # In production, initialize actual Gemini client
            # import google.generativeai as genai
            # genai.configure(api_key=self.config.llm.gemini_api_key)
            # self.api_client = genai.GenerativeModel(self.config.llm.gemini_model)
            
            if not self.config.llm.gemini_api_key:
                raise ValueError("Gemini API key not configured")
            
            self.logger.info("Gemini client initialized (mock)")
            
        except Exception as e:
            self.logger.error(f"Gemini initialization failed: {e}")
            raise
    
    async def _initialize_mock(self):
        """Initialize mock LLM client"""
        self.api_client = "mock_client"
        self.logger.info("Mock LLM client initialized")
    
    async def _generate_openai_response(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        system_message: Optional[str]
    ) -> str:
        """Generate response using OpenAI API"""
        
        # In production, make actual OpenAI API call
        # messages = []
        # if system_message:
        #     messages.append({"role": "system", "content": system_message})
        # messages.append({"role": "user", "content": prompt})
        # 
        # response = await self.api_client.chat.completions.create(
        #     model=self.config.llm.openai_model,
        #     messages=messages,
        #     max_tokens=max_tokens,
        #     temperature=temperature
        # )
        # 
        # return response.choices[0].message.content
        
        # Mock response for demonstration
        await asyncio.sleep(0.1)  # Simulate API delay
        return f"Mock OpenAI response to: {prompt[:50]}..."
    
    async def _generate_gemini_response(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        system_message: Optional[str]
    ) -> str:
        """Generate response using Google Gemini API"""
        
        # In production, make actual Gemini API call
        # full_prompt = prompt
        # if system_message:
        #     full_prompt = f"{system_message}\n\n{prompt}"
        # 
        # response = await self.api_client.generate_content_async(
        #     full_prompt,
        #     generation_config={
        #         "max_output_tokens": max_tokens,
        #         "temperature": temperature
        #     }
        # )
        # 
        # return response.text
        
        # Mock response for demonstration
        await asyncio.sleep(0.1)  # Simulate API delay
        return f"Mock Gemini response to: {prompt[:50]}..."
    
    async def _generate_mock_response(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        system_message: Optional[str]
    ) -> str:
        """Generate mock response for demonstration"""
        
        await asyncio.sleep(0.05)  # Simulate processing time
        
        # Generate contextual mock response based on prompt content
        prompt_lower = prompt.lower()
        
        if "fusion" in prompt_lower or "combine" in prompt_lower:
            return """Based on the provided search results, here is a comprehensive fused response:

The Advanced Parallel HybridRAG (APH) technology represents a significant advancement in RAG systems by executing VectorRAG and GraphRAG searches concurrently rather than sequentially. This parallel approach, combined with Intelligent Fusion (IF) algorithms, creates more comprehensive and contextually rich responses.

Key findings from the search results:
1. VectorRAG search provides semantic similarity matching
2. GraphRAG search enables relationship-based knowledge traversal
3. Intelligent fusion combines results using LLM-powered algorithms
4. The parallel execution significantly improves response time and quality

This approach differs from traditional RAG by leveraging the strengths of both vector and graph-based retrieval simultaneously."""

        elif "vector" in prompt_lower:
            return "Vector search results indicate semantic similarity matches based on embedding representations of the query content."
        
        elif "graph" in prompt_lower:
            return "Graph search results show relationship-based connections and knowledge graph traversal patterns relevant to the query."
        
        else:
            return f"Mock LLM response addressing the query about: {prompt[:100]}..."
    
    def get_stats(self) -> Dict[str, Any]:
        """Get LLM client statistics"""
        avg_response_time = self.total_response_time / max(self.total_requests, 1)
        avg_tokens_per_request = self.total_tokens / max(self.total_requests, 1)
        
        return {
            "provider": self.provider.value,
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "total_response_time": self.total_response_time,
            "average_response_time": avg_response_time,
            "average_tokens_per_request": avg_tokens_per_request,
            "initialized": self.initialized,
            "circuit_breaker_stats": self.circuit_breaker.get_stats()
        }
    
    def get_health_status(self) -> str:
        """Get health status"""
        if not self.initialized:
            return "not_initialized"
        elif self.provider == LLMProvider.MOCK:
            return "mock_mode"
        elif self.circuit_breaker.state.value == "open":
            return "circuit_open"
        else:
            return "healthy"

# =========================================================================
# Global LLM Client Instance
# =========================================================================
_llm_client: Optional[LLMClient] = None

async def get_llm_client(provider: Optional[str] = None) -> LLMClient:
    """Get the global LLM client instance"""
    global _llm_client
    
    if _llm_client is None:
        config = get_config()
        provider = provider or config.llm.default_provider
        _llm_client = LLMClient(provider)
        await _llm_client.initialize()
    
    return _llm_client
