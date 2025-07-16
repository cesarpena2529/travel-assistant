"""
Data models for the travel assistant API
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class SearchQuery(BaseModel):
    query: str
    max_results: Optional[int] = 10
    filters: Optional[Dict[str, Any]] = None

class SearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 5
    country_filter: Optional[str] = None

class SearchResult(BaseModel):
    score: float
    title: str
    content: str
    country: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    attractions: Optional[List[Dict]] = None
    weather: Optional[Dict] = None

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    query: str
    total_results: int
    processing_time: Dict[str, float]

class HealthResponse(BaseModel):
    status: str
    message: str
    services: Dict[str, str]

class ChatMessage(BaseModel):
    message: str
    max_results: Optional[int] = 5

class ChatRequest(BaseModel):
    message: str
    context_limit: Optional[int] = 5
    max_tokens: Optional[int] = 1000
    country_filter: Optional[str] = None

class ChatResponse(BaseModel):
    message: str
    sources: List[Dict[str, Any]]
    processing_time: Dict[str, float]
    model: Optional[str] = None
    tokens: Optional[Dict[str, int]] = None

class SearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 5
    country_filter: Optional[str] = None

class SearchResponse(BaseModel):
    results: List[SearchResult]
    query: str
    total_results: int
    processing_time: Dict[str, float]

class ChatRequest(BaseModel):
    message: str
    context_limit: Optional[int] = 5
    max_tokens: Optional[int] = 1000
    country_filter: Optional[str] = None
