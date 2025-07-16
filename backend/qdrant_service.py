"""
Qdrant search service for travel assistant
Replaces OpenSearch with local Qdrant vector database
"""

import os
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchText
import time
import logging

logger = logging.getLogger(__name__)

class QdrantSearchService:
    """Search service using Qdrant vector database"""
    
    def __init__(self, host: str = None, port: int = 6333, collection_name: str = "travel_kb"):
        """Initialize Qdrant client with Docker-aware host detection"""
        # 🐳 DOCKER MAGIC: Auto-detect if we're running in a container
        self.host = host or os.getenv('QDRANT_HOST', 'localhost')
        self.port = int(os.getenv('QDRANT_PORT', port))
        self.collection_name = collection_name
        self.client = None
        
        print(f"🔗 Connecting to Qdrant at {self.host}:{self.port}")
        self._connect()
    
    def _connect(self):
        """Connect to Qdrant"""
        try:
            self.client = QdrantClient(host=self.host, port=self.port)
            # Test connection
            collections = self.client.get_collections()
            logger.info(f"Connected to Qdrant at {self.host}:{self.port}")
            logger.info(f"Available collections: {[c.name for c in collections.collections]}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """Check if Qdrant is healthy and return status"""
        try:
            collections = self.client.get_collections()
            available_collections = [c.name for c in collections.collections]
            
            # Check if our target collection exists
            if self.collection_name in available_collections:
                collection_info = self.client.get_collection(self.collection_name)
                return {
                    "status": "healthy",
                    "service": "qdrant",
                    "host": f"{self.host}:{self.port}",
                    "collection": self.collection_name,
                    "total_points": collection_info.points_count,
                    "vector_count": collection_info.vectors_count,
                    "available_collections": available_collections,
                    "collection_exists": True
                }
            else:
                return {
                    "status": "healthy",
                    "service": "qdrant",
                    "host": f"{self.host}:{self.port}",
                    "collection": self.collection_name,
                    "available_collections": available_collections,
                    "collection_exists": False,
                    "message": f"Collection '{self.collection_name}' not found. Use /setup to initialize."
                }
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return {
                "status": "unhealthy",
                "service": "qdrant",
                "error": str(e)
            }
    
    def search_similar(self, query_vector: List[float], limit: int = 5, 
                      country_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents using vector similarity
        
        Args:
            query_vector: The embedding vector to search with
            limit: Number of results to return
            country_filter: Optional country filter
            
        Returns:
            List of similar documents with metadata
        """
        start_time = time.time()
        
        try:
            # Build filter if country specified
            query_filter = None
            if country_filter:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="country",
                            match=MatchValue(value=country_filter)
                        )
                    ]
                )
            
            # Perform vector search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False  # Don't return vectors to save bandwidth
            )
            
            search_time = time.time() - start_time
            
            # Transform results to match expected format
            results = []
            for result in search_results:
                payload = result.payload
                doc = {
                    "id": payload.get("chunk_id", ""),
                    "score": float(result.score),
                    "content": payload.get("text", ""),
                    "title": payload.get("title", ""),
                    "location": payload.get("location", ""),
                    "country": payload.get("country", ""),
                    "region": payload.get("region", ""),
                    "coordinates": payload.get("coordinates", ""),
                    "latitude": payload.get("latitude"),
                    "longitude": payload.get("longitude"),
                    "population": payload.get("population"),
                    "type": payload.get("type", ""),
                    "weather": payload.get("weather", {}),
                    "chunk_length": payload.get("chunk_length", 0)
                }
                results.append(doc)
            
            logger.info(f"Qdrant search completed in {search_time:.3f}s, returned {len(results)} results")
            
            return {
                "results": results,
                "search_time": search_time,
                "total_results": len(results),
                "service": "qdrant"
            }
            
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            raise Exception(f"Search failed: {e}")
    
    def search_with_location_boost(self, query_vector: List[float], query_text: str = "", 
                                  limit: int = 5, location_boost: float = 2.0) -> List[Dict[str, Any]]:
        """
        Enhanced search that boosts results from locations mentioned in the query
        
        Args:
            query_vector: The embedding vector to search with
            query_text: Original query text to extract location names
            limit: Number of results to return
            location_boost: Multiplier for boosting location-relevant results
            
        Returns:
            List of similar documents with location boosting applied
        """
        start_time = time.time()
        
        try:
            # Extract potential location names from query
            query_lower = query_text.lower()
            location_keywords = self._extract_location_keywords(query_lower)
            
            # Get more results than needed for re-ranking
            search_limit = min(limit * 3, 50)  # Get 3x results for re-ranking
            
            # Perform vector search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=search_limit,
                with_payload=True,
                with_vectors=False
            )
            
            # Re-rank results with location boosting
            boosted_results = []
            for result in search_results:
                payload = result.payload
                base_score = float(result.score)
                
                # Check if this result is from a mentioned location
                location_match = self._check_location_match(payload, location_keywords)
                
                # Apply location boost
                final_score = base_score * location_boost if location_match else base_score
                
                doc = {
                    "id": payload.get("chunk_id", ""),
                    "score": final_score,
                    "base_score": base_score,
                    "location_boosted": location_match,
                    "content": payload.get("text", ""),
                    "title": payload.get("title", ""),
                    "location": payload.get("location", ""),
                    "country": payload.get("country", ""),
                    "region": payload.get("region", ""),
                    "coordinates": payload.get("coordinates", ""),
                    "latitude": payload.get("latitude"),
                    "longitude": payload.get("longitude"),
                    "population": payload.get("population"),
                    "type": payload.get("type", ""),
                    "weather": payload.get("weather", {}),
                    "chunk_length": payload.get("chunk_length", 0)
                }
                boosted_results.append(doc)
            
            # Sort by boosted score and take top results
            boosted_results.sort(key=lambda x: x["score"], reverse=True)
            final_results = boosted_results[:limit]
            
            search_time = time.time() - start_time
            
            logger.info(f"Location-boosted search completed in {search_time:.3f}s")
            logger.info(f"Query keywords: {location_keywords}")
            logger.info(f"Boosted {sum(1 for r in final_results if r['location_boosted'])} out of {len(final_results)} results")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    def search_with_location_hybrid(self, query_vector: List[float], query_text: str = "", 
                                   limit: int = 5) -> List[Dict[str, Any]]:
        """
        Hybrid search: semantic + location filtering for better location relevance
        
        Args:
            query_vector: The embedding vector to search with
            query_text: Original query text to extract location names
            limit: Number of results to return
            
        Returns:
            List of documents combining semantic and location-filtered results
        """
        start_time = time.time()
        
        try:
            # Extract potential location and content keywords from query
            query_lower = query_text.lower()
            location_keywords = self._extract_location_keywords(query_lower)
            content_keywords = self._extract_content_keywords(query_lower)
            
            logger.info(f"Hybrid search for query: '{query_text}'")
            logger.info(f"Detected location keywords: {location_keywords}")
            logger.info(f"Detected content keywords: {content_keywords}")
            
            all_results = []
            
            # If location keywords found, get location-specific results first
            if location_keywords:
                for keyword in location_keywords:
                    # Search by exact location/title match
                    try:
                        location_filter = Filter(
                            should=[  # OR condition for multiple fields
                                FieldCondition(key="title", match=MatchValue(value=keyword.title())),
                                FieldCondition(key="location", match=MatchValue(value=keyword.title())),
                                FieldCondition(key="country", match=MatchValue(value=keyword.title()))
                            ]
                        )
                        
                        location_results = self.client.search(
                            collection_name=self.collection_name,
                            query_vector=query_vector,
                            query_filter=location_filter,
                            limit=limit,
                            with_payload=True,
                            with_vectors=False
                        )
                        
                        for result in location_results:
                            doc = self._format_search_result(result, location_boosted=True)
                            all_results.append(doc)
                        
                        logger.info(f"Found {len(location_results)} results for location '{keyword}'")
                        
                    except Exception as e:
                        logger.warning(f"Location filter search failed for '{keyword}': {e}")
                        continue
            
            # If we have good location results, return them
            if all_results:
                # Remove duplicates and sort by score
                seen_ids = set()
                unique_results = []
                for result in all_results:
                    result_id = result.get('id', '')
                    if result_id not in seen_ids:
                        seen_ids.add(result_id)
                        unique_results.append(result)
                
                unique_results.sort(key=lambda x: x.get('score', 0), reverse=True)
                final_results = unique_results[:limit]
                
                logger.info(f"Returning {len(final_results)} location-specific results")
                return final_results
            
            # If no location keywords but we have content keywords, try content-based search
            elif content_keywords:
                logger.info(f"No location found, searching by content keywords: {content_keywords}")
                content_results = self._search_by_content_keywords(query_vector, content_keywords, limit)
                
                if content_results:
                    # Remove duplicates and sort by score
                    seen_ids = set()
                    unique_results = []
                    for result in content_results:
                        result_id = result.get('id', '')
                        if result_id not in seen_ids:
                            seen_ids.add(result_id)
                            unique_results.append(result)
                    
                    unique_results.sort(key=lambda x: x.get('score', 0), reverse=True)
                    final_results = unique_results[:limit]
                    
                    logger.info(f"Returning {len(final_results)} content-specific results")
                    return final_results
            
            # Final fallback to regular semantic search
            logger.info("No specific keywords found, falling back to semantic search")
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            results = []
            for result in search_results:
                doc = self._format_search_result(result, location_boosted=False)
                results.append(doc)
            
            search_time = time.time() - start_time
            logger.info(f"Hybrid search completed in {search_time:.3f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise

    def _extract_location_keywords(self, query_text: str) -> List[str]:
        """Extract potential location names from query text"""
        # Common location indicators - keeping a focused list for better performance
        location_indicators = [
            'amsterdam', 'paris', 'london', 'berlin', 'rome', 'madrid', 'barcelona',
            'vienna', 'prague', 'budapest', 'warsaw', 'stockholm', 'copenhagen',
            'oslo', 'helsinki', 'dublin', 'edinburgh', 'brussels', 'zurich',
            'geneva', 'milan', 'florence', 'venice', 'naples', 'lisbon', 'porto',
            'athens', 'istanbul', 'moscow', 'dubai', 'tokyo', 'kyoto', 'osaka',
            'singapore', 'bangkok', 'kuala lumpur', 'jakarta', 'manila', 'seoul',
            'busan', 'taipei', 'hong kong', 'macau', 'shanghai', 'beijing',
            'mumbai', 'delhi', 'bangalore', 'kolkata', 'chennai', 'hyderabad',
            'sydney', 'melbourne', 'brisbane', 'perth', 'auckland', 'wellington',
            'toronto', 'vancouver', 'montreal', 'ottawa', 'calgary', 'edmonton',
            'new york', 'los angeles', 'chicago', 'houston', 'phoenix', 'philadelphia',
            'san antonio', 'san diego', 'dallas', 'san jose', 'austin', 'jacksonville',
            'san francisco', 'columbus', 'charlotte', 'fort worth', 'indianapolis',
            'seattle', 'denver', 'washington', 'boston', 'el paso', 'detroit',
            'nashville', 'portland', 'oklahoma city', 'las vegas', 'louisville',
            'baltimore', 'milwaukee', 'albuquerque', 'tucson', 'fresno', 'sacramento',
            'mesa', 'kansas city', 'atlanta', 'long beach', 'colorado springs',
            'raleigh', 'miami', 'virginia beach', 'omaha', 'oakland', 'minneapolis',
            'tulsa', 'arlington', 'new orleans', 'wichita', 'cleveland', 'tampa',
            'bakersfield', 'honolulu', 'anaheim', 'santa ana', 'corpus christi',
            'riverside', 'lexington', 'stockton', 'st. louis', 'saint paul',
            'cincinnati', 'pittsburgh', 'greensboro', 'newark', 'plano', 'henderson',
            'lincoln', 'buffalo', 'fort wayne', 'jersey city', 'chula vista',
            'orlando', 'norfolk', 'chandler', 'laredo', 'madison', 'durham',
            'lubbock', 'winston-salem', 'garland', 'glendale', 'hialeah', 'reno',
            'baton rouge', 'irvine', 'chesapeake', 'irving', 'scottsdale', 'fremont'
        ]
        
        # Find location keywords in the query (case-insensitive)
        found_keywords = []
        query_lower = query_text.lower()
        for keyword in location_indicators:
            if keyword in query_lower:
                found_keywords.append(keyword)
        
        return found_keywords

    def _check_location_match(self, payload: Dict, location_keywords: List[str]) -> bool:
        """Check if a document payload matches any of the location keywords"""
        if not location_keywords:
            return False
            
        # Check various location fields, handle None values
        location_fields = [
            str(payload.get("location", "") or "").lower(),
            str(payload.get("title", "") or "").lower(), 
            str(payload.get("country", "") or "").lower(),
            str(payload.get("region", "") or "").lower()
        ]
        
        for keyword in location_keywords:
            for field in location_fields:
                if keyword in field:
                    return True
                    
        return False

    def _extract_content_keywords(self, query: str) -> List[str]:
        """Extract content/topic keywords from query text"""
        content_keywords = [
            'beach', 'beaches', 'coast', 'coastal', 'seaside', 'shore', 'ocean', 'sea',
            'museum', 'museums', 'art', 'gallery', 'galleries', 'culture', 'cultural',
            'restaurant', 'restaurants', 'food', 'dining', 'cuisine', 'local food',
            'park', 'parks', 'garden', 'gardens', 'nature', 'hiking', 'walking',
            'church', 'churches', 'cathedral', 'cathedrals', 'religious', 'historic',
            'shopping', 'shops', 'market', 'markets', 'boutique', 'mall',
            'nightlife', 'bars', 'clubs', 'entertainment', 'music', 'festival',
            'attraction', 'attractions', 'sightseeing', 'tourist', 'landmark',
            'architecture', 'building', 'castle', 'palace', 'monument',
            'transport', 'transportation', 'metro', 'bus', 'train', 'tram',
            'hotel', 'hotels', 'accommodation', 'stay', 'lodging'
        ]
        
        found_keywords = []
        query_lower = query.lower()
        
        for keyword in content_keywords:
            if keyword in query_lower:
                found_keywords.append(keyword)
        
        return found_keywords

    def _search_by_content_keywords(self, query_vector: List[float], content_keywords: List[str], limit: int = 10) -> List[Dict[str, Any]]:
        """Search for content using keyword filtering on text content"""
        results = []
        
        for keyword in content_keywords:
            try:
                # Use Qdrant's text filtering capability
                keyword_filter = Filter(
                    must=[
                        FieldCondition(
                            key="text",
                            match=MatchText(text=keyword)
                        )
                    ]
                )
                
                search_results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    query_filter=keyword_filter,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False
                )
                
                for result in search_results:
                    doc = self._format_search_result(result, location_boosted=False)
                    doc['content_matched'] = keyword
                    results.append(doc)
                    
                logger.info(f"Found {len(search_results)} results for content keyword '{keyword}'")
                
            except Exception as e:
                logger.warning(f"Content keyword search failed for '{keyword}': {e}")
                # Fallback to manual text search
                try:
                    all_results = self.client.search(
                        collection_name=self.collection_name,
                        query_vector=query_vector,
                        limit=limit * 3,  # Get more for filtering
                        with_payload=True,
                        with_vectors=False
                    )
                    
                    # Manual filtering
                    for result in all_results:
                        text_content = result.payload.get("text", "").lower()
                        if keyword in text_content:
                            doc = self._format_search_result(result, location_boosted=False)
                            doc['content_matched'] = keyword
                            results.append(doc)
                            
                except Exception as e2:
                    logger.error(f"Fallback content search also failed for '{keyword}': {e2}")
                    continue
        
        return results

    def _format_search_result(self, result, location_boosted: bool = False) -> Dict[str, Any]:
        """Format a Qdrant search result into standard format"""
        payload = result.payload
        return {
            "id": payload.get("chunk_id", ""),
            "score": float(result.score),
            "location_boosted": location_boosted,
            "content": payload.get("text", ""),
            "title": payload.get("title", ""),
            "location": payload.get("location", ""),
            "country": payload.get("country", ""),
            "region": payload.get("region", ""),
            "coordinates": payload.get("coordinates", ""),
            "latitude": payload.get("latitude"),
            "longitude": payload.get("longitude"),
            "population": payload.get("population"),
            "type": payload.get("type", ""),
            "weather": payload.get("weather", {}),
            "chunk_length": payload.get("chunk_length", 0)
        }

    # Keep the original search method for backward compatibility
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            # Get a sample of points to analyze
            sample_points = self.client.scroll(
                collection_name=self.collection_name,
                limit=100
            )
            
            # Count unique countries/locations
            countries = set()
            locations = set()
            for point in sample_points[0]:
                if point.payload.get("country"):
                    countries.add(point.payload["country"])
                if point.payload.get("location"):
                    locations.add(point.payload["location"])
            
            return {
                "total_points": collection_info.points_count,
                "vector_count": collection_info.vectors_count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance": collection_info.config.params.vectors.distance.name,
                "sample_countries": list(countries)[:10],  # First 10
                "sample_locations": list(locations)[:10],   # First 10
                "total_sample_countries": len(countries),
                "total_sample_locations": len(locations)
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}
    
    def close(self):
        """Close the connection"""
        if self.client:
            # Qdrant client doesn't need explicit closing
            self.client = None
            logger.info("Qdrant connection closed")

# Global service instance
_qdrant_service = None

def initialize_qdrant_service():
    """Initialize the global Qdrant service"""
    global _qdrant_service
    _qdrant_service = QdrantSearchService()
    logger.info("✅ Qdrant service initialized")

def get_qdrant_service() -> QdrantSearchService:
    """Get the global Qdrant service instance"""
    global _qdrant_service
    if _qdrant_service is None:
        initialize_qdrant_service()
    return _qdrant_service
