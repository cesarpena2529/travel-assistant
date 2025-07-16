#!/usr/bin/env python3
"""
Test server using Claude Haiku via AWS Bedrock - parallel to clean_server.py
"""
print("✅ FastAPI server starting up...")
print("📦 Importing modules...")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.qdrant_service import QdrantSearchService
import boto3
import json
import openai
from dotenv import load_dotenv
import time
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

print("📋 Loading environment variables...")
load_dotenv()

print("🚀 Creating FastAPI app...")
app = FastAPI(title="Travel Assistant - Claude Haiku")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
print("🔧 Initializing services...")
qdrant_service = QdrantSearchService()
openai_client = openai.OpenAI()  # Still needed for embeddings
print("✅ Services initialized")

# Initialize Qdrant collection if it doesn't exist
print("🔗 Connecting to Qdrant at qdrant:6333")
try:
    client = QdrantClient(host="qdrant", port=6333)  # use 'qdrant' not 'localhost' inside ECS
    
    # Check if collection exists; create if not
    collections = client.get_collections().collections
    if not any(col.name == "travel_kb" for col in collections):
        print("📦 Creating travel_kb collection...")
        client.create_collection(
            collection_name="travel_kb",
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)  # use correct vector size!
        )
        print("✅ travel_kb collection created successfully")
    else:
        print("✅ travel_kb collection already exists")
except Exception as e:
    print(f"⚠️ Could not initialize Qdrant collection: {e}")
    print("   This is expected on first startup - collection will be created when needed")

# Initialize Bedrock client
try:
    bedrock_client = boto3.client(
        'bedrock-runtime',
        region_name='us-east-1'  # or your preferred region
    )
    bedrock_available = True
except Exception as e:
    print(f"Bedrock not available: {e}")
    bedrock_available = False

class ChatRequest(BaseModel):
    message: str
    max_results: int = 3

class ChatResponse(BaseModel):
    message: str
    sources: List[Dict[str, Any]]
    processing_time: Dict[str, float]  # Changed to match frontend expectations
    llm_used: str

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    print("🏥 Health check requested")
    try:
        # Basic health check - don't fail if qdrant is temporarily unavailable
        health_status = {
            "status": "healthy",
            "message": "Travel Assistant Claude Haiku API is running",
            "services": {
                "bedrock": "healthy" if bedrock_available else "unavailable",
                "openai": "healthy"
            }
        }
        
        # Try to check qdrant, but don't fail health check if it's down
        try:
            qdrant_health = qdrant_service.health_check()
            health_status["services"]["qdrant"] = {
                "status": "healthy",
                "points": qdrant_health.get("total_points", 0)
            }
        except Exception as qdrant_error:
            print(f"⚠️ Qdrant health check failed: {qdrant_error}")
            health_status["services"]["qdrant"] = {
                "status": "unavailable",
                "error": str(qdrant_error)
            }
        
        print("✅ Health check passed")
        return health_status
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}

def call_claude_haiku(user_message: str, context: str) -> str:
    """Call Claude Haiku via Bedrock with pure prompting approach"""
    if not bedrock_available:
        raise Exception("Bedrock not available")
    
    prompt = f"""You are an expert travel assistant. Analyze the user's question and provide a naturally-sized response that perfectly matches their intent:

🎯 INTELLIGENT RESPONSE SIZING:
• Quick questions ("best beach in Bali?") → Concise, focused answer (100-200 words)
• List requests ("top 10 beaches in Europe") → Comprehensive list with brief descriptions (300-500 words)  
• Detailed inquiries ("how to plan 3 days in Amsterdam?") → Thorough, practical guide (400-600 words)
• Simple facts ("is Paris expensive?") → Brief, direct answer (50-150 words)

🚀 RESPONSE QUALITY RULES:
1. ALWAYS finish your complete thought - never cut off mid-sentence
2. Match the depth they're asking for naturally
3. Be specific and actionable, not generic
4. End with ONE highly relevant follow-up question tailored to their exact query
5. Use enthusiastic but professional tone

📍 CONTEXT-AWARE FOLLOW-UPS:
• Beach/location queries → Ask about preferences (family-friendly vs romantic, etc.)
• City questions → Ask about interests (culture, food, nightlife, history)
• Planning questions → Ask about duration, budget, or travel style
• Transportation → Ask about starting point or specific needs
• Food queries → Ask about dining style or dietary preferences

User Question: {user_message}

Available Travel Information:
{context}

Provide your naturally-sized, complete response with a specific follow-up question:"""
    
    print(f"Prompting Claude with: {prompt}")  # Debugging line
    response = bedrock_client.invoke_model(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,  # Generous limit - Claude will naturally stop at appropriate length
            "messages": [
                {
                    "role": "user",
                    "content": prompt.replace("Human: ", "")
                }
            ]
        }),
        contentType="application/json",
        accept="application/json",
    )
    
    response_body = json.loads(response['body'].read())
    print(f"Claude response: {response_body}")  # Debugging line
    
    if 'content' in response_body and len(response_body['content']) > 0:
        return response_body['content'][0]['text'].strip()
    else:
        raise Exception("Invalid response from Claude")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat messages"""
    start_time = time.time()
    
    try:
        # Step 1: Get embedding for search
        embedding_start = time.time()
        embedding_response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=request.message
        )
        query_vector = embedding_response.data[0].embedding
        
        # Step 2: Search Qdrant for relevant context
        search_response = qdrant_service.search_similar(
            query_vector=query_vector,
            limit=request.max_results
        )
        search_results = search_response.get("results", [])
        search_time = time.time() - embedding_start
        
        # Step 3: Prepare context for Claude
        context = ""
        for i, result in enumerate(search_results[:3]):
            content = result.get('content', '')[:500]  # Limit content length
            location = result.get('location', 'Unknown')
            context += f"\\n\\nLocation: {location}\\n{content}"
        
        # Step 4: Try Claude first, fallback to OpenAI
        llm_start = time.time()
        llm_used = "Claude Haiku (Bedrock)"
        try:
            if bedrock_available:
                response_text = call_claude_haiku(request.message, context)
            else:
                raise Exception("Bedrock not available, using OpenAI fallback")
        except Exception as claude_error:
            print(f"Claude failed: {claude_error}, falling back to OpenAI")
            llm_used = "GPT-4o-mini (fallback)"
            
            # Fallback to OpenAI with pure prompting approach
            chat_response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": """You are an expert travel assistant. Analyze the user's question and provide a naturally-sized response:

🎯 INTELLIGENT RESPONSE SIZING:
• Quick questions → Concise, focused (100-200 words)
• List requests → Comprehensive with details (300-500 words)  
• Detailed inquiries → Thorough, practical guides (400-600 words)
• Simple facts → Brief, direct answers (50-150 words)

ALWAYS finish your complete thought and end with ONE relevant follow-up question."""
                    },
                    {
                        "role": "user", 
                        "content": f"User question: {request.message}\\n\\nRelevant travel information:{context}\\n\\nProvide your naturally-sized, complete response with a specific follow-up question."
                    }
                ],
                max_tokens=1000,  # Generous limit - let AI naturally determine length
                temperature=0.7
            )
            response_text = chat_response.choices[0].message.content
        
        llm_time = time.time() - llm_start
        total_time = time.time() - start_time
        
        return ChatResponse(
            message=response_text,
            sources=[],  # Remove sources for faster response and cleaner UI
            processing_time={
                "total_time": total_time,
                "search_time": search_time,
                "llm_time": llm_time
            },
            llm_used=llm_used
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.post("/setup")
async def setup_qdrant():
    """Setup Qdrant collection and load data - one-time operation"""
    try:
        import json
        from qdrant_client.models import Distance, VectorParams, PointStruct
        
        collection_name = "travel_kb"
        data_path = "data/kb_flattened.json"
        
        # Check if collection already exists
        try:
            collection_info = qdrant_service.client.get_collection(collection_name)
            return {
                "status": "already_exists",
                "message": f"Collection '{collection_name}' already exists with {collection_info.points_count} points"
            }
        except:
            # Collection doesn't exist, we'll create it
            pass
        
        # Create collection
        qdrant_service.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=1536,  # OpenAI text-embedding-3-small dimension
                distance=Distance.COSINE
            )
        )
        
        # Load and populate data
        with open(data_path, 'r', encoding='utf-8') as f:
            kb_data = json.load(f)
        
        points = []
        for idx, item in enumerate(kb_data):
            embedding = item.get('embedding')
            if not embedding or len(embedding) != 1536:
                continue
                
            point = PointStruct(
                id=idx,
                vector=embedding,
                payload={
                    'location_id': item.get('location_id'),
                    'destination_id': item.get('destination_id'), 
                    'title': item.get('title'),
                    'content': item.get('content'),
                    'country': item.get('country'),
                    'location': item.get('location'),
                    'type': item.get('type'),
                    'category': item.get('category'),
                    'tags': item.get('tags', []),
                    'url': item.get('url'),
                    'chunk_index': item.get('chunk_index', 0)
                }
            )
            points.append(point)
        
        # Upload points in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            qdrant_service.client.upsert(collection_name=collection_name, points=batch)
        
        # Verify
        collection_info = qdrant_service.client.get_collection(collection_name)
        
        return {
            "status": "success",
            "message": f"Successfully created collection '{collection_name}' with {collection_info.points_count} points",
            "total_points": collection_info.points_count
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Setup failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Travel Assistant with Claude Haiku...")
    print("📍 Server: http://localhost:8000")
    print(f"🤖 Bedrock available: {bedrock_available}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
