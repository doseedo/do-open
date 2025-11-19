"""
AI Chatbot Service for Music Production DAW
Integrates with OpenAI GPT-4 to provide intelligent music generation assistance
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import openai
import os
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Music AI Chatbot Service", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load OpenAI API key from environment
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY environment variable not set!")
else:
    openai.api_key = OPENAI_API_KEY
    logger.info("OpenAI API key loaded successfully")

# Pydantic models
class Message(BaseModel):
    role: str
    content: str
    timestamp: Optional[str] = None

class DAWContext(BaseModel):
    bpm: int
    key: str
    isBPMMode: bool
    totalDuration: float
    trackCount: int
    buses: List[Dict]

class ChatRequest(BaseModel):
    system_prompt: str
    daw_context: DAWContext
    message: str
    conversation_history: List[Message]

class ChatResponse(BaseModel):
    message: str
    timestamp: str
    model: str
    tokens_used: Optional[int] = None

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Music AI Chatbot",
        "status": "running",
        "openai_configured": bool(OPENAI_API_KEY)
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process chat request with OpenAI GPT-4
    
    Args:
        request: ChatRequest containing system prompt, DAW context, message, and history
        
    Returns:
        ChatResponse with AI-generated response
    """
    try:
        logger.info(f"Received chat request: {request.message[:100]}...")
        
        if not OPENAI_API_KEY:
            raise HTTPException(
                status_code=500,
                detail="OpenAI API key not configured"
            )
        
        # Build enhanced system prompt with current DAW context
        enhanced_system_prompt = f"""{request.system_prompt}

## Current DAW Session Context

The user is currently working on a music project with the following settings:
- **BPM (Tempo):** {request.daw_context.bpm}
- **Key:** {request.daw_context.key}
- **BPM Mode:** {'Enabled' if request.daw_context.isBPMMode else 'Disabled'}
- **Timeline Duration:** {request.daw_context.totalDuration} seconds
- **Total Tracks:** {request.daw_context.trackCount}

### Existing Buses/Tracks:
{json.dumps(request.daw_context.buses, indent=2)}

When providing suggestions or generating API calls, always consider this context to make relevant recommendations.
"""
        
        # Build conversation messages for OpenAI
        messages = [
            {"role": "system", "content": enhanced_system_prompt}
        ]
        
        # Add conversation history (limit to last 10 messages to manage token usage)
        history_limit = 10
        recent_history = request.conversation_history[-history_limit:] if len(request.conversation_history) > history_limit else request.conversation_history
        
        for msg in recent_history:
            if msg.role in ['user', 'assistant']:
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        # Add current user message
        messages.append({
            "role": "user",
            "content": request.message
        })
        
        logger.info(f"Calling OpenAI API with {len(messages)} messages...")
        
        # Call OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Use gpt-4-turbo for faster responses if available
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        
        # Extract response
        assistant_message = response.choices[0].message.content
        tokens_used = response.usage.total_tokens
        
        logger.info(f"OpenAI response received. Tokens used: {tokens_used}")
        
        return ChatResponse(
            message=assistant_message,
            timestamp=datetime.now().isoformat(),
            model=response.model,
            tokens_used=tokens_used
        )
        
    except openai.error.AuthenticationError as e:
        logger.error(f"OpenAI authentication error: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail="OpenAI API authentication failed. Check API key."
        )
    except openai.error.RateLimitError as e:
        logger.error(f"OpenAI rate limit error: {str(e)}")
        raise HTTPException(
            status_code=429,
            detail="OpenAI API rate limit exceeded. Please try again later."
        )
    except openai.error.APIError as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI API error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/api/chat/health")
async def health_check():
    """Check chatbot service health and OpenAI connectivity"""
    try:
        # Test OpenAI connection with a minimal request
        if OPENAI_API_KEY:
            test_response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            return {
                "status": "healthy",
                "openai_connected": True,
                "model": test_response.model
            }
        else:
            return {
                "status": "degraded",
                "openai_connected": False,
                "error": "API key not configured"
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "openai_connected": False,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090, log_level="info")
