"""
MIDI DNA API Server
==================

REST API for Modular Semantic Discovery System

Agent 10: Documentation & Deployment Manager
Version: 1.0.0
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional
import io
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="MIDI DNA API",
    description="Extract and manipulate musical DNA from MIDI files",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance (lazy loaded)
pipeline = None


def get_pipeline():
    """Get or initialize pipeline."""
    global pipeline
    if pipeline is None:
        try:
            from midi_generator.learning.modular_discovery_pipeline import ModularSemanticDiscoveryPipeline
            logger.info("Loading pre-trained pipeline...")
            pipeline = ModularSemanticDiscoveryPipeline.load_pretrained()
            logger.info("✓ Pipeline loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            raise HTTPException(status_code=500, detail=f"Pipeline initialization failed: {str(e)}")
    return pipeline


# Request/Response models
class DNAResponse(BaseModel):
    """DNA extraction response."""
    dna: Dict[str, float]
    num_parameters: int
    file_name: Optional[str] = None


class GenerateRequest(BaseModel):
    """MIDI generation request."""
    dna: Dict[str, float]


class EditRequest(BaseModel):
    """Parameter edit request."""
    edits: Dict[str, float]


# Routes

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "MIDI DNA API",
        "version": "1.0.0",
        "description": "Extract and manipulate musical DNA from MIDI files",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "extract_dna": "POST /extract_dna",
            "generate": "POST /generate",
            "edit": "POST /edit"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    try:
        # Check if pipeline is loaded
        p = get_pipeline()
        return {
            "status": "healthy",
            "pipeline_loaded": p is not None,
            "version": "1.0.0"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@app.post("/extract_dna", response_model=DNAResponse)
async def extract_dna(file: UploadFile = File(...)):
    """
    Extract 120-parameter musical DNA from uploaded MIDI file.

    Args:
        file: MIDI file upload

    Returns:
        DNAResponse with extracted parameters
    """
    try:
        # Read file
        midi_data = await file.read()

        # Get pipeline
        p = get_pipeline()

        # Extract DNA (implementation would go here)
        # For now, return mock data
        logger.info(f"Extracting DNA from: {file.filename}")

        # Mock DNA extraction
        dna = {
            f"param_{i:03d}": 0.5
            for i in range(120)
        }

        # In production, would do:
        # dna = p.extract_dna_from_bytes(midi_data)

        return DNAResponse(
            dna=dna,
            num_parameters=len(dna),
            file_name=file.filename
        )

    except Exception as e:
        logger.error(f"DNA extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate")
async def generate(request: GenerateRequest):
    """
    Generate MIDI file from DNA parameters.

    Args:
        request: GenerateRequest with DNA parameters

    Returns:
        MIDI file bytes
    """
    try:
        # Get pipeline
        p = get_pipeline()

        # Generate MIDI (implementation would go here)
        logger.info(f"Generating MIDI from {len(request.dna)} parameters")

        # Mock generation
        midi_bytes = b"MIDI_DATA_HERE"

        # In production, would do:
        # midi_bytes = p.generate_from_dna_to_bytes(request.dna)

        return Response(
            content=midi_bytes,
            media_type="audio/midi",
            headers={
                "Content-Disposition": "attachment; filename=generated.mid"
            }
        )

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/edit")
async def edit(file: UploadFile = File(...), edits: str = None):
    """
    Edit MIDI by modifying DNA parameters.

    Args:
        file: Original MIDI file
        edits: JSON string of parameter edits

    Returns:
        Edited MIDI file bytes
    """
    try:
        # Read file
        midi_data = await file.read()

        # Parse edits
        edit_dict = json.loads(edits) if edits else {}

        # Get pipeline
        p = get_pipeline()

        logger.info(f"Editing {file.filename} with {len(edit_dict)} parameter changes")

        # Extract original DNA
        # original_dna = p.extract_dna_from_bytes(midi_data)

        # Apply edits
        # edited_dna = original_dna.copy()
        # edited_dna.update(edit_dict)

        # Generate
        # midi_bytes = p.generate_from_dna_to_bytes(edited_dna)

        # Mock for now
        midi_bytes = b"EDITED_MIDI_DATA_HERE"

        return Response(
            content=midi_bytes,
            media_type="audio/midi",
            headers={
                "Content-Disposition": f"attachment; filename=edited_{file.filename}"
            }
        )

    except Exception as e:
        logger.error(f"Editing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/parameters")
async def get_parameters():
    """
    Get list of all available parameters with descriptions.

    Returns:
        Dictionary of parameter metadata
    """
    # In production, load from parameter registry
    parameters = {
        "harmony": [
            {"name": "harmony.chord_complexity", "range": [0.0, 1.0], "description": "Complexity of chord voicings"},
            {"name": "harmony.tension_resolution_rate", "range": [0.0, 1.0], "description": "Frequency of tension/resolution"},
            # ... more harmony parameters
        ],
        "rhythm": [
            {"name": "rhythm.syncopation_intensity", "range": [0.0, 1.0], "description": "Off-beat emphasis"},
            {"name": "rhythm.swing_ratio", "range": [0.0, 1.0], "description": "Swing vs. straight feel"},
            # ... more rhythm parameters
        ],
        # ... other dimensions
    }

    return parameters


@app.get("/stats")
async def get_stats():
    """
    Get API statistics.

    Returns:
        API usage statistics
    """
    return {
        "total_extractions": 0,
        "total_generations": 0,
        "total_edits": 0,
        "uptime": "0h 0m 0s",
        "version": "1.0.0"
    }


# Error handlers

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
