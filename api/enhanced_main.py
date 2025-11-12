"""
Enhanced API with rate limiting, monitoring, and health checks
"""

import os
import io
import base64
import time
from pathlib import Path
from typing import Optional
from collections import defaultdict
from datetime import datetime

import torch
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
import torchvision.transforms as transforms

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from model import Generator


# ==================== Configuration ====================
CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"
LATENT_DIM = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RATE_LIMIT_PER_MINUTE = 30
MAX_BATCH_SIZE = 16

# Global state
generator = None
request_stats = defaultdict(lambda: {"count": 0, "last_reset": time.time()})
generation_history = []


# ==================== Models ====================
class GenerateRequest(BaseModel):
    num_images: int = 1
    seed: Optional[int] = None


class ModelInfo(BaseModel):
    model_loaded: bool
    device: str
    latent_dim: int
    available_checkpoints: list[str]


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    device: str
    uptime_seconds: float


class StatsResponse(BaseModel):
    total_generations: int
    total_images: int
    avg_generation_time: float
    recent_requests: int


# ==================== App Setup ====================
app = FastAPI(
    title="Pokemon Card Generator API (Enhanced)",
    description="Enhanced API with rate limiting, monitoring, and health checks",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup time for uptime calculation
startup_time = time.time()


# ==================== Rate Limiting ====================
async def rate_limit_check(request: Request):
    """Simple rate limiting middleware"""
    client_ip = request.client.host
    current_time = time.time()

    # Reset counter if a minute has passed
    if current_time - request_stats[client_ip]["last_reset"] > 60:
        request_stats[client_ip] = {"count": 0, "last_reset": current_time}

    # Check rate limit
    request_stats[client_ip]["count"] += 1
    if request_stats[client_ip]["count"] > RATE_LIMIT_PER_MINUTE:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Maximum {RATE_LIMIT_PER_MINUTE} requests per minute."
        )


# ==================== Model Loading ====================
def load_latest_checkpoint():
    """Load the latest or final checkpoint"""
    global generator

    if not CHECKPOINT_DIR.exists():
        print(f"Checkpoint directory not found: {CHECKPOINT_DIR}")
        return False

    # Look for final checkpoint first
    final_checkpoint = CHECKPOINT_DIR / "checkpoint_epoch_final.pth"
    if final_checkpoint.exists():
        checkpoint_path = final_checkpoint
    else:
        # Find latest checkpoint
        checkpoints = list(CHECKPOINT_DIR.glob("checkpoint_epoch_*.pth"))
        if not checkpoints:
            print(f"No checkpoints found in {CHECKPOINT_DIR}")
            return False
        checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)

    try:
        generator = Generator(latent_dim=LATENT_DIM).to(DEVICE)
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        generator.eval()

        print(f"Model loaded from: {checkpoint_path}")
        print(f"Using device: {DEVICE}")
        return True
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return False


# ==================== Endpoints ====================
@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_latest_checkpoint()


@app.get("/", tags=["Root"])
async def root():
    """API root endpoint"""
    return {
        "message": "Pokemon Card Generator API (Enhanced)",
        "version": "2.0.0",
        "features": ["rate_limiting", "monitoring", "health_checks"],
        "endpoints": {
            "generate": "/generate",
            "info": "/info",
            "health": "/health",
            "stats": "/stats",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Health check endpoint for monitoring"""
    return HealthResponse(
        status="healthy" if generator is not None else "degraded",
        timestamp=datetime.utcnow().isoformat(),
        model_loaded=generator is not None,
        device=str(DEVICE),
        uptime_seconds=time.time() - startup_time
    )


@app.get("/stats", response_model=StatsResponse, tags=["Monitoring"])
async def get_stats():
    """Get API usage statistics"""
    if not generation_history:
        return StatsResponse(
            total_generations=0,
            total_images=0,
            avg_generation_time=0.0,
            recent_requests=0
        )

    total_generations = len(generation_history)
    total_images = sum(g["num_images"] for g in generation_history)
    avg_time = sum(g["duration"] for g in generation_history) / total_generations

    # Recent requests in last hour
    recent_cutoff = time.time() - 3600
    recent_requests = sum(1 for g in generation_history if g["timestamp"] > recent_cutoff)

    return StatsResponse(
        total_generations=total_generations,
        total_images=total_images,
        avg_generation_time=avg_time,
        recent_requests=recent_requests
    )


@app.get("/info", response_model=ModelInfo, tags=["Info"])
async def get_info():
    """Get information about the loaded model"""
    checkpoints = []
    if CHECKPOINT_DIR.exists():
        checkpoints = [p.name for p in CHECKPOINT_DIR.glob("checkpoint_epoch_*.pth")]

    return ModelInfo(
        model_loaded=generator is not None,
        device=str(DEVICE),
        latent_dim=LATENT_DIM,
        available_checkpoints=checkpoints
    )


@app.post("/generate", tags=["Generation"], dependencies=[Depends(rate_limit_check)])
async def generate_cards(request: GenerateRequest):
    """
    Generate Pokemon cards with rate limiting

    Args:
        request: Generation parameters

    Returns:
        Base64 encoded images
    """
    if generator is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train a model first."
        )

    if request.num_images < 1 or request.num_images > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Number of images must be between 1 and {MAX_BATCH_SIZE}"
        )

    start_time = time.time()

    try:
        # Set seed if provided
        if request.seed is not None:
            torch.manual_seed(request.seed)

        # Generate images
        with torch.no_grad():
            noise = torch.randn(request.num_images, LATENT_DIM, 1, 1, device=DEVICE)
            generated_images = generator(noise)

        # Convert to PIL images and encode as base64
        images_b64 = []
        for i in range(request.num_images):
            img_tensor = (generated_images[i] + 1) / 2
            img_tensor = torch.clamp(img_tensor, 0, 1)

            img_np = img_tensor.cpu().permute(1, 2, 0).numpy()
            img_np = (img_np * 255).astype('uint8')
            img_pil = Image.fromarray(img_np)

            buffer = io.BytesIO()
            img_pil.save(buffer, format='PNG')
            img_b64 = base64.b64encode(buffer.getvalue()).decode()
            images_b64.append(img_b64)

        duration = time.time() - start_time

        # Record statistics
        generation_history.append({
            "timestamp": time.time(),
            "num_images": request.num_images,
            "duration": duration,
            "seed": request.seed
        })

        # Keep only last 1000 entries
        if len(generation_history) > 1000:
            generation_history.pop(0)

        return {
            "success": True,
            "num_images": request.num_images,
            "images": images_b64,
            "generation_time": duration
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating images: {str(e)}"
        )


@app.get("/generate/single", tags=["Generation"], dependencies=[Depends(rate_limit_check)])
async def generate_single_card(seed: Optional[int] = None):
    """Generate a single Pokemon card"""
    if generator is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train a model first."
        )

    try:
        if seed is not None:
            torch.manual_seed(seed)

        with torch.no_grad():
            noise = torch.randn(1, LATENT_DIM, 1, 1, device=DEVICE)
            generated_image = generator(noise)[0]

        img_tensor = (generated_image + 1) / 2
        img_tensor = torch.clamp(img_tensor, 0, 1)

        img_np = img_tensor.cpu().permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype('uint8')
        img_pil = Image.fromarray(img_np)

        buffer = io.BytesIO()
        img_pil.save(buffer, format='PNG')
        buffer.seek(0)

        return StreamingResponse(buffer, media_type="image/png")

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating image: {str(e)}"
        )


@app.post("/reload", tags=["Admin"])
async def reload_model():
    """Reload the model from the latest checkpoint"""
    success = load_latest_checkpoint()
    if success:
        return {"success": True, "message": "Model reloaded successfully"}
    else:
        raise HTTPException(
            status_code=500,
            detail="Failed to reload model"
        )


# Mount static files for UI
ui_dir = Path(__file__).parent.parent / "ui"
if ui_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(ui_dir), html=True), name="ui")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
