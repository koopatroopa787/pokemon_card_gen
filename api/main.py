import io
import math
import os
import base64
import zipfile
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
import torchvision.transforms as transforms

# Add parent directory to path to import model
import sys
sys.path.append(str(Path(__file__).parent.parent))

from model import Generator


app = FastAPI(
    title="Pokemon Card Generator API",
    description="API for generating Pokemon cards using a trained GAN model",
    version="1.0.0"
)

# CORS middleware to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"
LATENT_DIM = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global model variable
generator = None


def _tensor_to_pil(img_tensor) -> Image.Image:
    """Convert a single image tensor (C, H, W) in [-1, 1] range to a PIL Image."""
    img_tensor = (img_tensor + 1) / 2
    img_tensor = torch.clamp(img_tensor, 0, 1)
    img_np = img_tensor.cpu().permute(1, 2, 0).numpy()
    img_np = (img_np * 255).astype('uint8')
    return Image.fromarray(img_np)


class GenerateRequest(BaseModel):
    num_images: int = 1
    seed: Optional[int] = None


class ModelInfo(BaseModel):
    model_loaded: bool
    device: str
    latent_dim: int
    available_checkpoints: list[str]


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
        # Initialize generator
        generator = Generator(latent_dim=LATENT_DIM).to(DEVICE)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        generator.eval()

        print(f"Model loaded from: {checkpoint_path}")
        print(f"Using device: {DEVICE}")
        return True
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_latest_checkpoint()


@app.get("/", tags=["Root"])
async def root():
    """API root endpoint"""
    return {
        "message": "Pokemon Card Generator API",
        "version": "1.0.0",
        "endpoints": {
            "generate": "/generate",
            "info": "/info",
            "docs": "/docs"
        }
    }


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


@app.post("/generate", tags=["Generation"])
async def generate_cards(request: GenerateRequest):
    """
    Generate Pokemon cards in batch and return them as base64 encoded JSON.

    Args:
        request: Generation parameters including number of images and optional seed.

    Returns:
        JSON list of base64-encoded PNGs.
    """
    if request.num_images < 1 or request.num_images > 16:
        raise HTTPException(
            status_code=400,
            detail="Number of images must be between 1 and 16"
        )

    if generator is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train a model first."
        )

    try:
        # Set seed if provided
        if request.seed is not None:
            torch.manual_seed(request.seed)

        # Generate images
        with torch.no_grad():
            noise = torch.randn(request.num_images, LATENT_DIM, 1, 1, device=DEVICE)
            generated_images = generator(noise)

        # Convert all tensors to PIL images (reuses shared helper)
        pil_images = [_tensor_to_pil(generated_images[i]) for i in range(request.num_images)]

        # Convert to base64
        base64_images = []
        for img in pil_images:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            base64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
            base64_images.append(base64_str)

        return {
            "success": True,
            "num_images": request.num_images,
            "images": base64_images
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating images: {str(e)}"
        )


@app.get("/generate/download", tags=["Generation"])
async def download_cards(num_images: int = 1, seed: Optional[int] = None, format: Optional[str] = None):
    """
    Generate Pokemon cards in batch and download them as a file.

    Args:
        num_images: Number of images to generate (1-16)
        seed: Optional seed for reproducibility
        format: Output format. Use ``grid`` to receive a single PNG contact sheet
                (4 columns, dynamic rows). Omit or pass any other value to receive
                a ZIP archive containing one PNG per card.

    Returns:
        - ``application/zip`` StreamingResponse by default (``card_01.png``, …).
        - ``image/png`` StreamingResponse when ``format=grid``.
    """
    if num_images < 1 or num_images > 16:
        raise HTTPException(
            status_code=400,
            detail="Number of images must be between 1 and 16"
        )

    if generator is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train a model first."
        )

    try:
        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)

        # Generate images
        with torch.no_grad():
            noise = torch.randn(num_images, LATENT_DIM, 1, 1, device=DEVICE)
            generated_images = generator(noise)

        # Convert all tensors to PIL images (reuses shared helper)
        pil_images = [_tensor_to_pil(generated_images[i]) for i in range(num_images)]

        if format == "grid" and num_images > 1:
            # --- Feature B: contact-sheet grid PNG ---
            COLS = 4
            rows = math.ceil(num_images / COLS)
            card_w, card_h = pil_images[0].size
            grid_img = Image.new("RGBA", (COLS * card_w, rows * card_h))
            for idx, img in enumerate(pil_images):
                col = idx % COLS
                row = idx // COLS
                grid_img.paste(img, (col * card_w, row * card_h))

            buffer = io.BytesIO()
            grid_img.save(buffer, format="PNG")
            buffer.seek(0)
            return StreamingResponse(buffer, media_type="image/png")

        else:
            # --- Feature A: ZIP archive (default) ---
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for i, img in enumerate(pil_images):
                    img_buf = io.BytesIO()
                    img.save(img_buf, format="PNG")
                    zf.writestr(f"card_{i + 1:02d}.png", img_buf.getvalue())
            zip_buf.seek(0)
            return StreamingResponse(
                zip_buf,
                media_type="application/zip",
                headers={"Content-Disposition": 'attachment; filename="cards.zip"'},
            )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating images: {str(e)}"
        )


@app.get("/generate/single", tags=["Generation"])
async def generate_single_card(seed: Optional[int] = None):
    """
    Generate a single Pokemon card and return as image

    Args:
        seed: Optional seed for reproducibility

    Returns:
        PNG image
    """
    if generator is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train a model first."
        )

    try:
        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)

        # Generate image
        with torch.no_grad():
            noise = torch.randn(1, LATENT_DIM, 1, 1, device=DEVICE)
            generated_image = generator(noise)[0]

        # Convert tensor to PIL image (reuses shared helper)
        img_pil = _tensor_to_pil(generated_image)

        # Return as streaming response
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
