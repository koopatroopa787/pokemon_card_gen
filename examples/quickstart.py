#!/usr/bin/env python3
"""
Quick start example for Pokemon Card Generator

This script demonstrates:
1. Loading a trained model
2. Generating cards
3. Saving results
"""

import torch
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from model import Generator
from torchvision.utils import save_image


def main():
    print("Pokemon Card Generator - Quick Start")
    print("=" * 50)

    # Configuration
    checkpoint_path = "checkpoints/checkpoint_epoch_final.pth"
    output_dir = "examples/output"
    num_cards = 9
    latent_dim = 100

    # Check if checkpoint exists
    if not Path(checkpoint_path).exists():
        print(f"\nError: Checkpoint not found at {checkpoint_path}")
        print("\nPlease train a model first:")
        print("  python train.py --num_epochs 100 --batch_size 32")
        return

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load model
    print(f"\nLoading model from: {checkpoint_path}")
    generator = Generator(latent_dim=latent_dim).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()

    print("Model loaded successfully!")

    # Generate cards
    print(f"\nGenerating {num_cards} Pokemon cards...")

    with torch.no_grad():
        # Create random noise
        noise = torch.randn(num_cards, latent_dim, 1, 1, device=device)

        # Generate images
        fake_images = generator(noise)

    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save as grid
    grid_path = f"{output_dir}/cards_grid.png"
    save_image(fake_images, grid_path, nrow=3, normalize=True)
    print(f"\nGrid saved to: {grid_path}")

    # Save individual cards
    for i in range(num_cards):
        card_path = f"{output_dir}/card_{i+1}.png"
        save_image(fake_images[i], card_path, normalize=True)

    print(f"Individual cards saved to: {output_dir}/")

    print("\n" + "=" * 50)
    print("Generation complete!")
    print("\nNext steps:")
    print("  1. Check the generated cards in the output directory")
    print("  2. Try different seeds for reproducible results")
    print("  3. Start the API server: python run_server.py")
    print("  4. Open the web UI: http://localhost:8000/ui")


if __name__ == "__main__":
    main()
