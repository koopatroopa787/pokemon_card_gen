#!/usr/bin/env python3
"""
Batch inference script for generating Pokemon cards
"""

import argparse
import os
from pathlib import Path
import torch
from torchvision.utils import save_image
from tqdm import tqdm

from model import Generator


def load_generator(checkpoint_path, latent_dim=100, device='cpu'):
    """
    Load trained generator model

    Args:
        checkpoint_path: Path to checkpoint file
        latent_dim: Latent dimension
        device: Device to load on

    Returns:
        Loaded generator model
    """
    generator = Generator(latent_dim=latent_dim).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()

    return generator


def generate_cards(
    generator,
    num_images,
    output_dir,
    batch_size=32,
    seed=None,
    device='cpu',
    save_grid=True,
    prefix='card'
):
    """
    Generate Pokemon cards

    Args:
        generator: Trained generator model
        num_images: Number of images to generate
        output_dir: Output directory
        batch_size: Batch size for generation
        seed: Random seed for reproducibility
        device: Device to generate on
        save_grid: Save images as grid
        prefix: Prefix for output filenames
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if seed is not None:
        torch.manual_seed(seed)
        print(f"Using seed: {seed}")

    latent_dim = generator.latent_dim
    all_images = []

    print(f"Generating {num_images} cards...")

    with torch.no_grad():
        for i in tqdm(range(0, num_images, batch_size)):
            current_batch_size = min(batch_size, num_images - i)

            # Generate noise
            noise = torch.randn(current_batch_size, latent_dim, 1, 1, device=device)

            # Generate images
            fake_images = generator(noise)
            all_images.append(fake_images.cpu())

            # Save individual images
            if not save_grid or num_images <= batch_size:
                for j in range(current_batch_size):
                    img_idx = i + j
                    save_image(
                        fake_images[j],
                        output_path / f"{prefix}_{img_idx:04d}.png",
                        normalize=True
                    )

    # Save as grid
    if save_grid and num_images > 1:
        all_images = torch.cat(all_images, dim=0)
        grid_path = output_path / f"{prefix}_grid.png"

        # Calculate grid size
        nrow = min(8, int(num_images ** 0.5))

        save_image(
            all_images,
            grid_path,
            nrow=nrow,
            normalize=True
        )
        print(f"Grid saved to: {grid_path}")

    print(f"Generation complete! Images saved to: {output_path}")


def interpolate_cards(
    generator,
    num_steps,
    output_dir,
    seed_start=None,
    seed_end=None,
    device='cpu'
):
    """
    Generate interpolation between two cards

    Args:
        generator: Trained generator model
        num_steps: Number of interpolation steps
        output_dir: Output directory
        seed_start: Seed for start point
        seed_end: Seed for end point
        device: Device to generate on
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    latent_dim = generator.latent_dim

    # Generate start and end points
    if seed_start is not None:
        torch.manual_seed(seed_start)
    z_start = torch.randn(1, latent_dim, 1, 1, device=device)

    if seed_end is not None:
        torch.manual_seed(seed_end)
    z_end = torch.randn(1, latent_dim, 1, 1, device=device)

    print(f"Generating {num_steps} interpolation steps...")

    all_images = []

    with torch.no_grad():
        for i in tqdm(range(num_steps)):
            # Linear interpolation
            alpha = i / (num_steps - 1)
            z = (1 - alpha) * z_start + alpha * z_end

            # Generate image
            fake_image = generator(z)
            all_images.append(fake_image.cpu())

            # Save individual frame
            save_image(
                fake_image,
                output_path / f"interp_{i:04d}.png",
                normalize=True
            )

    # Save as grid
    all_images = torch.cat(all_images, dim=0)
    save_image(
        all_images,
        output_path / "interpolation_grid.png",
        nrow=min(10, num_steps),
        normalize=True
    )

    print(f"Interpolation complete! Images saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate Pokemon cards')

    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')

    # Mode selection
    parser.add_argument('--mode', type=str, default='generate',
                       choices=['generate', 'interpolate'],
                       help='Generation mode')

    # Generation arguments
    parser.add_argument('--num-images', type=int, default=10,
                       help='Number of images to generate')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for generation')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--no-grid', action='store_true',
                       help='Do not save grid image')
    parser.add_argument('--prefix', type=str, default='card',
                       help='Prefix for output files')

    # Interpolation arguments
    parser.add_argument('--num-steps', type=int, default=10,
                       help='Number of interpolation steps')
    parser.add_argument('--seed-start', type=int, default=None,
                       help='Start seed for interpolation')
    parser.add_argument('--seed-end', type=int, default=None,
                       help='End seed for interpolation')

    # Model arguments
    parser.add_argument('--latent-dim', type=int, default=100,
                       help='Latent dimension')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto/cpu/cuda)')

    args = parser.parse_args()

    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from: {args.checkpoint}")
    generator = load_generator(args.checkpoint, args.latent_dim, device)
    print("Model loaded successfully!")

    # Generate or interpolate
    if args.mode == 'generate':
        generate_cards(
            generator=generator,
            num_images=args.num_images,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            seed=args.seed,
            device=device,
            save_grid=not args.no_grid,
            prefix=args.prefix
        )
    elif args.mode == 'interpolate':
        interpolate_cards(
            generator=generator,
            num_steps=args.num_steps,
            output_dir=args.output_dir,
            seed_start=args.seed_start,
            seed_end=args.seed_end,
            device=device
        )


if __name__ == '__main__':
    main()
