#!/usr/bin/env python3
"""
Training script specifically for Pokemon Card dataset

Supports training with metadata-aware dataset loader
"""

import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import Generator, Discriminator
from utils.pokemon_dataset import get_pokemon_dataloader


class PokemonCardGANTrainer:
    """
    Trainer for Pokemon Card GAN using the Pokemon dataset
    """

    def __init__(
        self,
        data_dir: str,
        metadata_file: str = None,
        output_dir: str = 'data/generated',
        checkpoint_dir: str = 'checkpoints',
        log_dir: str = 'logs',
        latent_dim: int = 100,
        img_size: int = 256,
        batch_size: int = 32,
        lr: float = 0.0002,
        beta1: float = 0.5,
        num_epochs: int = 100,
        device: str = None,
        use_metadata: bool = False
    ):
        """
        Initialize Pokemon Card GAN Trainer

        Args:
            data_dir: Directory containing Pokemon card images
            metadata_file: Optional metadata JSON/CSV file
            output_dir: Directory to save generated images
            checkpoint_dir: Directory to save model checkpoints
            log_dir: Directory to save training logs
            latent_dim: Dimension of latent noise vector
            img_size: Size of images (square)
            batch_size: Batch size for training
            lr: Learning rate
            beta1: Beta1 parameter for Adam optimizer
            num_epochs: Number of training epochs
            device: Device to train on (cpu/cuda)
            use_metadata: Whether to use metadata (for future conditional GAN)
        """
        self.data_dir = data_dir
        self.metadata_file = metadata_file
        self.output_dir = output_dir
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.batch_size = batch_size
        self.lr = lr
        self.beta1 = beta1
        self.num_epochs = num_epochs
        self.use_metadata = use_metadata

        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Initialize models
        self.generator = Generator(latent_dim=latent_dim).to(self.device)
        self.discriminator = Discriminator().to(self.device)

        # Initialize weights
        self.generator.apply(self._weights_init)
        self.discriminator.apply(self._weights_init)

        # Loss function
        self.criterion = nn.BCELoss()

        # Optimizers
        self.optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=self.lr,
            betas=(self.beta1, 0.999)
        )
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=self.lr,
            betas=(self.beta1, 0.999)
        )

        # Fixed noise for visualization
        self.fixed_noise = torch.randn(64, latent_dim, 1, 1, device=self.device)

        # Training history
        self.g_losses = []
        self.d_losses = []

    @staticmethod
    def _weights_init(m):
        """Initialize model weights"""
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def get_dataloader(self):
        """Create Pokemon card dataloader"""
        print(f"\nLoading Pokemon card dataset from: {self.data_dir}")
        if self.metadata_file:
            print(f"Using metadata from: {self.metadata_file}")

        dataloader = get_pokemon_dataloader(
            data_dir=self.data_dir,
            metadata_file=self.metadata_file,
            batch_size=self.batch_size,
            img_size=self.img_size,
            num_workers=4,
            shuffle=True,
            use_metadata=self.use_metadata
        )

        return dataloader

    def train(self):
        """Train the GAN"""
        try:
            dataloader = self.get_dataloader()
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("\nMake sure:")
            print(f"  1. Images exist in: {self.data_dir}")
            print(f"  2. Images are organized in subdirectories (for ImageFolder)")
            print(f"  3. Or use the Pokemon dataset format with metadata")
            return

        print(f"\nStarting training for {self.num_epochs} epochs...")
        print(f"Batch size: {self.batch_size}")
        print(f"Images per epoch: {len(dataloader.dataset)}")
        print(f"Batches per epoch: {len(dataloader)}")

        for epoch in range(self.num_epochs):
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}")

            for i, (real_imgs, _) in enumerate(pbar):
                batch_size = real_imgs.size(0)
                real_imgs = real_imgs.to(self.device)

                # Create labels
                real_labels = torch.ones(batch_size, 1, 1, 1, device=self.device)
                fake_labels = torch.zeros(batch_size, 1, 1, 1, device=self.device)

                # ---------------------
                # Train Discriminator
                # ---------------------
                self.optimizer_d.zero_grad()

                # Real images
                real_output = self.discriminator(real_imgs)
                d_loss_real = self.criterion(real_output, real_labels)

                # Fake images
                noise = torch.randn(batch_size, self.latent_dim, 1, 1, device=self.device)
                fake_imgs = self.generator(noise)
                fake_output = self.discriminator(fake_imgs.detach())
                d_loss_fake = self.criterion(fake_output, fake_labels)

                # Total discriminator loss
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.optimizer_d.step()

                # ---------------------
                # Train Generator
                # ---------------------
                self.optimizer_g.zero_grad()

                # Generate fake images and try to fool discriminator
                fake_output = self.discriminator(fake_imgs)
                g_loss = self.criterion(fake_output, real_labels)

                g_loss.backward()
                self.optimizer_g.step()

                # Update progress bar
                pbar.set_postfix({
                    'D_loss': f'{d_loss.item():.4f}',
                    'G_loss': f'{g_loss.item():.4f}'
                })

                # Store losses
                self.g_losses.append(g_loss.item())
                self.d_losses.append(d_loss.item())

            # Save generated images at the end of each epoch
            with torch.no_grad():
                fake_imgs = self.generator(self.fixed_noise)
                save_image(
                    fake_imgs,
                    os.path.join(self.output_dir, f'epoch_{epoch+1}.png'),
                    nrow=8,
                    normalize=True
                )

            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch + 1)

        # Save final model
        self.save_checkpoint('final')
        self.plot_losses()
        print("\nTraining completed!")

    def save_checkpoint(self, epoch):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'g_losses': self.g_losses,
            'd_losses': self.d_losses,
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        self.g_losses = checkpoint['g_losses']
        self.d_losses = checkpoint['d_losses']
        print(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint['epoch']

    def plot_losses(self):
        """Plot training losses"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.g_losses, label='Generator Loss', alpha=0.7)
        plt.plot(self.d_losses, label='Discriminator Loss', alpha=0.7)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Pokemon Card GAN Training Losses')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.log_dir, 'training_losses.png'))
        plt.close()
        print(f"Loss plot saved to {self.log_dir}/training_losses.png")


def main():
    parser = argparse.ArgumentParser(
        description='Train Pokemon Card GAN with Pokemon dataset'
    )

    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data/pokemon_cards/training/train',
                       help='Directory containing Pokemon card images')
    parser.add_argument('--metadata', type=str, default=None,
                       help='Optional metadata JSON/CSV file')
    parser.add_argument('--use-metadata', action='store_true',
                       help='Use metadata during training (for future conditional GAN)')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='data/generated',
                       help='Directory to save generated images')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Directory to save logs')

    # Model arguments
    parser.add_argument('--latent_dim', type=int, default=100,
                       help='Dimension of latent noise vector')
    parser.add_argument('--img_size', type=int, default=256,
                       help='Size of images (square)')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0002,
                       help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.5,
                       help='Beta1 for Adam optimizer')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')

    # Resume training
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume training')

    # Device
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu, default: auto)')

    args = parser.parse_args()

    # Initialize trainer
    trainer = PokemonCardGANTrainer(
        data_dir=args.data_dir,
        metadata_file=args.metadata,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        latent_dim=args.latent_dim,
        img_size=args.img_size,
        batch_size=args.batch_size,
        lr=args.lr,
        beta1=args.beta1,
        num_epochs=args.num_epochs,
        device=args.device,
        use_metadata=args.use_metadata
    )

    # Load checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    # Start training
    print("\n" + "="*50)
    print("Pokemon Card GAN Training")
    print("="*50)
    trainer.train()


if __name__ == '__main__':
    main()
