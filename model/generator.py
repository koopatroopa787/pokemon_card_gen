import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    DCGAN Generator for Pokemon card generation.
    Takes a latent vector and generates a Pokemon card image.
    """

    def __init__(self, latent_dim=100, img_channels=3, feature_maps=64):
        """
        Args:
            latent_dim: Dimension of the latent noise vector
            img_channels: Number of channels in output image (3 for RGB)
            feature_maps: Base number of feature maps
        """
        super(Generator, self).__init__()

        self.latent_dim = latent_dim

        # Input: latent_dim x 1 x 1
        # Output: img_channels x 256 x 256
        self.main = nn.Sequential(
            # Input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, feature_maps * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 16),
            nn.ReLU(True),
            # State: (feature_maps*16) x 4 x 4

            nn.ConvTranspose2d(feature_maps * 16, feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            # State: (feature_maps*8) x 8 x 8

            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            # State: (feature_maps*4) x 16 x 16

            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            # State: (feature_maps*2) x 32 x 32

            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            # State: feature_maps x 64 x 64

            nn.ConvTranspose2d(feature_maps, feature_maps // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps // 2),
            nn.ReLU(True),
            # State: (feature_maps//2) x 128 x 128

            nn.ConvTranspose2d(feature_maps // 2, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output: img_channels x 256 x 256
        )

    def forward(self, noise):
        """
        Args:
            noise: Tensor of shape (batch_size, latent_dim, 1, 1)
        Returns:
            Generated images of shape (batch_size, img_channels, 256, 256)
        """
        return self.main(noise)

    def generate(self, num_images=1, device='cpu'):
        """
        Generate random Pokemon cards

        Args:
            num_images: Number of images to generate
            device: Device to generate on
        Returns:
            Generated images
        """
        noise = torch.randn(num_images, self.latent_dim, 1, 1, device=device)
        with torch.no_grad():
            return self.forward(noise)
