import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    DCGAN Discriminator for Pokemon card generation.
    Distinguishes between real and fake Pokemon cards.
    """

    def __init__(self, img_channels=3, feature_maps=64):
        """
        Args:
            img_channels: Number of channels in input image (3 for RGB)
            feature_maps: Base number of feature maps
        """
        super(Discriminator, self).__init__()

        # Input: img_channels x 256 x 256
        # Output: 1 (real/fake probability)
        self.main = nn.Sequential(
            # Input: img_channels x 256 x 256
            nn.Conv2d(img_channels, feature_maps // 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (feature_maps//2) x 128 x 128

            nn.Conv2d(feature_maps // 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.LeakyReLU(0.2, inplace=True),
            # State: feature_maps x 64 x 64

            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (feature_maps*2) x 32 x 32

            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (feature_maps*4) x 16 x 16

            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (feature_maps*8) x 8 x 8

            nn.Conv2d(feature_maps * 8, feature_maps * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (feature_maps*16) x 4 x 4

            nn.Conv2d(feature_maps * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # Output: 1 x 1 x 1
        )

    def forward(self, img):
        """
        Args:
            img: Tensor of shape (batch_size, img_channels, 256, 256)
        Returns:
            Predictions of shape (batch_size, 1, 1, 1) - probability of being real
        """
        return self.main(img)
