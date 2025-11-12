"""
Evaluation metrics for GAN models
Includes Frechet Inception Distance (FID) and Inception Score (IS)
"""

import torch
import torch.nn as nn
import numpy as np
from scipy import linalg
from torchvision.models import inception_v3
from torch.nn.functional import adaptive_avg_pool2d


class InceptionV3Feature(nn.Module):
    """
    Pretrained InceptionV3 network for extracting features
    Used for FID and IS calculations
    """

    def __init__(self):
        super().__init__()
        inception = inception_v3(pretrained=True, transform_input=False)
        self.inception = inception
        self.inception.fc = nn.Identity()  # Remove final classification layer
        self.inception.eval()

    @torch.no_grad()
    def forward(self, x):
        """
        Extract features from images

        Args:
            x: Tensor of images (B, 3, 299, 299)

        Returns:
            Features of shape (B, 2048)
        """
        # Resize to 299x299 for InceptionV3
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = nn.functional.interpolate(
                x, size=(299, 299), mode='bilinear', align_corners=False
            )

        # Normalize to [-1, 1] if needed
        if x.max() > 1.0:
            x = x / 255.0

        # InceptionV3 expects values in range [-1, 1]
        x = 2 * x - 1

        # Extract features
        x = self.inception.Conv2d_1a_3x3(x)
        x = self.inception.Conv2d_2a_3x3(x)
        x = self.inception.Conv2d_2b_3x3(x)
        x = self.inception.maxpool1(x)
        x = self.inception.Conv2d_3b_1x1(x)
        x = self.inception.Conv2d_4a_3x3(x)
        x = self.inception.maxpool2(x)
        x = self.inception.Mixed_5b(x)
        x = self.inception.Mixed_5c(x)
        x = self.inception.Mixed_5d(x)
        x = self.inception.Mixed_6a(x)
        x = self.inception.Mixed_6b(x)
        x = self.inception.Mixed_6c(x)
        x = self.inception.Mixed_6d(x)
        x = self.inception.Mixed_6e(x)
        x = self.inception.Mixed_7a(x)
        x = self.inception.Mixed_7b(x)
        x = self.inception.Mixed_7c(x)

        # Global average pooling
        x = adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)

        return x


def calculate_fid(real_features, fake_features):
    """
    Calculate Frechet Inception Distance (FID)

    Lower is better. FID measures the distance between distributions of
    real and generated images.

    Args:
        real_features: Features from real images (N, 2048)
        fake_features: Features from fake images (M, 2048)

    Returns:
        FID score (float)
    """
    # Calculate mean and covariance
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)

    mu_fake = np.mean(fake_features, axis=0)
    sigma_fake = np.cov(fake_features, rowvar=False)

    # Calculate FID
    diff = mu_real - mu_fake
    covmean, _ = linalg.sqrtm(sigma_real @ sigma_fake, disp=False)

    # Handle numerical errors
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_real + sigma_fake - 2 * covmean)

    return float(fid)


def calculate_inception_score(images, inception_model, splits=10):
    """
    Calculate Inception Score (IS)

    Higher is better. IS measures both quality and diversity of generated images.

    Args:
        images: Generated images (N, 3, H, W) in range [0, 1]
        inception_model: InceptionV3 model for classification
        splits: Number of splits for calculation

    Returns:
        mean_is: Mean Inception Score
        std_is: Standard deviation of IS
    """
    N = len(images)
    preds = []

    with torch.no_grad():
        for i in range(0, N, 32):
            batch = images[i:i+32]
            if batch.shape[2] != 299 or batch.shape[3] != 299:
                batch = nn.functional.interpolate(
                    batch, size=(299, 299), mode='bilinear', align_corners=False
                )
            pred = inception_model(batch)
            preds.append(nn.functional.softmax(pred, dim=1).cpu().numpy())

    preds = np.concatenate(preds, axis=0)

    # Calculate IS for each split
    split_scores = []
    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(np.sum(pyx * np.log(pyx / py + 1e-10)))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


@torch.no_grad()
def extract_features(images, feature_extractor, batch_size=32, device='cpu'):
    """
    Extract inception features from a batch of images

    Args:
        images: Tensor of images (N, 3, H, W)
        feature_extractor: InceptionV3Feature model
        batch_size: Batch size for processing
        device: Device to run on

    Returns:
        Features array (N, 2048)
    """
    feature_extractor.to(device)
    feature_extractor.eval()

    features = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size].to(device)
        feat = feature_extractor(batch)
        features.append(feat.cpu().numpy())

    return np.concatenate(features, axis=0)


class GANMetrics:
    """
    Helper class for calculating GAN metrics
    """

    def __init__(self, device='cpu'):
        """
        Initialize metrics calculator

        Args:
            device: Device to run on (cpu/cuda)
        """
        self.device = device
        self.feature_extractor = InceptionV3Feature().to(device)
        self.feature_extractor.eval()

    @torch.no_grad()
    def compute_fid(self, real_images, fake_images, batch_size=32):
        """
        Compute FID between real and fake images

        Args:
            real_images: Real images tensor (N, 3, H, W) in range [0, 1]
            fake_images: Fake images tensor (M, 3, H, W) in range [0, 1]
            batch_size: Batch size for processing

        Returns:
            FID score
        """
        real_features = extract_features(
            real_images, self.feature_extractor, batch_size, self.device
        )
        fake_features = extract_features(
            fake_images, self.feature_extractor, batch_size, self.device
        )

        fid = calculate_fid(real_features, fake_features)
        return fid

    @torch.no_grad()
    def compute_is(self, fake_images, splits=10):
        """
        Compute Inception Score for fake images

        Args:
            fake_images: Fake images tensor (N, 3, H, W) in range [0, 1]
            splits: Number of splits

        Returns:
            (mean_is, std_is)
        """
        return calculate_inception_score(
            fake_images, self.feature_extractor.inception, splits
        )


if __name__ == '__main__':
    # Example usage
    print("Testing GAN metrics...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    metrics = GANMetrics(device=device)

    # Generate random images for testing
    real_images = torch.rand(100, 3, 256, 256)
    fake_images = torch.rand(100, 3, 256, 256)

    print("Computing FID...")
    fid = metrics.compute_fid(real_images, fake_images)
    print(f"FID: {fid:.2f}")

    print("Computing IS...")
    mean_is, std_is = metrics.compute_is(fake_images)
    print(f"IS: {mean_is:.2f} ± {std_is:.2f}")
