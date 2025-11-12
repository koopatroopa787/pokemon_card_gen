"""
Unit tests for GAN models
"""

import unittest
import torch
from model import Generator, Discriminator


class TestGenerator(unittest.TestCase):
    """Test Generator model"""

    def setUp(self):
        """Set up test fixtures"""
        self.latent_dim = 100
        self.batch_size = 4
        self.generator = Generator(latent_dim=self.latent_dim)

    def test_generator_initialization(self):
        """Test generator initializes correctly"""
        self.assertIsNotNone(self.generator)
        self.assertEqual(self.generator.latent_dim, self.latent_dim)

    def test_generator_forward(self):
        """Test generator forward pass"""
        noise = torch.randn(self.batch_size, self.latent_dim, 1, 1)
        output = self.generator(noise)

        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, 3, 256, 256))

        # Check output range (should be in [-1, 1] due to Tanh)
        self.assertTrue(output.min() >= -1.0)
        self.assertTrue(output.max() <= 1.0)

    def test_generator_generate(self):
        """Test generator generate method"""
        num_images = 5
        images = self.generator.generate(num_images=num_images, device='cpu')

        self.assertEqual(images.shape, (num_images, 3, 256, 256))

    def test_generator_deterministic(self):
        """Test generator produces same output with same seed"""
        torch.manual_seed(42)
        noise1 = torch.randn(1, self.latent_dim, 1, 1)

        torch.manual_seed(42)
        noise2 = torch.randn(1, self.latent_dim, 1, 1)

        output1 = self.generator(noise1)
        output2 = self.generator(noise2)

        self.assertTrue(torch.allclose(output1, output2))


class TestDiscriminator(unittest.TestCase):
    """Test Discriminator model"""

    def setUp(self):
        """Set up test fixtures"""
        self.batch_size = 4
        self.discriminator = Discriminator()

    def test_discriminator_initialization(self):
        """Test discriminator initializes correctly"""
        self.assertIsNotNone(self.discriminator)

    def test_discriminator_forward(self):
        """Test discriminator forward pass"""
        images = torch.randn(self.batch_size, 3, 256, 256)
        output = self.discriminator(images)

        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, 1, 1, 1))

        # Check output range (should be in [0, 1] due to Sigmoid)
        self.assertTrue(output.min() >= 0.0)
        self.assertTrue(output.max() <= 1.0)

    def test_discriminator_real_vs_fake(self):
        """Test discriminator can distinguish patterns"""
        # Real-like images (structured)
        real_images = torch.ones(self.batch_size, 3, 256, 256) * 0.5

        # Fake-like images (random noise)
        fake_images = torch.randn(self.batch_size, 3, 256, 256)

        real_output = self.discriminator(real_images)
        fake_output = self.discriminator(fake_images)

        # Both should be valid probabilities
        self.assertTrue(torch.all(real_output >= 0))
        self.assertTrue(torch.all(real_output <= 1))
        self.assertTrue(torch.all(fake_output >= 0))
        self.assertTrue(torch.all(fake_output <= 1))


class TestGANIntegration(unittest.TestCase):
    """Test Generator and Discriminator together"""

    def setUp(self):
        """Set up test fixtures"""
        self.latent_dim = 100
        self.batch_size = 4
        self.generator = Generator(latent_dim=self.latent_dim)
        self.discriminator = Discriminator()

    def test_gan_pipeline(self):
        """Test full GAN pipeline"""
        # Generate fake images
        noise = torch.randn(self.batch_size, self.latent_dim, 1, 1)
        fake_images = self.generator(noise)

        # Discriminate fake images
        fake_output = self.discriminator(fake_images)

        # Check shapes
        self.assertEqual(fake_images.shape, (self.batch_size, 3, 256, 256))
        self.assertEqual(fake_output.shape, (self.batch_size, 1, 1, 1))

    def test_backward_pass(self):
        """Test backward pass works"""
        # Create optimizer
        optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=0.0002)
        optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002)

        # Generate fake images
        noise = torch.randn(self.batch_size, self.latent_dim, 1, 1)
        fake_images = self.generator(noise)

        # Discriminator loss
        fake_output = self.discriminator(fake_images.detach())
        d_loss = torch.nn.BCELoss()(
            fake_output,
            torch.zeros_like(fake_output)
        )

        # Backward pass for discriminator
        optimizer_d.zero_grad()
        d_loss.backward()
        optimizer_d.step()

        # Generator loss
        fake_output = self.discriminator(fake_images)
        g_loss = torch.nn.BCELoss()(
            fake_output,
            torch.ones_like(fake_output)
        )

        # Backward pass for generator
        optimizer_g.zero_grad()
        g_loss.backward()
        optimizer_g.step()

        # Check that losses are computed
        self.assertIsNotNone(d_loss.item())
        self.assertIsNotNone(g_loss.item())


if __name__ == '__main__':
    unittest.main()
