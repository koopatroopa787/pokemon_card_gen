"""
Unit tests for API endpoints
"""

import unittest
import io
import zipfile
from unittest.mock import patch
from fastapi.testclient import TestClient
from PIL import Image
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.main import app


class TestAPI(unittest.TestCase):
    """Test API endpoints"""

    def setUp(self):
        """Set up test client"""
        self.client = TestClient(app)

    def test_root_endpoint(self):
        """Test root endpoint"""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("message", data)
        self.assertIn("version", data)

    def test_info_endpoint(self):
        """Test info endpoint"""
        response = self.client.get("/info")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("model_loaded", data)
        self.assertIn("device", data)
        self.assertIn("latent_dim", data)

    def test_generate_endpoint_validation(self):
        """Test generate endpoint validation"""
        # Test with invalid number of images (too many)
        response = self.client.post(
            "/generate",
            json={"num_images": 100}
        )
        self.assertEqual(response.status_code, 400)

        # Test with invalid number of images (negative)
        response = self.client.post(
            "/generate",
            json={"num_images": -1}
        )
        self.assertEqual(response.status_code, 400)

    def test_generate_endpoint_structure(self):
        """Test generate endpoint response structure"""
        # Note: This will fail if no model is loaded, which is expected in testing
        response = self.client.post(
            "/generate",
            json={"num_images": 1}
        )

        # Either succeeds with model or returns 503 without model
        self.assertIn(response.status_code, [200, 503])

        if response.status_code == 200:
            data = response.json()
            self.assertIn("success", data)
            self.assertIn("num_images", data)
            self.assertIn("images", data)
            self.assertEqual(len(data["images"]), 1)

    def test_download_zip_and_grid(self):
        """Download a batch in both supported formats."""
        def fake_generator(noise):
            return torch.zeros(noise.shape[0], 3, 2, 2, device=noise.device)

        with patch("api.main.generator", fake_generator):
            zip_response = self.client.get("/generate/download?num_images=3&seed=1")
            grid_response = self.client.get("/generate/download?num_images=3&seed=1&format=grid")

        self.assertEqual(zip_response.status_code, 200)
        self.assertEqual(zip_response.headers["content-type"], "application/zip")
        with zipfile.ZipFile(io.BytesIO(zip_response.content)) as archive:
            self.assertEqual(archive.namelist(), ["card_01.png", "card_02.png", "card_03.png"])
            for name in archive.namelist():
                with Image.open(io.BytesIO(archive.read(name))) as card:
                    self.assertEqual(card.size, (2, 2))

        self.assertEqual(grid_response.status_code, 200)
        self.assertEqual(grid_response.headers["content-type"], "image/png")
        with Image.open(io.BytesIO(grid_response.content)) as grid:
            self.assertEqual(grid.size, (4, 4))


if __name__ == '__main__':
    unittest.main()
