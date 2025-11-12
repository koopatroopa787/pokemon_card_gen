"""
Unit tests for API endpoints
"""

import unittest
from fastapi.testclient import TestClient
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


if __name__ == '__main__':
    unittest.main()
