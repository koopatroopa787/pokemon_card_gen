"""
Example: Using the Pokemon Card Generator API
"""

import requests
import base64
from PIL import Image
import io


def generate_cards(num_images=4, seed=None, api_url="http://localhost:8000"):
    """
    Generate Pokemon cards using the API

    Args:
        num_images: Number of cards to generate
        seed: Optional seed for reproducibility
        api_url: API base URL

    Returns:
        List of PIL Images
    """
    # Prepare request
    payload = {"num_images": num_images}
    if seed is not None:
        payload["seed"] = seed

    # Make request
    response = requests.post(f"{api_url}/generate", json=payload)

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.json())
        return []

    # Parse response
    data = response.json()
    print(f"Generated {data['num_images']} cards in {data.get('generation_time', 0):.2f}s")

    # Decode images
    images = []
    for img_b64 in data['images']:
        img_data = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(img_data))
        images.append(img)

    return images


def get_api_info(api_url="http://localhost:8000"):
    """Get API information"""
    response = requests.get(f"{api_url}/info")
    data = response.json()

    print("API Information:")
    print(f"  Model Loaded: {data['model_loaded']}")
    print(f"  Device: {data['device']}")
    print(f"  Latent Dimension: {data['latent_dim']}")
    print(f"  Available Checkpoints: {len(data['available_checkpoints'])}")

    return data


def get_health_status(api_url="http://localhost:8000"):
    """Get API health status (if using enhanced API)"""
    try:
        response = requests.get(f"{api_url}/health")
        data = response.json()

        print("Health Status:")
        print(f"  Status: {data['status']}")
        print(f"  Uptime: {data['uptime_seconds']:.0f} seconds")
        print(f"  Model Loaded: {data['model_loaded']}")

        return data
    except:
        print("Health endpoint not available (using standard API)")
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Pokemon Card Generator API Example')
    parser.add_argument('--api-url', type=str, default='http://localhost:8000',
                       help='API base URL')
    parser.add_argument('--num-images', type=int, default=4,
                       help='Number of images to generate')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed')
    parser.add_argument('--output', type=str, default='examples/output',
                       help='Output directory')

    args = parser.parse_args()

    # Get API info
    print("\n" + "="*50)
    get_api_info(args.api_url)

    print("\n" + "="*50)
    get_health_status(args.api_url)

    # Generate cards
    print("\n" + "="*50)
    print(f"Generating {args.num_images} cards...")
    images = generate_cards(
        num_images=args.num_images,
        seed=args.seed,
        api_url=args.api_url
    )

    if images:
        # Save images
        import os
        os.makedirs(args.output, exist_ok=True)

        for i, img in enumerate(images):
            output_path = os.path.join(args.output, f'card_{i+1}.png')
            img.save(output_path)
            print(f"Saved: {output_path}")

        print(f"\nAll cards saved to: {args.output}")
    else:
        print("No images generated")
