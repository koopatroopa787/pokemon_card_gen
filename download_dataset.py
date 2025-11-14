"""
Pokemon Card Dataset Downloader

Downloads Pokemon card images from a CSV file containing image URLs.
Supports the Pokemon TCG dataset format with metadata.
"""

import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from PIL import Image
import io


class PokemonCardDownloader:
    """Download Pokemon card images from URLs"""

    def __init__(
        self,
        csv_path: str,
        output_dir: str = "data/pokemon_cards",
        max_workers: int = 10,
        delay: float = 0.1
    ):
        """
        Initialize downloader

        Args:
            csv_path: Path to CSV file with columns: id, image_url, name, hp, caption, set_name
            output_dir: Directory to save images
            max_workers: Number of parallel download threads
            delay: Delay between requests (seconds)
        """
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
        self.delay = delay

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)

    def load_dataset(self) -> pd.DataFrame:
        """Load dataset from CSV"""
        print(f"Loading dataset from {self.csv_path}")
        df = pd.read_csv(self.csv_path)

        required_columns = ['id', 'image_url', 'name', 'caption']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        print(f"Loaded {len(df)} cards")
        return df

    def download_image(self, row: pd.Series) -> dict:
        """
        Download a single image

        Args:
            row: DataFrame row with card info

        Returns:
            Dictionary with download results
        """
        card_id = row['id']
        image_url = row['image_url']

        # Skip if already downloaded
        output_path = self.output_dir / "images" / f"{card_id}.jpg"
        if output_path.exists():
            return {"id": card_id, "status": "skipped", "path": str(output_path)}

        try:
            # Download image
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()

            # Open and verify image
            img = Image.open(io.BytesIO(response.content))

            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Save image
            img.save(output_path, 'JPEG', quality=95)

            # Small delay to be nice to the server
            time.sleep(self.delay)

            return {
                "id": card_id,
                "status": "success",
                "path": str(output_path),
                "size": img.size
            }

        except Exception as e:
            return {
                "id": card_id,
                "status": "failed",
                "error": str(e)
            }

    def download_all(self, limit: int = None) -> pd.DataFrame:
        """
        Download all images from dataset

        Args:
            limit: Optional limit on number of images to download

        Returns:
            DataFrame with download results
        """
        # Load dataset
        df = self.load_dataset()

        if limit:
            df = df.head(limit)
            print(f"Limiting to first {limit} cards")

        # Download images
        results = []
        print(f"\nDownloading {len(df)} images...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all download tasks
            futures = {
                executor.submit(self.download_image, row): idx
                for idx, row in df.iterrows()
            }

            # Process results as they complete
            for future in tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                results.append(result)

        # Create results DataFrame
        results_df = pd.DataFrame(results)

        # Print summary
        print("\n" + "="*50)
        print("Download Summary:")
        print(f"  Total: {len(results_df)}")
        print(f"  Success: {(results_df['status'] == 'success').sum()}")
        print(f"  Skipped: {(results_df['status'] == 'skipped').sum()}")
        print(f"  Failed: {(results_df['status'] == 'failed').sum()}")

        if (results_df['status'] == 'failed').any():
            print("\nFailed downloads:")
            failed = results_df[results_df['status'] == 'failed']
            for _, row in failed.head(10).iterrows():
                print(f"  {row['id']}: {row.get('error', 'Unknown error')}")

        return results_df

    def create_metadata_file(self):
        """Create metadata JSON file from CSV"""
        df = self.load_dataset()

        # Save as JSON
        metadata_path = self.output_dir / "metadata" / "cards.json"
        df.to_json(metadata_path, orient='records', indent=2)
        print(f"Metadata saved to: {metadata_path}")

        # Also save as CSV for easy viewing
        metadata_csv = self.output_dir / "metadata" / "cards.csv"
        df.to_csv(metadata_csv, index=False)
        print(f"Metadata CSV saved to: {metadata_csv}")

    def organize_by_set(self):
        """Organize downloaded images by set name"""
        df = self.load_dataset()

        print("\nOrganizing images by set...")

        for _, row in tqdm(df.iterrows(), total=len(df)):
            card_id = row['id']
            set_name = row.get('set_name', 'unknown')

            # Clean set name for directory
            set_dir = "".join(c for c in set_name if c.isalnum() or c in (' ', '-', '_'))
            set_dir = set_dir.replace(' ', '_')

            # Create set directory
            set_path = self.output_dir / "by_set" / set_dir
            set_path.mkdir(parents=True, exist_ok=True)

            # Copy or move image
            src = self.output_dir / "images" / f"{card_id}.jpg"
            dst = set_path / f"{card_id}.jpg"

            if src.exists() and not dst.exists():
                import shutil
                shutil.copy(src, dst)

        print("Organization complete!")

    def prepare_for_training(self, output_size: int = 256):
        """
        Prepare images for training

        Args:
            output_size: Target size for images
        """
        from utils.data_utils import ImagePreprocessor

        print(f"\nPreparing images for training (target size: {output_size}x{output_size})")

        preprocessor = ImagePreprocessor(target_size=output_size)

        # Process all downloaded images
        images_dir = self.output_dir / "images"
        training_dir = self.output_dir / "training"

        preprocessor.preprocess_dataset(
            str(images_dir),
            str(training_dir),
            organize_subdirs=True
        )

        print(f"Training data ready in: {training_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Download Pokemon card images from CSV'
    )
    parser.add_argument('csv_path', type=str,
                       help='Path to CSV file with card data')
    parser.add_argument('--output-dir', type=str,
                       default='data/pokemon_cards',
                       help='Output directory')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of downloads (for testing)')
    parser.add_argument('--max-workers', type=int, default=10,
                       help='Number of parallel download threads')
    parser.add_argument('--delay', type=float, default=0.1,
                       help='Delay between requests (seconds)')
    parser.add_argument('--organize-by-set', action='store_true',
                       help='Organize images by set name')
    parser.add_argument('--prepare-training', action='store_true',
                       help='Prepare images for training')
    parser.add_argument('--target-size', type=int, default=256,
                       help='Target size for training images')

    args = parser.parse_args()

    # Create downloader
    downloader = PokemonCardDownloader(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        delay=args.delay
    )

    # Download images
    results = downloader.download_all(limit=args.limit)

    # Create metadata
    downloader.create_metadata_file()

    # Organize by set if requested
    if args.organize_by_set:
        downloader.organize_by_set()

    # Prepare for training if requested
    if args.prepare_training:
        downloader.prepare_for_training(output_size=args.target_size)

    print("\n" + "="*50)
    print("Download complete!")
    print(f"Images saved to: {args.output_dir}/images/")
    print(f"Metadata saved to: {args.output_dir}/metadata/")

    if args.prepare_training:
        print(f"Training data: {args.output_dir}/training/")


if __name__ == "__main__":
    main()
