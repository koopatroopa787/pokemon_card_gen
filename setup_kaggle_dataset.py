#!/usr/bin/env python3
"""
Automatic Kaggle Pokemon Cards Dataset Downloader

Downloads the Pokemon cards dataset directly from Kaggle and prepares it for training.
No manual CSV upload needed!
"""

import os
import sys
import json
import zipfile
import shutil
from pathlib import Path
import subprocess


class KagglePokemonDownloader:
    """
    Download and prepare Pokemon cards dataset from Kaggle
    """

    def __init__(
        self,
        dataset_name: str = "priyamchoksi/pokemon-cards",
        output_dir: str = "data/pokemon_cards",
        auto_prepare: bool = True
    ):
        """
        Initialize Kaggle downloader

        Args:
            dataset_name: Kaggle dataset identifier
            output_dir: Directory to save dataset
            auto_prepare: Automatically prepare for training
        """
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir)
        self.auto_prepare = auto_prepare

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir = self.output_dir / "raw"
        self.images_dir = self.output_dir / "images"
        self.training_dir = self.output_dir / "training"

    def check_kaggle_setup(self):
        """Check if Kaggle API is set up correctly"""
        print("Checking Kaggle API setup...")

        # Check if kaggle is installed
        try:
            import kaggle
            print("✓ Kaggle library installed")
        except ImportError:
            print("✗ Kaggle library not installed")
            print("\nInstalling Kaggle API...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
            print("✓ Kaggle library installed")

        # Check for API credentials
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"

        if not kaggle_json.exists():
            print("\n" + "="*60)
            print("KAGGLE API SETUP REQUIRED")
            print("="*60)
            print("\nTo download datasets from Kaggle, you need API credentials.")
            print("\nSteps:")
            print("  1. Go to: https://www.kaggle.com/settings")
            print("  2. Scroll to 'API' section")
            print("  3. Click 'Create New Token'")
            print("  4. Download kaggle.json")
            print("  5. Move kaggle.json to: ~/.kaggle/kaggle.json")
            print("\nOr run:")
            print(f"  mkdir -p ~/.kaggle")
            print(f"  mv ~/Downloads/kaggle.json ~/.kaggle/")
            print(f"  chmod 600 ~/.kaggle/kaggle.json")
            print("\n" + "="*60)

            # Try to create directory
            kaggle_dir = Path.home() / ".kaggle"
            kaggle_dir.mkdir(exist_ok=True)

            raise FileNotFoundError(
                "Kaggle API credentials not found. Please set up kaggle.json"
            )

        # Check permissions
        if kaggle_json.exists():
            # Set correct permissions
            os.chmod(kaggle_json, 0o600)
            print("✓ Kaggle credentials found")

        return True

    def download_dataset(self):
        """Download dataset from Kaggle"""
        print(f"\nDownloading dataset: {self.dataset_name}")
        print(f"Destination: {self.raw_dir}")

        # Import kaggle after checking setup
        from kaggle.api.kaggle_api_extended import KaggleApi

        # Initialize API
        api = KaggleApi()
        api.authenticate()

        # Create raw directory
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        # Download dataset
        print("\nDownloading... (this may take a few minutes)")
        api.dataset_download_files(
            self.dataset_name,
            path=str(self.raw_dir),
            unzip=True
        )

        print("✓ Download complete!")

        # List downloaded files
        files = list(self.raw_dir.glob("*"))
        print(f"\nDownloaded {len(files)} files:")
        for f in files[:10]:  # Show first 10
            print(f"  - {f.name}")
        if len(files) > 10:
            print(f"  ... and {len(files) - 10} more")

        return True

    def prepare_dataset(self):
        """Prepare dataset for training"""
        print("\n" + "="*60)
        print("PREPARING DATASET FOR TRAINING")
        print("="*60)

        # Find CSV file
        csv_files = list(self.raw_dir.glob("*.csv"))
        if not csv_files:
            print("✗ No CSV file found in downloaded data")
            return False

        csv_file = csv_files[0]
        print(f"\nFound metadata: {csv_file.name}")

        # Check if images are already downloaded or need to be downloaded from URLs
        image_files = list(self.raw_dir.glob("*.jpg")) + \
                     list(self.raw_dir.glob("*.png")) + \
                     list(self.raw_dir.glob("images/*.jpg")) + \
                     list(self.raw_dir.glob("images/*.png"))

        if image_files:
            # Images are included in the dataset
            print(f"\n✓ Found {len(image_files)} images in dataset")
            self._prepare_from_images(csv_file, image_files)
        else:
            # Need to download images from URLs in CSV
            print("\nImages not included - will download from URLs in CSV")
            self._prepare_from_urls(csv_file)

        return True

    def _prepare_from_images(self, csv_file, image_files):
        """Prepare dataset when images are included"""
        import pandas as pd
        from PIL import Image
        from tqdm import tqdm

        # Create directories
        self.images_dir.mkdir(parents=True, exist_ok=True)
        (self.training_dir / "train").mkdir(parents=True, exist_ok=True)

        # Load metadata
        df = pd.read_csv(csv_file)
        print(f"\nLoaded {len(df)} card records from CSV")

        # Copy and resize images
        print("\nPreparing images for training...")

        processed = 0
        for img_path in tqdm(image_files):
            try:
                # Open and resize
                img = Image.open(img_path).convert('RGB')

                # Resize to 256x256
                img.thumbnail((256, 256), Image.LANCZOS)
                canvas = Image.new('RGB', (256, 256), (255, 255, 255))
                offset = ((256 - img.size[0]) // 2, (256 - img.size[1]) // 2)
                canvas.paste(img, offset)

                # Save to training directory
                output_path = self.training_dir / "train" / img_path.name
                canvas.save(output_path, 'JPEG', quality=95)

                processed += 1
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")

        print(f"\n✓ Prepared {processed} images for training")

        # Save metadata
        metadata_dir = self.output_dir / "metadata"
        metadata_dir.mkdir(exist_ok=True)

        df.to_json(metadata_dir / "cards.json", orient='records', indent=2)
        df.to_csv(metadata_dir / "cards.csv", index=False)
        print(f"✓ Metadata saved to {metadata_dir}")

    def _prepare_from_urls(self, csv_file):
        """Prepare dataset by downloading images from URLs"""
        print("\nDownloading images from URLs in CSV...")

        # Use the existing download_dataset.py functionality
        from download_dataset import PokemonCardDownloader

        downloader = PokemonCardDownloader(
            csv_path=str(csv_file),
            output_dir=str(self.output_dir),
            max_workers=10
        )

        # Download images
        downloader.download_all()

        # Create metadata
        downloader.create_metadata_file()

        # Prepare for training
        if self.auto_prepare:
            downloader.prepare_for_training(output_size=256)

    def verify_dataset(self):
        """Verify the prepared dataset"""
        print("\n" + "="*60)
        print("DATASET VERIFICATION")
        print("="*60)

        training_images = list((self.training_dir / "train").glob("*.jpg"))

        if not training_images:
            print("✗ No training images found!")
            return False

        print(f"\n✓ Training images: {len(training_images)}")
        print(f"✓ Location: {self.training_dir / 'train'}")

        # Check metadata
        metadata_file = self.output_dir / "metadata" / "cards.json"
        if metadata_file.exists():
            import pandas as pd
            df = pd.read_json(metadata_file)
            print(f"✓ Metadata records: {len(df)}")

            # Show some stats
            if 'hp' in df.columns:
                print(f"\nDataset Statistics:")
                print(f"  HP range: {df['hp'].min():.0f} - {df['hp'].max():.0f}")
                print(f"  Average HP: {df['hp'].mean():.1f}")

            if 'set_name' in df.columns:
                print(f"  Unique sets: {df['set_name'].nunique()}")

        return True

    def create_quick_start_script(self):
        """Create a quick start training script"""
        train_script = self.output_dir / "quick_train.sh"

        script_content = f"""#!/bin/bash
# Quick start training script for Pokemon cards

echo "Starting Pokemon Card GAN Training"
echo "==================================="

cd {Path.cwd()}

python train_pokemon.py \\
  --data_dir {self.training_dir / 'train'} \\
  --metadata {self.output_dir / 'metadata' / 'cards.json'} \\
  --num_epochs 100 \\
  --batch_size 32 \\
  --img_size 256

echo ""
echo "Training complete!"
echo "Start the API server with: python run_server.py"
"""

        train_script.write_text(script_content)
        train_script.chmod(0o755)

        print(f"\n✓ Quick start script created: {train_script}")
        print(f"  Run with: bash {train_script}")

    def run(self):
        """Run the complete download and preparation workflow"""
        print("\n" + "="*60)
        print("POKEMON CARDS DATASET - AUTOMATIC SETUP")
        print("="*60)
        print(f"\nDataset: {self.dataset_name}")
        print(f"Output: {self.output_dir}")

        try:
            # Step 1: Check Kaggle setup
            self.check_kaggle_setup()

            # Step 2: Download dataset
            if not (self.raw_dir / "pokemon_cards.csv").exists():
                self.download_dataset()
            else:
                print(f"\n✓ Dataset already downloaded at {self.raw_dir}")

            # Step 3: Prepare dataset
            if self.auto_prepare:
                self.prepare_dataset()

            # Step 4: Verify
            if self.verify_dataset():
                # Step 5: Create quick start script
                self.create_quick_start_script()

                # Success message
                print("\n" + "="*60)
                print("SUCCESS! Dataset is ready for training")
                print("="*60)
                print("\nNext steps:")
                print(f"  1. Train the model:")
                print(f"     python train_pokemon.py --data_dir {self.training_dir / 'train'}")
                print(f"\n  2. Or use the quick start script:")
                print(f"     bash {self.output_dir / 'quick_train.sh'}")
                print(f"\n  3. After training, start the API:")
                print(f"     python run_server.py")
                print(f"\n  4. Open the UI:")
                print(f"     http://localhost:8000/ui")
                print("\n" + "="*60)

                return True
            else:
                print("\n✗ Dataset verification failed")
                return False

        except FileNotFoundError as e:
            print(f"\n✗ Setup required: {e}")
            return False
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Download Pokemon cards dataset from Kaggle'
    )
    parser.add_argument('--dataset', type=str,
                       default='priyamchoksi/pokemon-cards',
                       help='Kaggle dataset name')
    parser.add_argument('--output-dir', type=str,
                       default='data/pokemon_cards',
                       help='Output directory')
    parser.add_argument('--no-prepare', action='store_true',
                       help='Skip automatic preparation')

    args = parser.parse_args()

    # Create downloader
    downloader = KagglePokemonDownloader(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        auto_prepare=not args.no_prepare
    )

    # Run
    success = downloader.run()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
