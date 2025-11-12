"""
Data preprocessing and validation utilities
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional
import shutil

from PIL import Image
import numpy as np
from tqdm import tqdm


class DatasetValidator:
    """Validate and preprocess image datasets"""

    def __init__(self, min_size: int = 64, max_size: int = 4096):
        """
        Initialize validator

        Args:
            min_size: Minimum image dimension
            max_size: Maximum image dimension
        """
        self.min_size = min_size
        self.max_size = max_size
        self.valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    def validate_image(self, image_path: Path) -> Tuple[bool, str]:
        """
        Validate a single image

        Args:
            image_path: Path to image

        Returns:
            (is_valid, error_message)
        """
        # Check extension
        if image_path.suffix.lower() not in self.valid_extensions:
            return False, f"Invalid extension: {image_path.suffix}"

        try:
            # Try to open image
            with Image.open(image_path) as img:
                # Check if corrupted
                img.verify()

            # Reopen for further checks
            with Image.open(image_path) as img:
                width, height = img.size

                # Check dimensions
                if width < self.min_size or height < self.min_size:
                    return False, f"Image too small: {width}x{height}"

                if width > self.max_size or height > self.max_size:
                    return False, f"Image too large: {width}x{height}"

                # Check mode (convert if needed)
                if img.mode not in ['RGB', 'RGBA', 'L']:
                    return False, f"Unsupported mode: {img.mode}"

                return True, "OK"

        except Exception as e:
            return False, f"Error opening image: {str(e)}"

    def validate_dataset(self, dataset_dir: str, verbose: bool = True) -> dict:
        """
        Validate entire dataset

        Args:
            dataset_dir: Directory containing images
            verbose: Print progress

        Returns:
            Dictionary with validation results
        """
        dataset_path = Path(dataset_dir)

        if not dataset_path.exists():
            return {
                'total': 0,
                'valid': 0,
                'invalid': 0,
                'errors': [f"Dataset directory not found: {dataset_dir}"]
            }

        # Find all images
        image_files = []
        for ext in self.valid_extensions:
            image_files.extend(dataset_path.rglob(f"*{ext}"))
            image_files.extend(dataset_path.rglob(f"*{ext.upper()}"))

        results = {
            'total': len(image_files),
            'valid': 0,
            'invalid': 0,
            'errors': [],
            'valid_files': [],
            'invalid_files': []
        }

        # Validate each image
        iterator = tqdm(image_files, desc="Validating images") if verbose else image_files

        for img_path in iterator:
            is_valid, message = self.validate_image(img_path)

            if is_valid:
                results['valid'] += 1
                results['valid_files'].append(str(img_path))
            else:
                results['invalid'] += 1
                results['invalid_files'].append(str(img_path))
                results['errors'].append(f"{img_path}: {message}")

        return results

    def clean_dataset(
        self,
        dataset_dir: str,
        output_dir: Optional[str] = None,
        remove_invalid: bool = False
    ) -> dict:
        """
        Clean dataset by removing/moving invalid images

        Args:
            dataset_dir: Source directory
            output_dir: Output directory (if None, clean in place)
            remove_invalid: If True, delete invalid images; else move to 'invalid' folder

        Returns:
            Cleaning results
        """
        results = self.validate_dataset(dataset_dir, verbose=True)

        if remove_invalid and results['invalid'] > 0:
            print(f"\nRemoving {results['invalid']} invalid images...")

            for invalid_file in tqdm(results['invalid_files'], desc="Removing"):
                try:
                    os.remove(invalid_file)
                except Exception as e:
                    print(f"Error removing {invalid_file}: {e}")

        elif not remove_invalid and results['invalid'] > 0:
            invalid_dir = Path(dataset_dir) / "invalid"
            invalid_dir.mkdir(exist_ok=True)

            print(f"\nMoving {results['invalid']} invalid images to {invalid_dir}...")

            for invalid_file in tqdm(results['invalid_files'], desc="Moving"):
                try:
                    shutil.move(
                        invalid_file,
                        invalid_dir / Path(invalid_file).name
                    )
                except Exception as e:
                    print(f"Error moving {invalid_file}: {e}")

        return results


class ImagePreprocessor:
    """Preprocess images for training"""

    def __init__(self, target_size: int = 256):
        """
        Initialize preprocessor

        Args:
            target_size: Target size for images
        """
        self.target_size = target_size

    def preprocess_image(self, image_path: Path, output_path: Path):
        """
        Preprocess a single image

        Args:
            image_path: Input image path
            output_path: Output image path
        """
        with Image.open(image_path) as img:
            # Convert to RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Resize maintaining aspect ratio
            img.thumbnail((self.target_size, self.target_size), Image.LANCZOS)

            # Create square canvas
            canvas = Image.new('RGB', (self.target_size, self.target_size), (255, 255, 255))

            # Paste image in center
            offset = ((self.target_size - img.size[0]) // 2,
                     (self.target_size - img.size[1]) // 2)
            canvas.paste(img, offset)

            # Save
            canvas.save(output_path, 'JPEG', quality=95)

    def preprocess_dataset(
        self,
        input_dir: str,
        output_dir: str,
        organize_subdirs: bool = True
    ):
        """
        Preprocess entire dataset

        Args:
            input_dir: Input directory
            output_dir: Output directory
            organize_subdirs: Organize into train subdirectory
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)

        if organize_subdirs:
            output_path = output_path / "train"

        output_path.mkdir(parents=True, exist_ok=True)

        # Find all images
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        image_files = []
        for ext in valid_extensions:
            image_files.extend(input_path.rglob(f"*{ext}"))
            image_files.extend(input_path.rglob(f"*{ext.upper()}"))

        print(f"Found {len(image_files)} images to preprocess")

        # Process each image
        for img_path in tqdm(image_files, desc="Preprocessing"):
            try:
                output_file = output_path / f"{img_path.stem}.jpg"
                self.preprocess_image(img_path, output_file)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        print(f"Preprocessing complete! Output: {output_path}")


def get_dataset_statistics(dataset_dir: str) -> dict:
    """
    Get statistics about a dataset

    Args:
        dataset_dir: Dataset directory

    Returns:
        Dictionary with statistics
    """
    dataset_path = Path(dataset_dir)
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    # Find all images
    image_files = []
    for ext in valid_extensions:
        image_files.extend(dataset_path.rglob(f"*{ext}"))
        image_files.extend(dataset_path.rglob(f"*{ext.upper()}"))

    if not image_files:
        return {
            'total_images': 0,
            'error': 'No images found'
        }

    sizes = []
    aspects = []
    modes = []

    for img_path in tqdm(image_files, desc="Analyzing dataset"):
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                sizes.append((width, height))
                aspects.append(width / height)
                modes.append(img.mode)
        except:
            continue

    if not sizes:
        return {
            'total_images': len(image_files),
            'error': 'Could not open any images'
        }

    widths, heights = zip(*sizes)

    stats = {
        'total_images': len(image_files),
        'dimensions': {
            'min_width': min(widths),
            'max_width': max(widths),
            'avg_width': int(np.mean(widths)),
            'min_height': min(heights),
            'max_height': max(heights),
            'avg_height': int(np.mean(heights)),
        },
        'aspect_ratios': {
            'min': min(aspects),
            'max': max(aspects),
            'avg': np.mean(aspects),
        },
        'color_modes': {mode: modes.count(mode) for mode in set(modes)},
    }

    return stats


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Dataset utilities')
    parser.add_argument('command', choices=['validate', 'clean', 'preprocess', 'stats'],
                       help='Command to run')
    parser.add_argument('--input', required=True, help='Input directory')
    parser.add_argument('--output', help='Output directory (for preprocess/clean)')
    parser.add_argument('--target-size', type=int, default=256,
                       help='Target size for preprocessing')
    parser.add_argument('--remove-invalid', action='store_true',
                       help='Remove invalid images (for clean)')

    args = parser.parse_args()

    if args.command == 'validate':
        validator = DatasetValidator()
        results = validator.validate_dataset(args.input)
        print(f"\nValidation Results:")
        print(f"Total: {results['total']}")
        print(f"Valid: {results['valid']}")
        print(f"Invalid: {results['invalid']}")
        if results['errors']:
            print(f"\nErrors:")
            for error in results['errors'][:10]:  # Show first 10
                print(f"  {error}")

    elif args.command == 'clean':
        validator = DatasetValidator()
        results = validator.clean_dataset(
            args.input,
            args.output,
            remove_invalid=args.remove_invalid
        )
        print(f"\nCleaning complete!")
        print(f"Valid images: {results['valid']}")
        print(f"Invalid images: {results['invalid']}")

    elif args.command == 'preprocess':
        if not args.output:
            print("Error: --output required for preprocess")
            exit(1)

        preprocessor = ImagePreprocessor(target_size=args.target_size)
        preprocessor.preprocess_dataset(args.input, args.output)

    elif args.command == 'stats':
        stats = get_dataset_statistics(args.input)
        print(f"\nDataset Statistics:")
        print(f"Total images: {stats['total_images']}")
        if 'error' not in stats:
            print(f"\nDimensions:")
            for key, value in stats['dimensions'].items():
                print(f"  {key}: {value}")
            print(f"\nAspect Ratios:")
            for key, value in stats['aspect_ratios'].items():
                print(f"  {key}: {value:.2f}")
            print(f"\nColor Modes:")
            for mode, count in stats['color_modes'].items():
                print(f"  {mode}: {count}")
