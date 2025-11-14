"""
Pokemon Card Dataset with Metadata Support

Custom PyTorch Dataset for Pokemon cards with attributes like HP, name, type, etc.
Supports both unconditional and conditional generation.
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np


class PokemonCardDataset(Dataset):
    """
    Pokemon Card Dataset with metadata

    Supports loading images along with metadata (HP, name, caption, set)
    for conditional generation tasks.
    """

    def __init__(
        self,
        data_dir: str,
        metadata_file: Optional[str] = None,
        transform: Optional[transforms.Compose] = None,
        img_size: int = 256,
        use_metadata: bool = False,
        normalize_hp: bool = True
    ):
        """
        Initialize Pokemon Card Dataset

        Args:
            data_dir: Directory containing images
            metadata_file: Optional JSON/CSV file with card metadata
            transform: Optional torchvision transforms
            img_size: Image size (for default transform)
            use_metadata: Whether to return metadata with images
            normalize_hp: Normalize HP values to [0, 1]
        """
        self.data_dir = Path(data_dir)
        self.use_metadata = use_metadata
        self.normalize_hp = normalize_hp

        # Find all images
        self.image_files = self._find_images()

        # Load metadata if provided
        self.metadata = None
        if metadata_file and Path(metadata_file).exists():
            self.metadata = self._load_metadata(metadata_file)

        # Set up transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transform

        print(f"Loaded {len(self.image_files)} Pokemon card images")
        if self.metadata is not None:
            print(f"Loaded metadata for {len(self.metadata)} cards")

    def _find_images(self) -> List[Path]:
        """Find all image files in directory"""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = []

        for ext in valid_extensions:
            image_files.extend(self.data_dir.rglob(f"*{ext}"))

        return sorted(image_files)

    def _load_metadata(self, metadata_file: str) -> pd.DataFrame:
        """Load metadata from JSON or CSV"""
        metadata_path = Path(metadata_file)

        if metadata_path.suffix == '.json':
            df = pd.read_json(metadata_path)
        elif metadata_path.suffix == '.csv':
            df = pd.read_csv(metadata_path)
        else:
            raise ValueError(f"Unsupported metadata format: {metadata_path.suffix}")

        # Set index to id for quick lookup
        if 'id' in df.columns:
            df = df.set_index('id')

        return df

    def _get_card_id_from_filename(self, filepath: Path) -> str:
        """Extract card ID from filename"""
        return filepath.stem

    def _get_metadata_for_card(self, card_id: str) -> Optional[Dict]:
        """Get metadata for a specific card"""
        if self.metadata is None:
            return None

        try:
            if card_id in self.metadata.index:
                return self.metadata.loc[card_id].to_dict()
        except:
            pass

        return None

    def _process_metadata(self, metadata: Dict) -> Dict[str, torch.Tensor]:
        """
        Process metadata into tensor format

        Returns:
            Dictionary with processed metadata tensors
        """
        processed = {}

        # HP (normalized to [0, 1])
        if 'hp' in metadata:
            hp = float(metadata['hp']) if metadata['hp'] else 0.0
            if self.normalize_hp:
                # Normalize HP (typical range: 30-340)
                hp = (hp - 30.0) / (340.0 - 30.0)
                hp = np.clip(hp, 0.0, 1.0)
            processed['hp'] = torch.tensor(hp, dtype=torch.float32)

        # Name (as string, could be embedded later)
        if 'name' in metadata:
            processed['name'] = str(metadata['name'])

        # Set name
        if 'set_name' in metadata:
            processed['set_name'] = str(metadata['set_name'])

        # Caption (for potential text conditioning)
        if 'caption' in metadata:
            processed['caption'] = str(metadata['caption'])

        return processed

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Get item from dataset

        Returns:
            If use_metadata is False: (image_tensor,)
            If use_metadata is True: (image_tensor, metadata_dict)
        """
        # Load image
        img_path = self.image_files[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return black image as fallback
            image = torch.zeros(3, 256, 256)

        # Return without metadata if not needed
        if not self.use_metadata:
            return image, 0  # Return dummy label for compatibility

        # Get metadata
        card_id = self._get_card_id_from_filename(img_path)
        metadata = self._get_metadata_for_card(card_id)

        if metadata:
            processed_metadata = self._process_metadata(metadata)
        else:
            # Return empty metadata if not found
            processed_metadata = {
                'hp': torch.tensor(0.0, dtype=torch.float32),
                'name': 'Unknown',
                'set_name': 'Unknown',
                'caption': ''
            }

        return image, processed_metadata


class ConditionalPokemonDataset(PokemonCardDataset):
    """
    Pokemon Card Dataset for Conditional GAN

    Returns images along with condition vectors (HP, type embeddings, etc.)
    """

    def __init__(
        self,
        data_dir: str,
        metadata_file: str,
        transform: Optional[transforms.Compose] = None,
        img_size: int = 256,
        condition_dim: int = 10
    ):
        """
        Initialize Conditional Pokemon Dataset

        Args:
            data_dir: Directory containing images
            metadata_file: JSON/CSV file with card metadata (required)
            transform: Optional torchvision transforms
            img_size: Image size
            condition_dim: Dimension of condition vector
        """
        super().__init__(
            data_dir=data_dir,
            metadata_file=metadata_file,
            transform=transform,
            img_size=img_size,
            use_metadata=True,
            normalize_hp=True
        )

        self.condition_dim = condition_dim

        # Build set name to index mapping
        if self.metadata is not None and 'set_name' in self.metadata.columns:
            unique_sets = self.metadata['set_name'].unique()
            self.set_to_idx = {name: idx for idx, name in enumerate(unique_sets)}
            self.num_sets = len(unique_sets)
            print(f"Found {self.num_sets} unique sets")
        else:
            self.set_to_idx = {}
            self.num_sets = 0

    def _create_condition_vector(self, metadata: Dict) -> torch.Tensor:
        """
        Create condition vector from metadata

        The condition vector includes:
        - HP (normalized): 1 dimension
        - Set one-hot encoding: num_sets dimensions
        - Padding to reach condition_dim

        Returns:
            Condition vector of shape (condition_dim,)
        """
        condition_parts = []

        # Add HP
        if 'hp' in metadata:
            condition_parts.append(metadata['hp'].unsqueeze(0))
        else:
            condition_parts.append(torch.tensor([0.0]))

        # Add set one-hot encoding (if available)
        if self.num_sets > 0 and 'set_name' in metadata:
            set_name = metadata['set_name']
            set_idx = self.set_to_idx.get(set_name, 0)

            # Create one-hot vector (limit to reasonable size)
            max_sets = min(self.num_sets, self.condition_dim - 1)
            one_hot = torch.zeros(max_sets)
            if set_idx < max_sets:
                one_hot[set_idx] = 1.0

            condition_parts.append(one_hot)

        # Concatenate all parts
        condition_vector = torch.cat(condition_parts)

        # Pad or truncate to condition_dim
        if len(condition_vector) < self.condition_dim:
            padding = torch.zeros(self.condition_dim - len(condition_vector))
            condition_vector = torch.cat([condition_vector, padding])
        elif len(condition_vector) > self.condition_dim:
            condition_vector = condition_vector[:self.condition_dim]

        return condition_vector

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item with condition vector

        Returns:
            (image_tensor, condition_vector)
        """
        image, metadata = super().__getitem__(idx)

        if isinstance(metadata, dict):
            condition = self._create_condition_vector(metadata)
        else:
            # Fallback to zero condition
            condition = torch.zeros(self.condition_dim)

        return image, condition


def get_pokemon_dataloader(
    data_dir: str,
    metadata_file: Optional[str] = None,
    batch_size: int = 32,
    img_size: int = 256,
    num_workers: int = 4,
    shuffle: bool = True,
    use_metadata: bool = False,
    conditional: bool = False,
    condition_dim: int = 10
):
    """
    Create DataLoader for Pokemon cards

    Args:
        data_dir: Directory containing images
        metadata_file: Optional metadata JSON/CSV
        batch_size: Batch size
        img_size: Image size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        use_metadata: Return metadata with images
        conditional: Use conditional dataset
        condition_dim: Dimension of condition vector (for conditional)

    Returns:
        DataLoader
    """
    from torch.utils.data import DataLoader

    if conditional:
        if not metadata_file:
            raise ValueError("metadata_file required for conditional dataset")

        dataset = ConditionalPokemonDataset(
            data_dir=data_dir,
            metadata_file=metadata_file,
            img_size=img_size,
            condition_dim=condition_dim
        )
    else:
        dataset = PokemonCardDataset(
            data_dir=data_dir,
            metadata_file=metadata_file,
            img_size=img_size,
            use_metadata=use_metadata
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return dataloader


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description='Test Pokemon Card Dataset')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory with images')
    parser.add_argument('--metadata', type=str, default=None,
                       help='Metadata file (JSON/CSV)')
    parser.add_argument('--conditional', action='store_true',
                       help='Use conditional dataset')

    args = parser.parse_args()

    # Create dataset
    if args.conditional:
        dataset = ConditionalPokemonDataset(
            data_dir=args.data_dir,
            metadata_file=args.metadata
        )
    else:
        dataset = PokemonCardDataset(
            data_dir=args.data_dir,
            metadata_file=args.metadata,
            use_metadata=True
        )

    print(f"\nDataset size: {len(dataset)}")

    # Test loading
    print("\nTesting data loading...")
    image, metadata = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Metadata: {metadata}")

    # Test dataloader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for batch_images, batch_metadata in dataloader:
        print(f"\nBatch images shape: {batch_images.shape}")
        if args.conditional:
            print(f"Batch conditions shape: {batch_metadata.shape}")
        else:
            print(f"Batch metadata: {batch_metadata}")
        break

    print("\nDataset test complete!")
