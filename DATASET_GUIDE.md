# Pokemon Card Dataset Guide

This guide explains how to use the Pokemon Trading Card Game dataset with this generator.

## Dataset Format

The Pokemon card dataset should be a CSV file with the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| `id` | Unique card identifier | `pl3-1` |
| `image_url` | URL to high-resolution image | `https://images.pokemontcg.io/pl3/1_hires.png` |
| `name` | Pokemon name | `Absol G` |
| `hp` | Health points | `70` |
| `caption` | Card description | `A Basic, SP Pokemon Card of type Darkness...` |
| `set_name` | Set name | `Supreme Victors` |

### Dataset Statistics
- **Total cards**: 13,139
- **Unique Pokemon**: ~2,459
- **HP range**: 30-340
- **Unique sets**: ~130

## Quick Start

### Step 1: Download the Dataset

```bash
# Download images from CSV
python download_dataset.py pokemon_cards.csv \
  --output-dir data/pokemon_cards \
  --max-workers 10 \
  --prepare-training

# Options:
# --limit 100              # Download only first 100 (for testing)
# --organize-by-set        # Organize into subdirectories by set
# --target-size 256        # Resize images to 256x256
```

This will:
1. Download all card images from URLs
2. Save metadata as JSON
3. Prepare images for training (resize, organize)

### Step 2: Verify the Download

```bash
# Check dataset statistics
python -m utils.data_utils stats \
  --input data/pokemon_cards/images

# Validate images
python -m utils.data_utils validate \
  --input data/pokemon_cards/images
```

### Step 3: Train the Model

#### Option A: Basic Training (No Metadata)

```bash
python train.py \
  --data_dir data/pokemon_cards/training \
  --num_epochs 100 \
  --batch_size 32
```

#### Option B: Training with Custom Dataset (With Metadata)

```bash
python train_pokemon.py \
  --data_dir data/pokemon_cards/images \
  --metadata data/pokemon_cards/metadata/cards.json \
  --num_epochs 100 \
  --batch_size 32
```

## Dataset Structure

After downloading, your directory structure will look like:

```
data/pokemon_cards/
├── images/                    # All downloaded images
│   ├── pl3-1.jpg
│   ├── pl3-2.jpg
│   └── ...
├── metadata/                  # Card metadata
│   ├── cards.json            # Full metadata as JSON
│   └── cards.csv             # Full metadata as CSV
├── by_set/                   # Optional: Organized by set
│   ├── Supreme_Victors/
│   ├── Base_Set/
│   └── ...
└── training/                 # Prepared for training
    └── train/                # Training images (256x256)
        ├── pl3-1.jpg
        └── ...
```

## Download Options

### Test with Limited Dataset

```bash
# Download only 100 cards for testing
python download_dataset.py pokemon_cards.csv \
  --limit 100 \
  --output-dir data/test_download
```

### Organize by Pokemon Set

```bash
# Organize images into subdirectories by set name
python download_dataset.py pokemon_cards.csv \
  --organize-by-set
```

This creates directories like:
- `data/pokemon_cards/by_set/Base_Set/`
- `data/pokemon_cards/by_set/Jungle/`
- `data/pokemon_cards/by_set/Team_Rocket/`

### Custom Image Size

```bash
# Prepare images at different resolution
python download_dataset.py pokemon_cards.csv \
  --prepare-training \
  --target-size 512  # 512x512 images
```

## Using Metadata in Training

The dataset includes rich metadata that can be used for:

### 1. Filtering by HP Range

```python
from utils.pokemon_dataset import PokemonCardDataset
import pandas as pd

# Load metadata
metadata = pd.read_json('data/pokemon_cards/metadata/cards.json')

# Filter high HP cards
high_hp = metadata[metadata['hp'] > 100]
print(f"Cards with HP > 100: {len(high_hp)}")

# Save filtered IDs
high_hp['id'].to_csv('high_hp_cards.txt', index=False, header=False)
```

### 2. Training on Specific Sets

```python
# Filter by set
base_set = metadata[metadata['set_name'] == 'Base Set']

# Train only on Base Set cards
# (Manually copy these images to a separate directory)
```

### 3. Conditional Generation (Advanced)

Use metadata for conditional GAN training:

```python
from utils.pokemon_dataset import get_pokemon_dataloader

# Create conditional dataloader
dataloader = get_pokemon_dataloader(
    data_dir='data/pokemon_cards/images',
    metadata_file='data/pokemon_cards/metadata/cards.json',
    conditional=True,
    condition_dim=20,
    batch_size=32
)

# Each batch includes images and condition vectors (HP, set, etc.)
for images, conditions in dataloader:
    print(f"Images: {images.shape}")
    print(f"Conditions: {conditions.shape}")  # [batch, 20]
    break
```

## Data Preprocessing

### Clean and Validate

```bash
# Validate all downloaded images
python -m utils.data_utils validate \
  --input data/pokemon_cards/images

# Clean dataset (remove/move invalid images)
python -m utils.data_utils clean \
  --input data/pokemon_cards/images
```

### Analyze Dataset

```bash
# Get detailed statistics
python -m utils.data_utils stats \
  --input data/pokemon_cards/images
```

This shows:
- Total images
- Image dimensions (min, max, average)
- Aspect ratios
- Color modes
- File sizes

## Troubleshooting

### Download Failures

If some downloads fail:

```bash
# Re-run the download script
# It will skip already downloaded images
python download_dataset.py pokemon_cards.csv \
  --output-dir data/pokemon_cards
```

### Slow Downloads

Adjust the number of workers and delay:

```bash
# More workers = faster, but may trigger rate limits
python download_dataset.py pokemon_cards.csv \
  --max-workers 20 \
  --delay 0.05  # Reduce delay between requests
```

### Out of Disk Space

Download in batches:

```bash
# Download first 1000
python download_dataset.py pokemon_cards.csv \
  --limit 1000 \
  --output-dir data/batch1

# Then next 1000
# (Modify CSV to skip first 1000 rows)
```

### Image Quality Issues

Some cards may be:
- Different sizes
- Different aspect ratios
- Low quality

Use the preprocessing tool:

```bash
python -m utils.data_utils preprocess \
  --input data/pokemon_cards/images \
  --output data/pokemon_cards/cleaned \
  --target-size 256
```

## Advanced Usage

### Custom Dataset Class

```python
from utils.pokemon_dataset import PokemonCardDataset

# Create custom dataset
dataset = PokemonCardDataset(
    data_dir='data/pokemon_cards/images',
    metadata_file='data/pokemon_cards/metadata/cards.json',
    use_metadata=True,
    img_size=256
)

# Access image and metadata
image, metadata = dataset[0]
print(f"HP: {metadata['hp']}")
print(f"Name: {metadata['name']}")
print(f"Set: {metadata['set_name']}")
```

### Filter by Criteria

```python
import pandas as pd

# Load metadata
df = pd.read_json('data/pokemon_cards/metadata/cards.json')

# Filter examples
rare_cards = df[df['caption'].str.contains('Rare Holo')]
high_hp_fire = df[(df['hp'] > 120) & (df['caption'].str.contains('Fire'))]
base_set_only = df[df['set_name'] == 'Base Set']

# Save filtered lists
rare_cards['id'].to_csv('rare_cards.txt', index=False, header=False)
```

## Dataset Recommendations

### For Best Results

1. **Minimum dataset size**: 500+ cards
2. **Recommended size**: 2,000+ cards
3. **Optimal size**: 5,000+ cards (use full dataset!)

### Training Tips

1. **Start with a subset**: Test with 100-500 cards first
2. **Check quality**: Validate downloads before training
3. **Balance dataset**: Consider HP distribution, sets, types
4. **Use full resolution**: Download highest quality images available

## Example Workflow

Complete workflow from CSV to trained model:

```bash
# 1. Download dataset
python download_dataset.py pokemon_cards.csv \
  --output-dir data/pokemon_cards \
  --prepare-training

# 2. Validate downloads
python -m utils.data_utils validate \
  --input data/pokemon_cards/training/train

# 3. Get statistics
python -m utils.data_utils stats \
  --input data/pokemon_cards/training/train

# 4. Train model
python train.py \
  --data_dir data/pokemon_cards/training \
  --num_epochs 100 \
  --batch_size 32 \
  --img_size 256

# 5. Generate cards
python generate.py \
  --checkpoint checkpoints/checkpoint_epoch_final.pth \
  --num-images 16 \
  --output-dir generated_cards
```

## Dataset Sources

Where to get the Pokemon card CSV:
1. **Kaggle**: Search for "Pokemon Trading Card Game" datasets
2. **Pokemon TCG API**: https://pokemontcg.io/
3. **Custom scraping**: Use the TCG database with proper attribution

## Legal Notes

- Respect copyright and terms of service
- Pokemon cards are © Nintendo/Game Freak/Pokemon Company
- Use for educational and research purposes
- Don't redistribute card images without permission
- Check the dataset license before using

## Need Help?

- Check the main README.md for general setup
- See SETUP.md for troubleshooting
- Open an issue for dataset-specific problems
- Read the API docs for programmatic access

---

Happy training! 🎴✨
