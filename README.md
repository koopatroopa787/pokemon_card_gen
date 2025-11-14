# Pokemon Card Generator 🎴

An AI-powered Pokemon card generator using Deep Convolutional Generative Adversarial Networks (DCGAN). Train your own model on Pokemon card images and generate unique cards through a beautiful web interface.

> **NEW**: Automatically download 13,000+ Pokemon cards from Kaggle with ONE command! See [Quick Start](#quick-start) below.

## Features

- **DCGAN Model**: Deep Convolutional GAN architecture optimized for 256x256 image generation
- **Kaggle Dataset Integration**: Automatically download 13,000+ Pokemon cards from Kaggle ⭐ NEW
- **One-Command Setup**: Complete setup with `bash quick_setup.sh` ⭐ NEW
- **Metadata Support**: Use card attributes (HP, name, set, type) for enhanced training
- **REST API**: FastAPI-based backend with comprehensive endpoints
- **Modern Web UI**: Beautiful, responsive interface for card generation
- **Easy Training**: Simple training pipeline with data augmentation
- **Flexible Generation**: Generate 1-16 cards at once with optional seeding
- **Docker Support**: Easy deployment with Docker and Docker Compose

## 🚀 Quick Start (Recommended)

### Automated Setup from Kaggle

**Windows (PowerShell):**
```powershell
.\quick_setup.ps1
```

**Linux/macOS:**
```bash
bash quick_setup.sh
```

**Cross-platform (Python - Works Everywhere):**
```bash
python quick_setup.py
```

This will:
1. ✅ Install all dependencies
2. ✅ Download 13,000+ Pokemon cards from Kaggle
3. ✅ Prepare images for training
4. ✅ Create quick-start training script

**First time?** You'll need [Kaggle API credentials](KAGGLE_SETUP.md). It takes 2 minutes to set up.

See [KAGGLE_SETUP.md](KAGGLE_SETUP.md) for detailed Kaggle setup instructions.

### Manual Setup from Kaggle

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset from Kaggle
python setup_kaggle_dataset.py

# 3. Train the model
python train_pokemon.py \
  --data_dir data/pokemon_cards/training/train \
  --num_epochs 100

# 4. Start the API server
python run_server.py

# 5. Open http://localhost:8000/ui
```

### Alternative: Use Your Own Images

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Organize your images
mkdir -p data/dataset/cards
# Add your Pokemon card images to data/dataset/cards/

# 3. Train the model
python train.py --num_epochs 100 --batch_size 32

# 4. Start the API server
python run_server.py

# 5. Open http://localhost:8000/ui
```

## Project Structure

```
pokemon_card_gen/
├── model/                  # GAN model architecture
│   ├── generator.py       # Generator model
│   └── discriminator.py   # Discriminator model
├── api/                   # FastAPI backend
│   ├── main.py           # Standard API
│   └── enhanced_main.py  # API with monitoring
├── ui/                    # Web interface
│   ├── index.html
│   ├── style.css
│   └── script.js
├── utils/                 # Utilities
│   ├── config.py         # Configuration management
│   ├── metrics.py        # FID/IS evaluation
│   ├── data_utils.py     # Data preprocessing
│   └── pokemon_dataset.py # Pokemon card dataset ⭐ NEW
├── data/                  # Data directory
│   ├── dataset/          # Training images
│   ├── pokemon_cards/    # Downloaded Pokemon cards ⭐ NEW
│   └── generated/        # Generated samples
├── checkpoints/          # Model checkpoints
├── logs/                 # Training logs
├── download_dataset.py   # Dataset downloader ⭐ NEW
├── train.py             # Training script
├── train_pokemon.py     # Pokemon dataset training ⭐ NEW
├── generate.py          # Batch generation
├── requirements.txt     # Python dependencies
├── Dockerfile          # Docker configuration
├── docker-compose.yml  # Docker Compose setup
├── README.md          # This file
├── DATASET_GUIDE.md   # Pokemon dataset guide ⭐ NEW
└── README_ENHANCED.md # Complete documentation
```

## Pokemon TCG Dataset

This project now supports the official Pokemon Trading Card Game dataset with 13,000+ cards!

### Dataset Features
- **13,139 unique cards**
- **High-resolution images** (downloaded automatically)
- **Rich metadata**: HP, name, set, type, rarity, description
- **130+ card sets** from across Pokemon TCG history

### Quick Dataset Setup

```bash
# Download Pokemon cards from CSV
python download_dataset.py pokemon_cards.csv \
  --output-dir data/pokemon_cards \
  --prepare-training

# Train on Pokemon cards
python train_pokemon.py \
  --data_dir data/pokemon_cards/training/train \
  --metadata data/pokemon_cards/metadata/cards.json \
  --num_epochs 100
```

For complete dataset documentation, see [DATASET_GUIDE.md](DATASET_GUIDE.md).

## Training

### Basic Training

```bash
python train.py --num_epochs 100 --batch_size 32
```

### Training with Pokemon Dataset

```bash
python train_pokemon.py \
  --data_dir data/pokemon_cards/training/train \
  --metadata data/pokemon_cards/metadata/cards.json \
  --num_epochs 100 \
  --batch_size 32
```

### Training Options

```bash
python train.py \
  --data_dir data/dataset \
  --num_epochs 200 \
  --batch_size 16 \
  --lr 0.0002 \
  --img_size 256 \
  --latent_dim 100
```

### Resume Training

```bash
python train.py --checkpoint checkpoints/checkpoint_epoch_50.pth
```

## Generation

### Web UI

```bash
python run_server.py
# Open http://localhost:8000/ui
```

### API

```bash
# Generate 4 cards
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"num_images": 4, "seed": 42}'
```

### Batch Generation (CLI)

```bash
# Generate 100 cards
python generate.py \
  --checkpoint checkpoints/checkpoint_epoch_final.pth \
  --num-images 100 \
  --output-dir output/batch1
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/info` | GET | Model information |
| `/generate` | POST | Generate multiple cards |
| `/generate/single` | GET | Generate single card |
| `/reload` | POST | Reload model |
| `/docs` | GET | API documentation |

## Docker Deployment

### Using Docker Compose

```bash
docker-compose up -d
```

### Using Docker

```bash
docker build -t pokemon-card-gen .
docker run -p 8000:8000 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  pokemon-card-gen
```

## Advanced Features

### Dataset Validation

```bash
# Validate images
python -m utils.data_utils validate --input data/dataset

# Get statistics
python -m utils.data_utils stats --input data/dataset
```

### Model Evaluation

```python
from utils.metrics import GANMetrics

metrics = GANMetrics(device='cuda')
fid = metrics.compute_fid(real_images, generated_images)
print(f"FID: {fid:.2f}")  # Lower is better
```

### Configuration

```bash
# Copy default config
cp config.default.yaml config.yaml

# Edit config.yaml and train
python train.py  # Uses config.yaml automatically
```

## Requirements

- Python 3.8 or higher
- PyTorch 2.0+
- 4GB+ RAM (8GB+ recommended)
- CUDA-capable GPU (optional, but recommended)

See `requirements.txt` for full dependency list.

## Documentation

- **README.md** (this file) - Quick start and overview
- **KAGGLE_SETUP.md** - Kaggle dataset setup (recommended) ⭐ NEW
- **DATASET_GUIDE.md** - Pokemon TCG dataset guide
- **README_ENHANCED.md** - Complete feature documentation
- **SETUP.md** - Detailed setup instructions
- **CONTRIBUTING.md** - Contribution guidelines

## Testing

```bash
# Run tests
pytest

# With coverage
pytest --cov=. --cov-report=html
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- DCGAN architecture based on [Radford et al. (2015)](https://arxiv.org/abs/1511.06434)
- Built with [PyTorch](https://pytorch.org/) and [FastAPI](https://fastapi.tiangolo.com/)
- Pokemon cards are © Nintendo/Game Freak/Pokemon Company
- Dataset integration inspired by [Pokemon TCG](https://pokemontcg.io/)

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/pokemon_card_gen/issues)
- **Documentation**: See docs/ directory
- **Dataset Help**: See [DATASET_GUIDE.md](DATASET_GUIDE.md)

---

**Made with ❤️ for Pokemon fans and AI enthusiasts**

If you found this project helpful, please consider giving it a ⭐!
