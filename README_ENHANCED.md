# Pokemon Card Generator 🎴✨

[![CI/CD](https://github.com/yourusername/pokemon_card_gen/workflows/CI%2FCD%20Pipeline/badge.svg)](https://github.com/yourusername/pokemon_card_gen/actions)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-ready, state-of-the-art AI system for generating unique Pokemon cards using Deep Convolutional Generative Adversarial Networks (DCGAN). Features include a modern web UI, RESTful API, comprehensive testing, and professional development tools.

## 🌟 Features

### Core Functionality
- **Advanced GAN Architecture**: DCGAN optimized for 256x256 high-quality image generation
- **Flexible Training Pipeline**: Full training script with checkpointing, resumption, and monitoring
- **REST API**: FastAPI-based backend with rate limiting and health checks
- **Modern Web UI**: Beautiful, responsive interface with real-time generation
- **Batch Generation**: Generate 1-16 cards simultaneously with optional seeding

### Professional Tools
- **Configuration Management**: YAML-based configuration system
- **Model Evaluation**: FID and Inception Score metrics
- **Data Utilities**: Dataset validation, preprocessing, and statistics
- **Batch Inference**: CLI tool for generating large batches
- **Comprehensive Testing**: Unit tests for models and API
- **CI/CD Pipeline**: GitHub Actions for automated testing
- **Docker Support**: One-command deployment with Docker Compose
- **Monitoring**: Health checks, metrics, and usage statistics

## 📁 Project Structure

```
pokemon_card_gen/
├── model/                    # GAN architecture
│   ├── generator.py         # Generator network
│   └── discriminator.py     # Discriminator network
├── api/                      # Backend API
│   ├── main.py              # Standard API
│   └── enhanced_main.py     # API with rate limiting & monitoring
├── ui/                       # Web interface
│   ├── index.html
│   ├── style.css
│   └── script.js
├── utils/                    # Utilities
│   ├── config.py            # Configuration management
│   ├── metrics.py           # FID/IS evaluation
│   └── data_utils.py        # Data preprocessing
├── tests/                    # Unit tests
│   ├── test_models.py
│   └── test_api.py
├── .github/                  # CI/CD and templates
│   ├── workflows/ci.yml
│   └── ISSUE_TEMPLATE/
├── data/                     # Data directories
│   ├── dataset/             # Training images
│   └── generated/           # Generated samples
├── checkpoints/             # Model checkpoints
├── logs/                    # Training logs
├── train.py                 # Training script
├── generate.py              # Batch inference script
├── config.default.yaml      # Default configuration
├── requirements.txt         # Python dependencies
├── requirements-dev.txt     # Development dependencies
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose setup
├── CONTRIBUTING.md         # Contribution guidelines
└── README.md              # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- PyTorch 2.0+
- 4GB+ RAM (8GB+ recommended)
- CUDA-capable GPU (optional, but recommended for training)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pokemon_card_gen.git
cd pokemon_card_gen

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Training

```bash
# 1. Prepare your dataset
mkdir -p data/dataset/cards
# Add your Pokemon card images to data/dataset/cards/

# 2. Train the model
python train.py --num_epochs 100 --batch_size 32

# 3. Start the API server
python run_server.py

# 4. Open the web UI
# Navigate to http://localhost:8000/ui
```

## 📖 Detailed Usage

### Training

#### Basic Training
```bash
python train.py --num_epochs 100 --batch_size 32
```

#### Advanced Training with Configuration
```bash
# Copy and edit the configuration file
cp config.default.yaml config.yaml
nano config.yaml

# Train with custom config
python train.py --config config.yaml
```

#### Training Options
```bash
python train.py \
  --data_dir data/dataset \
  --num_epochs 200 \
  --batch_size 16 \
  --lr 0.0002 \
  --img_size 256 \
  --latent_dim 100 \
  --checkpoint checkpoints/checkpoint_epoch_50.pth  # Resume training
```

### Generation

#### Web UI
```bash
python run_server.py
# Open http://localhost:8000/ui
```

#### API Usage
```bash
# Generate 4 cards
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"num_images": 4, "seed": 42}'

# Generate single card as image
curl "http://localhost:8000/generate/single?seed=123" -o card.png
```

#### Batch Generation (CLI)
```bash
# Generate 100 cards
python generate.py \
  --checkpoint checkpoints/checkpoint_epoch_final.pth \
  --num-images 100 \
  --output-dir output/batch1

# Generate interpolation between two cards
python generate.py \
  --checkpoint checkpoints/checkpoint_epoch_final.pth \
  --mode interpolate \
  --num-steps 20 \
  --seed-start 42 \
  --seed-end 123
```

### Data Preprocessing

```bash
# Validate dataset
python -m utils.data_utils validate --input data/raw_images

# Get dataset statistics
python -m utils.data_utils stats --input data/raw_images

# Preprocess and resize images
python -m utils.data_utils preprocess \
  --input data/raw_images \
  --output data/dataset \
  --target-size 256
```

### Model Evaluation

```python
from utils.metrics import GANMetrics
import torch

metrics = GANMetrics(device='cuda')

# Compute FID score
fid = metrics.compute_fid(real_images, generated_images)
print(f"FID: {fid:.2f}")

# Compute Inception Score
mean_is, std_is = metrics.compute_is(generated_images)
print(f"IS: {mean_is:.2f} ± {std_is:.2f}")
```

## 🔧 Configuration

Edit `config.yaml` to customize:

```yaml
model:
  latent_dim: 100
  img_size: 256

training:
  batch_size: 32
  num_epochs: 100
  learning_rate: 0.0002
  use_amp: true  # Mixed precision training

data:
  dataset_dir: "data/dataset"
  augmentation:
    horizontal_flip: true
    color_jitter: true

logging:
  use_tensorboard: true
  save_interval: 10

api:
  port: 8000
  enable_rate_limit: true
  rate_limit_per_minute: 30
```

## 🧪 Testing

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_models.py

# Run linting
flake8 .
black --check .
isort --check .
```

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Using Docker

```bash
# Build image
docker build -t pokemon-card-gen .

# Run container
docker run -p 8000:8000 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/data:/app/data \
  pokemon-card-gen
```

## 📊 API Endpoints

### Standard API (`main.py`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/info` | GET | Model information |
| `/generate` | POST | Generate multiple cards |
| `/generate/single` | GET | Generate single card |
| `/reload` | POST | Reload model |
| `/docs` | GET | API documentation |

### Enhanced API (`enhanced_main.py`)

Includes all standard endpoints plus:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check for monitoring |
| `/stats` | GET | Usage statistics |

**Rate Limiting**: 30 requests per minute per IP (configurable)

## 🎯 Performance Tips

### Training
- Use GPU for 10-50x speed improvement
- Increase batch size if you have enough VRAM
- Use mixed precision training (`use_amp: true`) for 2x speed on modern GPUs
- Monitor training with TensorBoard: `tensorboard --logdir runs`

### Generation
- Batch generation is more efficient than individual requests
- Use seeding for reproducible results
- GPU generation is much faster for large batches

### Production
- Use the enhanced API for rate limiting and monitoring
- Set up health checks: `curl http://localhost:8000/health`
- Monitor metrics: `curl http://localhost:8000/stats`
- Use Docker for consistent deployment

## 📈 Model Evaluation Metrics

### Frechet Inception Distance (FID)
- Measures similarity between real and generated image distributions
- **Lower is better**
- Good FID: < 50, Excellent FID: < 20

### Inception Score (IS)
- Measures quality and diversity of generated images
- **Higher is better**
- Range: 1-10, Good IS: > 3

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Quick Contribution Steps
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Commit (`git commit -m 'feat: add amazing feature'`)
6. Push (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- DCGAN architecture based on [Radford et al. (2015)](https://arxiv.org/abs/1511.06434)
- Built with [PyTorch](https://pytorch.org/), [FastAPI](https://fastapi.tiangolo.com/), and [React](https://reactjs.org/)
- Inspired by the Pokemon Trading Card Game

## 📧 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/pokemon_card_gen/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/pokemon_card_gen/discussions)
- **Email**: your.email@example.com

## 🗺️ Roadmap

- [ ] StyleGAN2 architecture implementation
- [ ] Conditional generation (by type, rarity, etc.)
- [ ] Text-to-image generation
- [ ] Image-to-image translation
- [ ] Model zoo with pretrained weights
- [ ] Web-based training interface
- [ ] Advanced editing tools
- [ ] Community gallery

---

**Made with ❤️ for Pokemon fans and AI enthusiasts**

If you found this project helpful, please consider giving it a ⭐!
