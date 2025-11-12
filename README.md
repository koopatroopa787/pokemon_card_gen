# Pokemon Card Generator 🎴

An AI-powered Pokemon card generator using Deep Convolutional Generative Adversarial Networks (DCGAN). Train your own model on Pokemon card images and generate unique cards through a beautiful web interface.

## Features

- **DCGAN Model**: Deep Convolutional GAN architecture optimized for 256x256 image generation
- **REST API**: FastAPI-based backend with comprehensive endpoints
- **Modern Web UI**: Beautiful, responsive interface for card generation
- **Easy Training**: Simple training pipeline with data augmentation
- **Flexible Generation**: Generate 1-16 cards at once with optional seeding
- **Docker Support**: Easy deployment with Docker and Docker Compose

## Project Structure

```
pokemon_card_gen/
├── model/                  # GAN model architecture
│   ├── __init__.py
│   ├── generator.py       # Generator model
│   └── discriminator.py   # Discriminator model
├── api/                   # FastAPI backend
│   ├── __init__.py
│   └── main.py           # API endpoints
├── ui/                    # Web interface
│   ├── index.html
│   ├── style.css
│   └── script.js
├── data/                  # Data directory
│   ├── dataset/          # Training images (organized in subdirectories)
│   └── generated/        # Generated samples during training
├── checkpoints/          # Model checkpoints
├── logs/                 # Training logs
├── train.py             # Training script
├── requirements.txt     # Python dependencies
├── Dockerfile          # Docker configuration
└── docker-compose.yml  # Docker Compose configuration
```

## Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU training)

### Installation

1. **Clone the repository**

```bash
git clone <repository-url>
cd pokemon_card_gen
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Prepare your dataset**

Organize your Pokemon card images in subdirectories:

```
data/dataset/
└── cards/
    ├── card1.jpg
    ├── card2.jpg
    └── ...
```

Note: Images will be automatically resized to 256x256 during training.

### Training the Model

Train the GAN model on your dataset:

```bash
python train.py --data_dir data/dataset --num_epochs 100 --batch_size 32
```

Training options:
- `--data_dir`: Directory containing training images (default: data/dataset)
- `--output_dir`: Directory to save generated images (default: data/generated)
- `--checkpoint_dir`: Directory to save checkpoints (default: checkpoints)
- `--latent_dim`: Dimension of latent noise vector (default: 100)
- `--img_size`: Size of images (default: 256)
- `--batch_size`: Batch size for training (default: 32)
- `--lr`: Learning rate (default: 0.0002)
- `--num_epochs`: Number of training epochs (default: 100)
- `--checkpoint`: Path to checkpoint to resume training

**Training Tips:**
- Start with 100-200 epochs for initial results
- Monitor the generated samples in `data/generated/`
- Use GPU for faster training (CUDA will be automatically detected)
- Larger batch sizes work better with more GPU memory

### Running the API Server

Start the FastAPI server:

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Or directly:

```bash
cd api
python main.py
```

The API will be available at:
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Web UI: http://localhost:8000/ui

### Using the Web Interface

1. Open your browser and navigate to http://localhost:8000/ui
2. Adjust the number of cards to generate (1-16)
3. Optionally set a seed for reproducible results
4. Click "Generate Cards"
5. View, enlarge, and download generated cards

## API Endpoints

### `GET /`
API information and available endpoints

### `GET /info`
Get model information including:
- Model loaded status
- Device (CPU/CUDA)
- Latent dimension
- Available checkpoints

### `POST /generate`
Generate multiple Pokemon cards

**Request Body:**
```json
{
    "num_images": 4,
    "seed": 42
}
```

**Response:**
```json
{
    "success": true,
    "num_images": 4,
    "images": ["base64_encoded_image1", "base64_encoded_image2", ...]
}
```

### `GET /generate/single`
Generate a single card and return as PNG image

**Query Parameters:**
- `seed` (optional): Seed for reproducibility

### `POST /reload`
Reload the model from the latest checkpoint

## Docker Deployment

### Using Docker

Build and run with Docker:

```bash
# Build the image
docker build -t pokemon-card-gen .

# Run the container
docker run -p 8000:8000 -v $(pwd)/data:/app/data -v $(pwd)/checkpoints:/app/checkpoints pokemon-card-gen
```

### Using Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Model Architecture

### Generator
- Input: 100-dimensional latent vector
- Output: 256x256 RGB image
- Architecture: 7-layer Transposed Convolutional Network
- Activation: ReLU (hidden layers), Tanh (output)

### Discriminator
- Input: 256x256 RGB image
- Output: Real/Fake probability
- Architecture: 7-layer Convolutional Network
- Activation: LeakyReLU (hidden layers), Sigmoid (output)

## Training Details

- **Loss Function**: Binary Cross Entropy (BCE)
- **Optimizer**: Adam (lr=0.0002, beta1=0.5)
- **Data Augmentation**: Random horizontal flip, color jitter
- **Checkpointing**: Saved every 10 epochs + final model
- **Visualization**: Generated samples saved every epoch

## Tips for Better Results

1. **Dataset Quality**
   - Use high-quality, consistent images
   - Minimum 500-1000 images recommended
   - More data = better results

2. **Training**
   - Train for at least 100 epochs
   - Monitor generated samples to check progress
   - If mode collapse occurs, reduce learning rate

3. **Generation**
   - Use seed for reproducible results
   - Generate multiple batches to find best results
   - Save your favorite seeds for later use

## Troubleshooting

### Model not loading in API
- Ensure you've trained a model first
- Check that checkpoints exist in `checkpoints/` directory
- Verify checkpoint files are not corrupted

### Training fails with "No images found"
- Check that images are in subdirectories under `data/dataset/`
- Example structure: `data/dataset/cards/*.jpg`

### Out of memory errors
- Reduce batch size: `--batch_size 16` or `--batch_size 8`
- Reduce image size: `--img_size 128`
- Use CPU training if GPU memory is insufficient

### Poor quality results
- Train for more epochs
- Increase dataset size
- Check data quality and consistency

## Requirements

See `requirements.txt` for full dependency list. Key dependencies:
- PyTorch >= 2.0.0
- FastAPI >= 0.104.0
- Pillow >= 10.0.0
- Uvicorn >= 0.24.0

## License

This project is open source and available for educational purposes.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Built with PyTorch and FastAPI
- DCGAN architecture based on Radford et al. (2015)
- Inspired by the Pokemon Trading Card Game

---

Built with ❤️ for Pokemon fans and AI enthusiasts
