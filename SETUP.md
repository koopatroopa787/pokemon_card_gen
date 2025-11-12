# Setup Guide

This guide will help you set up and run the Pokemon Card Generator from scratch.

## Step 1: Environment Setup

### Option A: Using Python Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option B: Using Conda

```bash
# Create conda environment
conda create -n pokemon-card-gen python=3.10

# Activate environment
conda activate pokemon-card-gen

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Prepare Dataset

1. **Create the dataset directory structure:**

```bash
mkdir -p data/dataset/cards
```

2. **Add your Pokemon card images:**
   - Place all your Pokemon card images in `data/dataset/cards/`
   - Supported formats: JPG, PNG, JPEG
   - Recommended: At least 500-1000 images for good results
   - Images will be automatically resized to 256x256

Example structure:
```
data/dataset/
└── cards/
    ├── pikachu_001.jpg
    ├── charizard_006.jpg
    ├── mewtwo_150.jpg
    └── ...
```

**Where to get Pokemon card images:**
- Download from Pokemon card databases
- Use web scraping tools (respect copyright)
- Use your own scanned cards
- Use publicly available datasets

## Step 3: Train the Model

### Basic Training

```bash
python train.py --num_epochs 100 --batch_size 32
```

### Advanced Training Options

```bash
# Training with custom settings
python train.py \
    --data_dir data/dataset \
    --output_dir data/generated \
    --checkpoint_dir checkpoints \
    --num_epochs 200 \
    --batch_size 16 \
    --lr 0.0002 \
    --img_size 256

# Resume training from checkpoint
python train.py --checkpoint checkpoints/checkpoint_epoch_50.pth
```

### Training Tips

- **Start small**: Begin with 50-100 epochs to test
- **Monitor progress**: Check `data/generated/` for sample outputs
- **GPU recommended**: Training is much faster with CUDA
- **Batch size**: Reduce if you get out-of-memory errors
- **Time estimate**:
  - CPU: ~10-30 minutes per epoch (depending on dataset size)
  - GPU: ~1-5 minutes per epoch

## Step 4: Run the API Server

### Option A: Using Python

```bash
# Method 1: Using the convenience script
python run_server.py

# Method 2: Using uvicorn directly
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Method 3: From the API directory
cd api
python main.py
```

### Option B: Using Docker

```bash
# Build and run with docker-compose
docker-compose up -d

# Or build manually
docker build -t pokemon-card-gen .
docker run -p 8000:8000 -v $(pwd)/checkpoints:/app/checkpoints pokemon-card-gen
```

## Step 5: Use the Application

### Web Interface

1. Open your browser
2. Navigate to: http://localhost:8000/ui
3. Select number of cards to generate (1-16)
4. Optional: Set a seed for reproducibility
5. Click "Generate Cards"
6. View, enlarge, and download your cards!

### API Usage

#### Using cURL

```bash
# Generate 4 cards
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"num_images": 4}'

# Generate with specific seed
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"num_images": 1, "seed": 42}'

# Get a single image
curl "http://localhost:8000/generate/single?seed=123" -o card.png
```

#### Using Python

```python
import requests
import base64
from PIL import Image
import io

# Generate cards
response = requests.post(
    "http://localhost:8000/generate",
    json={"num_images": 4, "seed": 42}
)

data = response.json()

# Save the first image
img_data = base64.b64decode(data['images'][0])
img = Image.open(io.BytesIO(img_data))
img.save('generated_card.png')
```

#### Using JavaScript

```javascript
// Generate cards
fetch('http://localhost:8000/generate', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        num_images: 4,
        seed: 42
    })
})
.then(response => response.json())
.then(data => {
    // data.images contains base64 encoded images
    console.log(`Generated ${data.num_images} cards`);
});
```

## Troubleshooting

### "No images found" error during training

**Problem**: Dataset not organized correctly

**Solution**:
```bash
# Check your directory structure
ls data/dataset/

# Should show subdirectories (e.g., 'cards')
# Images should be inside subdirectories

# Correct structure:
data/dataset/cards/*.jpg  ✓
# Wrong structure:
data/dataset/*.jpg  ✗
```

### "Model not loaded" in API

**Problem**: No trained model available

**Solution**:
1. Train a model first: `python train.py`
2. Verify checkpoint exists: `ls checkpoints/`
3. Restart API server: `python run_server.py`

### Out of memory errors

**Problem**: Not enough GPU/RAM memory

**Solution**:
```bash
# Reduce batch size
python train.py --batch_size 8

# Reduce image size
python train.py --img_size 128

# Use CPU instead of GPU (slower)
export CUDA_VISIBLE_DEVICES=""
python train.py
```

### Poor quality generated images

**Problem**: Insufficient training or data

**Solution**:
1. Train for more epochs (200-500)
2. Add more training images (1000+ recommended)
3. Ensure training images are high quality
4. Check that loss values are decreasing

### Port already in use

**Problem**: Port 8000 is already occupied

**Solution**:
```bash
# Use a different port
python -m uvicorn api.main:app --host 0.0.0.0 --port 8080

# Or kill the process using port 8000
# Linux/Mac:
lsof -ti:8000 | xargs kill -9
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

## Quick Reference

### Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Train model (basic)
python train.py

# Train model (custom)
python train.py --num_epochs 200 --batch_size 16

# Run server
python run_server.py

# Run with Docker
docker-compose up -d

# View logs
docker-compose logs -f

# Stop Docker services
docker-compose down
```

### File Locations

- **Training data**: `data/dataset/`
- **Generated samples**: `data/generated/`
- **Model checkpoints**: `checkpoints/`
- **Training logs**: `logs/`
- **Web UI**: `ui/`
- **API code**: `api/`
- **Model code**: `model/`

## Next Steps

1. **Experiment with parameters**: Try different training settings
2. **Collect more data**: More images = better results
3. **Fine-tune the model**: Resume training with different hyperparameters
4. **Integrate into projects**: Use the API in your own applications
5. **Share your results**: Show off your generated cards!

## Need Help?

- Check the main README.md for detailed documentation
- Review the API docs at http://localhost:8000/docs
- Ensure all dependencies are installed correctly
- Verify Python version is 3.8 or higher

Happy generating! 🎴
