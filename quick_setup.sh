#!/bin/bash
# One-command setup for Pokemon Card Generator with Kaggle dataset

set -e  # Exit on error

echo "======================================"
echo "Pokemon Card Generator - Quick Setup"
echo "======================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python $python_version"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Setup Kaggle credentials if not exists
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo ""
    echo "======================================"
    echo "KAGGLE API SETUP REQUIRED"
    echo "======================================"
    echo ""
    echo "To download the Pokemon cards dataset, you need Kaggle API credentials."
    echo ""
    echo "Steps:"
    echo "  1. Go to: https://www.kaggle.com/settings"
    echo "  2. Scroll to 'API' section"
    echo "  3. Click 'Create New Token'"
    echo "  4. Download kaggle.json"
    echo ""
    echo "Then run:"
    echo "  mkdir -p ~/.kaggle"
    echo "  mv ~/Downloads/kaggle.json ~/.kaggle/"
    echo "  chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    echo "After setting up kaggle.json, run this script again."
    echo "======================================"
    exit 1
fi

# Download and prepare dataset
echo ""
echo "Downloading Pokemon cards dataset from Kaggle..."
python setup_kaggle_dataset.py

# Done
echo ""
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "The Pokemon card dataset is ready for training."
echo ""
echo "Next steps:"
echo "  1. Train the model:"
echo "     python train_pokemon.py --data_dir data/pokemon_cards/training/train"
echo ""
echo "  2. Or use the quick train script:"
echo "     bash data/pokemon_cards/quick_train.sh"
echo ""
echo "  3. After training, start the API:"
echo "     python run_server.py"
echo ""
echo "  4. Open the UI:"
echo "     http://localhost:8000/ui"
echo ""
echo "======================================"
