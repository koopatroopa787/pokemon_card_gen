# Kaggle Dataset Setup Guide

This guide shows you how to automatically download and use the Pokemon cards dataset from Kaggle.

## 🚀 One-Command Setup

**Windows (PowerShell):**
```powershell
.\quick_setup.ps1
```

**Linux/macOS:**
```bash
bash quick_setup.sh
```

**Cross-platform (Python):**
```bash
python quick_setup.py
```

That's it! This will:
1. Install all dependencies
2. Download 13,000+ Pokemon cards from Kaggle
3. Prepare images for training
4. Create a quick-start training script

## 📋 Prerequisites

### 1. Kaggle Account

You need a free Kaggle account. [Sign up here](https://www.kaggle.com/account/login).

### 2. Kaggle API Credentials

Get your API credentials:

1. Go to [Kaggle Settings](https://www.kaggle.com/settings)
2. Scroll to the **API** section
3. Click **"Create New Token"**
4. Download the `kaggle.json` file

### 3. Install Credentials

Put the credentials in the right location:

#### Linux/Mac:
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

#### Windows:
```cmd
mkdir %USERPROFILE%\.kaggle
move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\kaggle.json
```

## 🎯 Usage

### Method 1: Automated Setup (Recommended)

```bash
# Run the complete setup
bash quick_setup.sh
```

This handles everything automatically!

### Method 2: Manual Steps

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download and prepare dataset
python setup_kaggle_dataset.py

# 3. Train the model
python train_pokemon.py --data_dir data/pokemon_cards/training/train
```

### Method 3: Custom Options

```bash
# Download to custom location
python setup_kaggle_dataset.py --output-dir my_custom_path

# Download without auto-preparation
python setup_kaggle_dataset.py --no-prepare

# Use different Kaggle dataset
python setup_kaggle_dataset.py --dataset username/dataset-name
```

## 📦 What Gets Downloaded

The Kaggle dataset includes:
- **13,000+ Pokemon card images** (or image URLs)
- **Metadata CSV** with card info (HP, name, set, etc.)
- **High-resolution images** ready for training

## 📁 Directory Structure After Setup

```
data/pokemon_cards/
├── raw/                        # Raw downloaded data
│   ├── pokemon_cards.csv      # Metadata
│   └── images/                # Card images (if included)
├── images/                     # Processed images
│   ├── card1.jpg
│   └── ...
├── training/                   # Training-ready data
│   └── train/                 # 256x256 resized images
│       ├── card1.jpg
│       └── ...
├── metadata/                   # Metadata files
│   ├── cards.json             # JSON format
│   └── cards.csv              # CSV format
└── quick_train.sh             # Quick start training script
```

## 🎮 Training After Setup

Once setup is complete, you have several options:

### Option 1: Quick Start Script

```bash
bash data/pokemon_cards/quick_train.sh
```

### Option 2: Full Control

```bash
python train_pokemon.py \
  --data_dir data/pokemon_cards/training/train \
  --metadata data/pokemon_cards/metadata/cards.json \
  --num_epochs 100 \
  --batch_size 32
```

### Option 3: Use Configuration

```bash
cp config.default.yaml config.yaml
# Edit config.yaml with your settings
python train_pokemon.py
```

## 🔧 Troubleshooting

### Issue: "kaggle.json not found"

**Solution:**
```bash
# Make sure kaggle.json is in the right place
ls ~/.kaggle/kaggle.json

# If not, download it from Kaggle and move it
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Issue: "Permission denied"

**Solution:**
```bash
# Fix permissions
chmod 600 ~/.kaggle/kaggle.json
```

### Issue: "Dataset not found"

**Solution:**
The dataset might have been moved or renamed. Check the current dataset URL:
- Visit: https://www.kaggle.com/datasets/priyamchoksi/pokemon-cards
- Use the exact dataset name in the command

### Issue: "Download fails"

**Solution:**
```bash
# Check your internet connection
# Make sure you're authenticated with Kaggle
kaggle datasets list  # This should work if setup is correct

# Try downloading manually
kaggle datasets download -d priyamchoksi/pokemon-cards
```

### Issue: "Not enough disk space"

**Solution:**
The dataset requires ~2-5GB. Free up space or use a custom location:
```bash
python setup_kaggle_dataset.py --output-dir /path/with/more/space
```

## 📊 Dataset Information

After download, you can check dataset statistics:

```bash
# Get dataset info
python -m utils.data_utils stats --input data/pokemon_cards/training/train

# Validate images
python -m utils.data_utils validate --input data/pokemon_cards/training/train
```

## 🎯 Quick Reference

| Command | Description |
|---------|-------------|
| `bash quick_setup.sh` | Complete automated setup |
| `python setup_kaggle_dataset.py` | Download & prepare dataset |
| `python train_pokemon.py --data_dir data/pokemon_cards/training/train` | Start training |
| `python run_server.py` | Start API server |
| `python generate.py --checkpoint checkpoints/checkpoint_epoch_final.pth` | Generate cards |

## 🌐 Alternative: Manual Download

If you prefer to download manually:

1. Go to: https://www.kaggle.com/datasets/priyamchoksi/pokemon-cards
2. Click **"Download"**
3. Extract to `data/pokemon_cards/raw/`
4. Run preparation:
   ```bash
   python setup_kaggle_dataset.py --no-prepare
   ```

## 💡 Tips

1. **First Time**: Run with `--limit 100` to test with a small subset
2. **Slow Download**: The download might take 5-15 minutes depending on your connection
3. **Resume**: If download fails, just run the script again - it will resume
4. **Storage**: Make sure you have at least 5GB free space

## 📝 Next Steps

After successful setup:

1. **Verify dataset**: Check that images are in `data/pokemon_cards/training/train/`
2. **Start training**: Use one of the training methods above
3. **Monitor progress**: Check `data/generated/` for sample outputs
4. **Generate cards**: After training, use the API or CLI to generate new cards

## 🆘 Need Help?

- Check the main [README.md](README.md) for general setup
- See [DATASET_GUIDE.md](DATASET_GUIDE.md) for dataset details
- Open an issue on GitHub for support

---

**Ready to create amazing Pokemon cards? Let's go! 🎴✨**
