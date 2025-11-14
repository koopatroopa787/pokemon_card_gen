# Windows Setup Guide

This guide is specifically for Windows users setting up the Pokemon Card Generator.

## 🪟 Quick Start for Windows

### Option 1: PowerShell Script (Recommended)

```powershell
.\quick_setup.ps1
```

### Option 2: Python Script (Also Works)

```cmd
python quick_setup.py
```

### Option 3: Manual Setup

```cmd
# Install dependencies
pip install -r requirements.txt

# Download dataset
python setup_kaggle_dataset.py

# Train
python train_pokemon.py --data_dir data\pokemon_cards\training\train
```

## 📋 Prerequisites

### 1. Python Installation

Download and install Python 3.8 or higher:
- Go to: https://www.python.org/downloads/
- Download the latest Python 3.x installer
- **Important**: Check "Add Python to PATH" during installation

Verify installation:
```cmd
python --version
```

### 2. Kaggle Account & Credentials

1. Create a free account at [Kaggle.com](https://www.kaggle.com)
2. Go to [Kaggle Settings](https://www.kaggle.com/settings)
3. Scroll to "API" section
4. Click "Create New Token"
5. Download `kaggle.json`

### 3. Install Kaggle Credentials

**Using PowerShell:**
```powershell
# Create Kaggle directory
New-Item -ItemType Directory -Force -Path $env:USERPROFILE\.kaggle

# Move kaggle.json
Move-Item $env:USERPROFILE\Downloads\kaggle.json $env:USERPROFILE\.kaggle\
```

**Using Command Prompt:**
```cmd
# Create Kaggle directory
mkdir %USERPROFILE%\.kaggle

# Move kaggle.json
move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\
```

## 🎯 Step-by-Step Setup

### Step 1: Clone/Download Repository

Download from GitHub or clone:
```cmd
git clone <your-repo-url>
cd pokemon_card_gen
```

### Step 2: Run Setup Script

**PowerShell (Run as Administrator if needed):**
```powershell
.\quick_setup.ps1
```

**Or Python:**
```cmd
python quick_setup.py
```

This will automatically:
- Install all Python packages
- Download 13,000+ Pokemon cards from Kaggle
- Prepare images for training
- Set up the project

### Step 3: Train the Model

After setup completes:
```cmd
python train_pokemon.py --data_dir data\pokemon_cards\training\train --num_epochs 100
```

### Step 4: Run the API Server

```cmd
python run_server.py
```

Then open your browser to: http://localhost:8000/ui

## 🔧 Common Windows Issues

### Issue: "Python not found"

**Solution:**
- Reinstall Python and check "Add to PATH"
- Or add Python manually to PATH:
  1. Search "Environment Variables" in Windows
  2. Edit "Path" variable
  3. Add Python installation directory

### Issue: "pip not found"

**Solution:**
```cmd
python -m pip install --upgrade pip
```

### Issue: "PowerShell execution policy"

**Solution:**
```powershell
# Run PowerShell as Administrator
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue: "Module not found" errors

**Solution:**
```cmd
# Reinstall requirements
pip install -r requirements.txt --upgrade
```

### Issue: "kaggle.json not found"

**Solution:**
Make sure kaggle.json is in the right location:
```cmd
dir %USERPROFILE%\.kaggle\kaggle.json
```

Should show the file. If not, download it again from Kaggle.

### Issue: "SSL Certificate" errors

**Solution:**
```cmd
pip install --upgrade certifi
```

### Issue: "Permission denied" for ports

**Solution:**
- Use a different port:
  ```cmd
  python run_server.py --port 8080
  ```
- Or run Command Prompt/PowerShell as Administrator

## 💡 Windows-Specific Tips

### Use Windows Terminal (Recommended)

Windows Terminal provides a better experience:
- Download from Microsoft Store
- Supports multiple tabs
- Better color support
- Modern interface

### Check Disk Space

The dataset requires ~5GB:
```powershell
Get-PSDrive C | Select-Object Free
```

### GPU Support (NVIDIA)

If you have an NVIDIA GPU:

1. Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
2. Install cuDNN: https://developer.nvidia.com/cudnn
3. PyTorch will automatically use GPU

Verify GPU:
```cmd
python -c "import torch; print(torch.cuda.is_available())"
```

### Antivirus Interference

Some antivirus software may slow down:
- Add project folder to exclusions
- Temporarily disable during download

## 📁 Windows File Paths

In Windows, use backslashes `\` or forward slashes `/`:

```cmd
# Both work
python train_pokemon.py --data_dir data\pokemon_cards\training\train
python train_pokemon.py --data_dir data/pokemon_cards/training/train
```

## 🎮 Training on Windows

### Basic Training

```cmd
python train_pokemon.py --data_dir data\pokemon_cards\training\train
```

### With GPU (if available)

Training will automatically use GPU if available. Check with:
```cmd
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not available')"
```

### Resume Training

```cmd
python train_pokemon.py --checkpoint checkpoints\checkpoint_epoch_50.pth
```

## 🌐 Using the API

### Start Server

```cmd
python run_server.py
```

### Access UI

Open browser to: http://localhost:8000/ui

### API Documentation

Open browser to: http://localhost:8000/docs

## 🐳 Docker on Windows

If you prefer Docker:

1. Install Docker Desktop for Windows
2. Enable WSL2 backend (recommended)
3. Run:

```cmd
docker-compose up -d
```

## 🆘 Getting Help

### Check Python Environment

```cmd
python --version
pip --version
pip list
```

### Check Kaggle Setup

```cmd
python -c "from kaggle.api.kaggle_api_extended import KaggleApi; api = KaggleApi(); api.authenticate(); print('Kaggle API configured!')"
```

### Verbose Error Messages

```cmd
python setup_kaggle_dataset.py -v
```

## ✅ Success Checklist

- [ ] Python 3.8+ installed
- [ ] pip working
- [ ] Kaggle account created
- [ ] kaggle.json downloaded and placed in `%USERPROFILE%\.kaggle\`
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Dataset downloaded (13,000+ images)
- [ ] Training started successfully

## 📚 Next Steps

After successful setup:

1. **Explore the dataset:**
   ```cmd
   python -m utils.data_utils stats --input data\pokemon_cards\training\train
   ```

2. **Start training:**
   ```cmd
   python train_pokemon.py --data_dir data\pokemon_cards\training\train
   ```

3. **Monitor progress:**
   - Check `data\generated\` for sample outputs
   - View loss plot in `logs\training_losses.png`

4. **Generate cards:**
   ```cmd
   python generate.py --checkpoint checkpoints\checkpoint_epoch_final.pth --num-images 10
   ```

5. **Use the web UI:**
   ```cmd
   python run_server.py
   ```

## 🎓 Additional Resources

- [Main README](README.md) - General documentation
- [Kaggle Setup Guide](KAGGLE_SETUP.md) - Detailed Kaggle instructions
- [Dataset Guide](DATASET_GUIDE.md) - Dataset information
- [API Documentation](http://localhost:8000/docs) - After starting server

---

**Happy generating on Windows! 🪟🎴**
