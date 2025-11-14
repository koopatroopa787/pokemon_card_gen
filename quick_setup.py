#!/usr/bin/env python3
"""
Cross-platform Pokemon Card Generator Setup Script

Works on Windows, Linux, and macOS without needing bash or PowerShell
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def print_header(text, color='cyan'):
    """Print colored header"""
    colors = {
        'cyan': '\033[96m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'red': '\033[91m',
        'reset': '\033[0m'
    }

    if platform.system() == 'Windows':
        # Windows doesn't support ANSI colors in older terminals
        print("=" * 40)
        print(text)
        print("=" * 40)
    else:
        color_code = colors.get(color, colors['cyan'])
        reset = colors['reset']
        print(f"{color_code}{'=' * 40}{reset}")
        print(f"{color_code}{text}{reset}")
        print(f"{color_code}{'=' * 40}{reset}")


def print_step(text, status='info'):
    """Print step with status indicator"""
    indicators = {
        'info': '→',
        'success': '✓',
        'error': '✗',
        'warning': '⚠'
    }

    indicator = indicators.get(status, '→')
    print(f"{indicator} {text}")


def check_python_version():
    """Check Python version"""
    print_step("Checking Python version...", 'info')

    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_step(f"Python {version_str} detected", 'error')
        print_step("Python 3.8 or higher required", 'error')
        return False

    print_step(f"Python {version_str}", 'success')
    return True


def install_dependencies():
    """Install Python dependencies"""
    print("\n")
    print_step("Installing dependencies...", 'info')

    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print_step("Dependencies installed", 'success')
        return True
    except subprocess.CalledProcessError:
        print_step("Failed to install dependencies", 'error')
        return False


def check_kaggle_credentials():
    """Check if Kaggle credentials are set up"""
    print("\n")
    print_step("Checking Kaggle API credentials...", 'info')

    # Determine Kaggle directory based on OS
    if platform.system() == 'Windows':
        kaggle_dir = Path.home() / '.kaggle'
    else:
        kaggle_dir = Path.home() / '.kaggle'

    kaggle_json = kaggle_dir / 'kaggle.json'

    if not kaggle_json.exists():
        print("\n")
        print_header("KAGGLE API SETUP REQUIRED", 'red')
        print("\nTo download the Pokemon cards dataset, you need Kaggle API credentials.\n")
        print("Steps:")
        print("  1. Go to: https://www.kaggle.com/settings")
        print("  2. Scroll to 'API' section")
        print("  3. Click 'Create New Token'")
        print("  4. Download kaggle.json")
        print("\nThen move it to the right location:\n")

        if platform.system() == 'Windows':
            print(f"  1. Create directory: {kaggle_dir}")
            print(f"  2. Move kaggle.json to: {kaggle_json}")
            print("\nPowerShell commands:")
            print(f"  New-Item -ItemType Directory -Force -Path {kaggle_dir}")
            print(f"  Move-Item $env:USERPROFILE\\Downloads\\kaggle.json {kaggle_json}")
        else:
            print(f"  mkdir -p {kaggle_dir}")
            print(f"  mv ~/Downloads/kaggle.json {kaggle_json}")
            print(f"  chmod 600 {kaggle_json}")

        print("\nAfter setting up kaggle.json, run this script again.")
        print_header("", 'red')
        return False

    print_step("Kaggle credentials found", 'success')

    # Set correct permissions on Unix-like systems
    if platform.system() != 'Windows':
        try:
            os.chmod(kaggle_json, 0o600)
        except:
            pass

    return True


def download_dataset():
    """Download and prepare Pokemon cards dataset"""
    print("\n")
    print_step("Downloading Pokemon cards dataset from Kaggle...", 'info')

    try:
        result = subprocess.run(
            [sys.executable, "setup_kaggle_dataset.py"],
            check=True
        )
        return result.returncode == 0
    except subprocess.CalledProcessError:
        print_step("Dataset download failed", 'error')
        return False


def main():
    """Main setup function"""
    print("\n")
    print_header("Pokemon Card Generator - Quick Setup")

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Install dependencies
    if not install_dependencies():
        sys.exit(1)

    # Check Kaggle credentials
    if not check_kaggle_credentials():
        sys.exit(1)

    # Download dataset
    if not download_dataset():
        sys.exit(1)

    # Success!
    print("\n")
    print_header("Setup Complete!", 'green')
    print("\nThe Pokemon card dataset is ready for training.\n")
    print("Next steps:")
    print("  1. Train the model:")
    print("     python train_pokemon.py --data_dir data/pokemon_cards/training/train")
    print("\n  2. After training, start the API:")
    print("     python run_server.py")
    print("\n  3. Open the UI:")
    print("     http://localhost:8000/ui")
    print("\n")
    print_header("", 'green')


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
