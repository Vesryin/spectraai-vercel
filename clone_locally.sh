#!/bin/bash

# Spectra AI - Local Clone Setup Script
# This script helps you clone the Spectra AI repository to your local machine

echo "🌟 Spectra AI - Local Clone Setup 🌟"
echo "======================================"
echo ""

# Repository URL
REPO_URL="https://github.com/Vesryin/Spectra-AI-Library-Version.git"
DEFAULT_DIR="Spectra-AI-Library-Version"

echo "Repository: $REPO_URL"
echo ""

# Ask for clone directory
read -p "Enter directory name for local clone (default: $DEFAULT_DIR): " CLONE_DIR
CLONE_DIR=${CLONE_DIR:-$DEFAULT_DIR}

echo ""
echo "Cloning repository to: $CLONE_DIR"
echo "Running: git clone $REPO_URL $CLONE_DIR"
echo ""

# Clone the repository
git clone "$REPO_URL" "$CLONE_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Successfully cloned Spectra AI repository!"
    echo ""
    echo "Next steps:"
    echo "1. cd $CLONE_DIR"
    echo "2. Create a .env file with your API keys"
    echo "3. Install dependencies: pip install -r requirements.txt"
    echo "4. Run the application: python app.py"
    echo ""
    echo "🎵 Happy coding with Spectra AI! 🎵"
else
    echo ""
    echo "❌ Failed to clone repository. Please check your internet connection and try again."
fi
