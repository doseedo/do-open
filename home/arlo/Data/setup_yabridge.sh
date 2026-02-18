#!/bin/bash
# Setup yabridge to use Windows VST plugins on Linux

set -e

echo "==================================="
echo "Yabridge Setup for Linux VST Hosting"
echo "==================================="

# Check for Wine
if ! command -v wine &> /dev/null; then
    echo "Installing Wine..."
    sudo dpkg --add-architecture i386
    sudo apt update
    sudo apt install -y wine64 wine32 wine winetricks
else
    echo "✓ Wine is already installed"
fi

# Download and install yabridge
YABRIDGE_VERSION="5.1.0"
YABRIDGE_URL="https://github.com/robbert-vdh/yabridge/releases/download/${YABRIDGE_VERSION}/yabridge-${YABRIDGE_VERSION}.tar.gz"

echo ""
echo "Downloading yabridge ${YABRIDGE_VERSION}..."
cd /tmp
wget -O yabridge.tar.gz "$YABRIDGE_URL"

echo "Extracting yabridge..."
mkdir -p ~/.local/share/yabridge
tar -xzf yabridge.tar.gz -C ~/.local/share/yabridge --strip-components=1

# Add to PATH if not already there
if ! grep -q "yabridge" ~/.bashrc; then
    echo 'export PATH="$HOME/.local/share/yabridge:$PATH"' >> ~/.bashrc
    export PATH="$HOME/.local/share/yabridge:$PATH"
fi

echo "✓ Yabridge installed to ~/.local/share/yabridge"

# Setup VST directories
mkdir -p ~/.vst3/yabridge
mkdir -p ~/vst-plugins-windows

echo ""
echo "==================================="
echo "Setup complete!"
echo "==================================="
echo ""
echo "Next steps:"
echo "1. Place your Windows VST plugins in: ~/vst-plugins-windows/"
echo "2. Run: yabridgectl add ~/vst-plugins-windows"
echo "3. Run: yabridgectl sync"
echo "4. Your VST plugins will be available in: ~/.vst3/yabridge/"
echo ""
echo "Note: You may need to restart your terminal or run:"
echo "  source ~/.bashrc"
