#!/bin/bash
# TensorBoard Launcher for Domain Encoder Training
#
# Usage:
#   ./launch_tensorboard.sh [log_dir] [port]
#
# Examples:
#   ./launch_tensorboard.sh                                    # Use default log dir and port
#   ./launch_tensorboard.sh output/semantic_discovery/logs     # Custom log dir
#   ./launch_tensorboard.sh output/logs 6007                   # Custom log dir and port

# Default configuration
DEFAULT_LOG_DIR="output/semantic_discovery/logs"
DEFAULT_PORT=6006

# Parse arguments
LOG_DIR="${1:-$DEFAULT_LOG_DIR}"
PORT="${2:-$DEFAULT_PORT}"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║         Domain Encoder Training - TensorBoard Launcher           ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Log Directory: $LOG_DIR"
echo "🌐 Port: $PORT"
echo ""

# Check if log directory exists
if [ ! -d "$LOG_DIR" ]; then
    echo "⚠️  Warning: Log directory does not exist: $LOG_DIR"
    echo "   Creating directory..."
    mkdir -p "$LOG_DIR"
    echo "   ✅ Directory created"
    echo ""
    echo "   Note: Start training to generate logs, then refresh TensorBoard"
    echo ""
fi

# Check if tensorboard is installed
if ! command -v tensorboard &> /dev/null; then
    echo "❌ Error: TensorBoard is not installed"
    echo ""
    echo "To install TensorBoard:"
    echo "  pip install tensorboard"
    echo ""
    exit 1
fi

# Check if port is already in use
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  Warning: Port $PORT is already in use"
    echo ""
    echo "Options:"
    echo "  1. Use a different port: ./launch_tensorboard.sh $LOG_DIR 6007"
    echo "  2. Kill existing process: kill \$(lsof -t -i:$PORT)"
    echo ""
    read -p "Kill existing process and continue? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        kill $(lsof -t -i:$PORT)
        echo "✅ Process killed"
        sleep 1
    else
        echo "Exiting..."
        exit 1
    fi
fi

echo "🚀 Starting TensorBoard..."
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  🌐 TensorBoard URL: http://localhost:$PORT"
echo ""
echo "  📈 Available Metrics:"
echo "     • Loss/train              - Training loss (target: < 10)"
echo "     • Loss/val                - Validation loss"
echo "     • Loss/reconstruction     - Feature reconstruction loss"
echo "     • Loss/sparsity           - Feature sparsity loss"
echo "     • Loss/locality           - Locality constraint loss"
echo "     • Loss/orthogonality      - Feature orthogonality loss"
echo "     • Learning_Rate           - Current learning rate"
echo "     • Feature_Sparsity        - Sparsity of learned features"
echo ""
echo "  💡 Tips:"
echo "     • Press Ctrl+C to stop TensorBoard"
echo "     • Refresh browser to see new data"
echo "     • Use 'Smoothing' slider to reduce noise"
echo "     • Compare multiple runs in the same view"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Launch TensorBoard
tensorboard --logdir="$LOG_DIR" --port=$PORT --bind_all
