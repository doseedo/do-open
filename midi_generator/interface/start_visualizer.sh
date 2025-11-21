#!/bin/bash

# MIDI Parameter Visualizer Startup Script
# This script starts both the backend API server and frontend web server

echo "========================================================================"
echo "🎵 MIDI Parameter Visualizer"
echo "========================================================================"
echo ""

# Check if we're in the right directory
if [ ! -f "server.py" ]; then
    echo "❌ Error: server.py not found. Please run this script from the interface/ directory."
    exit 1
fi

# Check Python dependencies
echo "Checking dependencies..."
python3 -c "import flask, flask_cors" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing dependencies. Installing Flask and Flask-CORS..."
    pip3 install flask flask-cors
fi

echo "✓ Dependencies OK"
echo ""

# Start backend server in background
echo "Starting backend API server on http://localhost:5001..."
python3 server.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 2

# Check if backend started successfully
if ! ps -p $BACKEND_PID > /dev/null; then
    echo "❌ Failed to start backend server"
    exit 1
fi

echo "✓ Backend server started (PID: $BACKEND_PID)"
echo ""

# Start frontend server
echo "Starting frontend web server on http://localhost:8000..."
echo ""
echo "========================================================================"
echo "✅ Visualizer is ready!"
echo "========================================================================"
echo ""
echo "  🌐 Open in browser: http://localhost:8000"
echo ""
echo "  - Drag and drop MIDI files to analyze"
echo "  - Adjust parameters in real-time"
echo "  - Export modified parameters"
echo ""
echo "Press Ctrl+C to stop both servers"
echo "========================================================================"
echo ""

# Trap Ctrl+C to kill both servers
trap "echo ''; echo 'Stopping servers...'; kill $BACKEND_PID 2>/dev/null; exit" INT TERM

# Start frontend server (this will block)
python3 -m http.server 8000

# If we get here, frontend stopped, so kill backend
kill $BACKEND_PID 2>/dev/null
