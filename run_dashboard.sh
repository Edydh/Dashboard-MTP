#!/bin/bash

# Mileage Tracker Pro Dashboard Runner Script

echo "🚗 Starting Mileage Tracker Pro Dashboard..."
echo "=========================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are installed
if ! pip show streamlit &> /dev/null; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found!"
    echo "   Please run: python3 setup.py"
    echo "   Or copy env.example to .env and add your service role credentials"
    exit 1
fi

# Run the dashboard
echo "🚀 Launching dashboard..."
echo "   Opening in browser: http://localhost:8501"
echo "   Binding to localhost only by default"
echo ""
echo "   Press Ctrl+C to stop the dashboard"
echo "=========================================="

streamlit run dashboard.py --server.address localhost
