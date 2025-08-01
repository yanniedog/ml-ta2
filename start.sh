#!/bin/bash

clear

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                    ML-TA - Machine Learning Technical Analysis               ║"
echo "║                              Starting Application                            ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if Python is installed
echo "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo ""
    echo "ERROR: Python 3 is not installed or not in PATH."
    echo "Please install Python 3.10 or later."
    echo ""
    exit 1
fi

echo "✓ Python is installed"
echo ""

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 -c 'import sys; print("{0}.{1}".format(sys.version_info.major, sys.version_info.minor))')
if (( $(echo "$PYTHON_VERSION < 3.10" | bc -l) )); then
    echo ""
    echo "ERROR: Python 3.10 or later is required. Found version $PYTHON_VERSION"
    echo "Please install Python 3.10 or later."
    echo ""
    exit 1
fi

echo "✓ Python version $PYTHON_VERSION is compatible"
echo ""

# Check if virtual environment exists
echo "Checking virtual environment..."
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo ""
        echo "ERROR: Failed to create virtual environment."
        echo ""
        exit 1
    fi
    echo "✓ Virtual environment created"
else
    echo "✓ Using existing virtual environment"
fi

echo ""
# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Failed to activate virtual environment."
    echo ""
    exit 1
fi

echo "✓ Virtual environment activated"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip > /dev/null 2>&1
if [ -f "requirements.txt" ]; then
    echo "Installing from requirements.txt..."
    pip install -r requirements.txt > /dev/null 2>&1
fi

if [ -f "requirements-dev.txt" ]; then
    echo "Installing from requirements-dev.txt..."
    pip install -r requirements-dev.txt > /dev/null 2>&1
fi

echo "✓ Dependencies installed"
echo ""

# Run the application
echo "Starting ML-TA application..."
echo "The web interface will be available at: http://localhost:5000"
echo ""
echo "NOTE: To stop the application, press Ctrl+C"
echo ""
sleep 3
python -m src.web.app

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Failed to start ML-TA application."
    echo "Please check the error messages above."
    echo ""
    echo "You can try running the following commands manually:"
    echo "  source venv/bin/activate"
    echo "  python -m src.web.app"
    echo ""
    exit 1
fi
