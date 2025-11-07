#!/bin/bash

# Enhanced Training Dashboard Launcher
# This script starts the enhanced real-time training dashboard
# with all new features including validation tracking, checkpoint management,
# event timeline, and GPU monitoring

echo "=========================================="
echo "Enhanced Training Dashboard for GLM-4.5-Air"
echo "=========================================="
echo ""
echo "Starting dashboard with enhanced features:"
echo "  ✓ Validation loss & perplexity tracking"
echo "  ✓ Best checkpoint identification"
echo "  ✓ Gradient norm monitoring"
echo "  ✓ Checkpoint management panel"
echo "  ✓ Event timeline with markers"
echo "  ✓ GPU resource monitoring"
echo "  ✓ Auto-refresh every 30 seconds"
echo ""
echo "Dashboard will be accessible at:"
echo "  http://localhost:5000"
echo "  http://$(hostname -I | awk '{print $1}'):5000"
echo ""
echo "Press Ctrl+C to stop the dashboard"
echo "=========================================="
echo ""

# Check if Flask is installed
if ! python3 -c "import flask" 2>/dev/null; then
    echo "⚠️  Flask not found. Installing..."
    pip install flask matplotlib numpy
fi

# Check if gunicorn is available (recommended for production)
if command -v gunicorn &> /dev/null; then
    echo "Using gunicorn (production mode)..."
    gunicorn -w 2 -b 0.0.0.0:5000 --timeout 120 training_dashboard:app
else
    echo "Using Flask development server..."
    echo "For better performance, install gunicorn: pip install gunicorn"
    python3 training_dashboard_enhanced.py
fi
