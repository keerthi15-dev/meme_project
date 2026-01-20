#!/bin/bash

# MSME Project - Stop All Services
# This script stops all running MSME services

echo "🛑 Stopping MSME services..."

# Kill processes by port
kill_port() {
    local port=$1
    local pid=$(lsof -ti:$port)
    if [ ! -z "$pid" ]; then
        kill -9 $pid 2>/dev/null
        echo "   ✓ Stopped service on port $port"
    else
        echo "   • No service found on port $port"
    fi
}

# Stop all services
kill_port 5000  # Backend
kill_port 8000  # AI Service
kill_port 5173  # Client

echo ""
echo "✅ All services stopped!"
