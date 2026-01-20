#!/bin/bash

# MSME Project - Start All Services
# This script starts all three services in separate terminal tabs/windows

echo "🚀 Starting MSME Business Intelligence Platform..."
echo ""

# Get the project root directory
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if we're on macOS (for Terminal.app support)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "📱 Detected macOS - Opening services in separate Terminal tabs..."
    
    # Open new Terminal tabs for each service
    osascript <<EOF
tell application "Terminal"
    activate
    
    -- Backend Server
    tell application "System Events" to keystroke "t" using {command down}
    delay 0.5
    do script "cd '$PROJECT_DIR/server' && echo '🔧 Starting Backend Server...' && npm start" in front window
    
    -- AI Service
    tell application "System Events" to keystroke "t" using {command down}
    delay 0.5
    do script "cd '$PROJECT_DIR/ai_service' && echo '🤖 Starting AI Service...' && source venv/bin/activate && python main.py" in front window
    
    -- Client Dashboard
    tell application "System Events" to keystroke "t" using {command down}
    delay 0.5
    do script "cd '$PROJECT_DIR/client' && echo '🎨 Starting Client Dashboard...' && npm run dev" in front window
end tell
EOF

    echo ""
    echo "✅ All services are starting in separate Terminal tabs!"
    echo ""
    echo "📊 Services:"
    echo "   • Backend Server:    http://localhost:5000"
    echo "   • AI Service:        http://localhost:8000"
    echo "   • Client Dashboard:  http://localhost:5173"
    echo ""
    echo "🌐 Open your browser to: http://localhost:5173"
    echo ""
    echo "⚠️  To stop all services, press Ctrl+C in each terminal tab"
    
else
    echo "⚠️  Non-macOS system detected. Starting services in background..."
    echo "   (You may prefer to run each service manually in separate terminals)"
    echo ""
    
    # Start services in background
    cd "$PROJECT_DIR/server" && npm start &
    SERVER_PID=$!
    
    cd "$PROJECT_DIR/ai_service" && source venv/bin/activate && python main.py &
    AI_PID=$!
    
    cd "$PROJECT_DIR/client" && npm run dev &
    CLIENT_PID=$!
    
    echo "✅ Services started!"
    echo "   Backend PID: $SERVER_PID"
    echo "   AI Service PID: $AI_PID"
    echo "   Client PID: $CLIENT_PID"
    echo ""
    echo "🌐 Open your browser to: http://localhost:5173"
    echo ""
    echo "⚠️  To stop all services, run: kill $SERVER_PID $AI_PID $CLIENT_PID"
fi
