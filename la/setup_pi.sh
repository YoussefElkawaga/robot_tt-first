#!/bin/bash

echo "🍓 Setting up Conversation Robot for Raspberry Pi"
echo "================================================"

# Check if running on Raspberry Pi
if grep -q "BCM\|ARM" /proc/cpuinfo; then
    echo "✅ Raspberry Pi detected"
    IS_PI=true
else
    echo "💻 Not running on Raspberry Pi, but continuing setup..."
    IS_PI=false
fi

# Update system packages
echo "📦 Updating system packages..."
sudo apt update
sudo apt upgrade -y

# Install system dependencies
echo "🔧 Installing system dependencies..."
sudo apt install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    portaudio19-dev \
    espeak \
    espeak-data \
    libespeak1 \
    libespeak-dev \
    festival \
    festvox-kallpc16k \
    alsa-utils \
    pulseaudio \
    pulseaudio-utils \
    sox \
    libsox-fmt-all \
    ffmpeg \
    cmake \
    build-essential \
    pkg-config

# Install camera dependencies for Pi
if [ "$IS_PI" = true ]; then
    echo "📷 Installing Raspberry Pi camera dependencies..."
    sudo apt install -y \
        python3-opencv \
        libopencv-dev \
        python3-picamera \
        v4l-utils
    
    # Enable camera interface
    echo "🎥 Enabling camera interface..."
    sudo raspi-config nonint do_camera 0
    
    # Add user to video group
    sudo usermod -a -G video $USER
fi

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv robot_env
source robot_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies with Pi optimizations
echo "📚 Installing Python packages..."

if [ "$IS_PI" = true ]; then
    echo "Installing packages optimized for Raspberry Pi..."
    
    # Install numpy first (pre-compiled for Pi)
    pip install numpy
    
    # Install OpenCV for Pi (lighter version)
    pip install opencv-python-headless
    
    # Install other packages with no cache to save space
    pip install --no-cache-dir \
        pyaudio \
        SpeechRecognition \
        pyttsx3 \
        python-dotenv \
        requests \
        pvporcupine \
        google-generativeai \
        fer \
        elevenlabs
else
    echo "Installing standard packages..."
    pip install \
        numpy \
        opencv-python \
        pyaudio \
        SpeechRecognition \
        pyttsx3 \
        python-dotenv \
        requests \
        pvporcupine \
        google-generativeai \
        fer \
        elevenlabs
fi

# Test audio system
echo "🔊 Testing audio system..."
if command -v speaker-test &> /dev/null; then
    echo "Testing speakers (you should hear a tone)..."
    timeout 3s speaker-test -t sine -f 1000 -l 1 -s 1 || echo "Speaker test completed"
fi

# Test microphone
echo "🎤 Testing microphone..."
if command -v arecord &> /dev/null; then
    echo "Recording 2 seconds of audio to test microphone..."
    timeout 2s arecord -f cd -t wav /tmp/test_mic.wav 2>/dev/null || echo "Microphone test completed"
    rm -f /tmp/test_mic.wav
fi

# Test camera
if [ "$IS_PI" = true ]; then
    echo "📷 Testing camera..."
    python3 -c "
import cv2
import sys

try:
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print('✅ Camera working! Frame shape:', frame.shape)
        else:
            print('❌ Camera opened but cannot read frames')
        cap.release()
    else:
        print('❌ Cannot open camera')
except Exception as e:
    print('❌ Camera test failed:', e)
"
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file template..."
    cat > .env << 'EOF'
# API Keys - Replace with your actual keys
PORCUPINE_ACCESS_KEY=qqlP6xCMkzy3yWVx9Wg3RDsATOG1d06E1KAgbFilHWeoAl3zcIjkag==
GEMINI_API_KEY=AIzaSyBuFAaIvXFRRX_LfAaTFnVTFFva-eV2Zw8
ELEVENLABS_API_KEY=sk_a815878bc3184834c55fe90e89c9588bcb96759e64d9cb61
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM

# Wake word settings
WAKE_WORD=alexa
CUSTOM_WAKE_WORDS=jarvis,computer,hey google

# Raspberry Pi optimized settings
USE_EMOTION_DETECTION=true
SHOW_WEBCAM=false
PROCESS_EVERY_N_FRAMES=30
CAMERA_WIDTH=320
CAMERA_HEIGHT=240
CAMERA_FPS=15

# Memory settings
USE_MEMORY=true
MEMORY_SIZE=5
SAVE_HISTORY=true

# Voice settings
VOICE_RATE=150
VOICE_VOLUME=0.8
ENABLE_BEEP=true

# Speech recognition settings
SPEECH_RECOGNITION=google
SILENCE_THRESHOLD=1.0
SPEECH_TIMEOUT=2.0
PHRASE_TIMEOUT=5.0
EOF
    echo "✅ .env file created with Pi-optimized settings"
else
    echo "✅ .env file already exists"
fi

# Create startup script
echo "🚀 Creating startup script..."
cat > start_robot.sh << 'EOF'
#!/bin/bash
echo "🤖 Starting Conversation Robot..."

# Activate virtual environment
source robot_env/bin/activate

# Check if camera is available (for Pi)
if [ -e /dev/video0 ]; then
    echo "📷 Camera detected"
else
    echo "⚠️ No camera detected - emotion detection may not work"
fi

# Start the robot
python3 conversation_robot_pi.py
EOF

chmod +x start_robot.sh

# Create systemd service for auto-start (optional)
echo "⚙️ Creating systemd service (optional auto-start)..."
cat > conversation-robot.service << EOF
[Unit]
Description=Conversation Robot
After=network.target sound.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
ExecStart=$(pwd)/start_robot.sh
Restart=always
RestartSec=10
Environment=DISPLAY=:0

[Install]
WantedBy=multi-user.target
EOF

echo "📋 Setup Summary:"
echo "=================="
echo "✅ System packages installed"
echo "✅ Python virtual environment created"
echo "✅ Python packages installed"
echo "✅ Audio system tested"
if [ "$IS_PI" = true ]; then
    echo "✅ Camera system tested"
    echo "✅ Pi-specific optimizations applied"
fi
echo "✅ Configuration files created"
echo "✅ Startup script created"

echo ""
echo "🎯 Next Steps:"
echo "=============="
echo "1. Edit .env file with your actual API keys if needed"
echo "2. Test the robot: ./start_robot.sh"
echo "3. Optional: Install as service: sudo cp conversation-robot.service /etc/systemd/system/"
echo "4. Optional: Enable auto-start: sudo systemctl enable conversation-robot"

if [ "$IS_PI" = true ]; then
    echo ""
    echo "🍓 Raspberry Pi Specific Notes:"
    echo "==============================="
    echo "• Camera interface has been enabled"
    echo "• You may need to reboot for camera changes to take effect"
    echo "• Audio output: Use 'sudo raspi-config' to select audio output (HDMI/3.5mm)"
    echo "• For best performance, use a Class 10 SD card"
    echo "• Consider using a USB microphone for better audio quality"
fi

echo ""
echo "🔧 Troubleshooting:"
echo "==================="
echo "• Audio issues: Check 'alsamixer' and 'pavucontrol'"
echo "• Camera issues: Check 'v4l2-ctl --list-devices'"
echo "• Permission issues: Make sure user is in 'audio' and 'video' groups"
echo "• Performance issues: Close unnecessary applications"

echo ""
echo "✅ Setup complete! Run './start_robot.sh' to start the robot."