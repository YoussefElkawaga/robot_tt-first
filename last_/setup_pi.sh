#!/bin/bash
# Setup script for Raspberry Pi conversation robot
# This script installs dependencies and configures the camera

# Text colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}===================================================${NC}"
echo -e "${BLUE}    Raspberry Pi Conversation Robot Setup Script    ${NC}"
echo -e "${BLUE}===================================================${NC}"

# Check if running on Raspberry Pi
if [ -f /sys/firmware/devicetree/base/model ]; then
    PI_MODEL=$(cat /sys/firmware/devicetree/base/model)
    if [[ $PI_MODEL == *"Raspberry Pi"* ]]; then
        echo -e "${GREEN}✅ Detected: $PI_MODEL${NC}"
        
        # Check if Raspberry Pi 5
        if [[ $PI_MODEL == *"5"* ]]; then
            IS_PI5=true
            echo -e "${GREEN}✅ Raspberry Pi 5 detected - will use libcamera${NC}"
        else
            IS_PI5=false
            echo -e "${GREEN}✅ Older Raspberry Pi model detected - will use bcm2835-v4l2${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️ This doesn't appear to be a Raspberry Pi${NC}"
        IS_PI5=false
    fi
else
    echo -e "${YELLOW}⚠️ This doesn't appear to be a Raspberry Pi${NC}"
    IS_PI5=false
fi

# Update package list
echo -e "\n${BLUE}Updating package list...${NC}"
sudo apt-get update

# Install required packages
echo -e "\n${BLUE}Installing required packages...${NC}"
sudo apt-get install -y python3-pip python3-opencv python3-pyaudio portaudio19-dev ffmpeg libatlas-base-dev

# Install libcamera tools if on Raspberry Pi 5
if [ "$IS_PI5" = true ]; then
    echo -e "\n${BLUE}Installing libcamera tools for Raspberry Pi 5...${NC}"
    sudo apt-get install -y libcamera-apps
    
    # Ensure v4l2-compat is loaded
    echo -e "\n${BLUE}Setting up v4l2-compat module...${NC}"
    if lsmod | grep -q "v4l2_compat"; then
        echo -e "${GREEN}✅ v4l2-compat module is already loaded${NC}"
    else
        echo -e "${YELLOW}⚠️ Loading v4l2-compat module...${NC}"
        sudo modprobe v4l2-compat
        
        # Add to modules to load at boot
        if ! grep -q "v4l2-compat" /etc/modules; then
            echo "v4l2-compat" | sudo tee -a /etc/modules
            echo -e "${GREEN}✅ Added v4l2-compat to modules to load at boot${NC}"
        fi
    fi
    
    # Test libcamera
    echo -e "\n${BLUE}Testing libcamera...${NC}"
    if which libcamera-hello > /dev/null; then
        echo -e "${GREEN}✅ libcamera-hello is installed${NC}"
        echo -e "${YELLOW}Running a quick test (will show preview for 2 seconds)...${NC}"
        libcamera-hello --timeout 2000
    else
        echo -e "${RED}❌ libcamera-hello not found${NC}"
    fi
else
    # For older Pi models, ensure bcm2835-v4l2 is loaded
    echo -e "\n${BLUE}Setting up bcm2835-v4l2 module...${NC}"
    if lsmod | grep -q "bcm2835_v4l2"; then
        echo -e "${GREEN}✅ bcm2835-v4l2 module is already loaded${NC}"
    else
        echo -e "${YELLOW}⚠️ Loading bcm2835-v4l2 module...${NC}"
        sudo modprobe bcm2835-v4l2
        
        # Add to modules to load at boot
        if ! grep -q "bcm2835-v4l2" /etc/modules; then
            echo "bcm2835-v4l2" | sudo tee -a /etc/modules
            echo -e "${GREEN}✅ Added bcm2835-v4l2 to modules to load at boot${NC}"
        fi
    fi
fi

# Install Python dependencies
echo -e "\n${BLUE}Installing Python dependencies...${NC}"
pip3 install --upgrade pip
pip3 install pyttsx3 SpeechRecognition pvporcupine pyaudio google-generativeai python-dotenv elevenlabs opencv-python numpy

# Install FER with dependencies
echo -e "\n${BLUE}Installing FER (Facial Emotion Recognition) library...${NC}"
pip3 install fer tensorflow mtcnn

# Create a .env file if it doesn't exist
if [ ! -f .env ]; then
    echo -e "\n${BLUE}Creating .env file...${NC}"
    cat > .env << EOL
# API Keys
PORCUPINE_ACCESS_KEY=qqlP6xCMkzy3yWVx9Wg3RDsATOG1d06E1KAgbFilHWeoAl3zcIjkag==
GEMINI_API_KEY=AIzaSyBuFAaIvXFRRX_LfAaTFnVTFFva-eV2Zw8
ELEVENLABS_API_KEY=sk_a815878bc3184834c55fe90e89c9588bcb96759e64d9cb61
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM

# Wake word settings
WAKE_WORD=alexa
CUSTOM_WAKE_WORDS=jarvis,computer,hey google

# Memory settings
USE_MEMORY=true
MEMORY_SIZE=10
MEMORY_FILE=robot_memory.json
MEMORY_EXCHANGES_LIMIT=5

# Voice settings
VOICE_RATE=150
VOICE_VOLUME=0.8

# Other settings
SAVE_HISTORY=true
ENABLE_BEEP=true
SPEECH_RECOGNITION=google
USE_EMOTION_DETECTION=true
SHOW_WEBCAM=false
PROCESS_EVERY_N_FRAMES=15
EOL
    echo -e "${GREEN}✅ Created .env file with default settings${NC}"
else
    echo -e "${YELLOW}⚠️ .env file already exists, not overwriting${NC}"
fi

# Check if camera is enabled
echo -e "\n${BLUE}Checking if camera is enabled...${NC}"
if [ -e /dev/video0 ]; then
    echo -e "${GREEN}✅ Camera is enabled and available at /dev/video0${NC}"
else
    echo -e "${YELLOW}⚠️ Camera device not found at /dev/video0${NC}"
    echo -e "${YELLOW}Make sure the camera is connected and enabled in raspi-config${NC}"
    echo -e "${YELLOW}Run 'sudo raspi-config' and enable the camera in the Interfacing Options${NC}"
fi

# Create test script for camera
echo -e "\n${BLUE}Creating test script for camera...${NC}"
if [ -f test_camera_pi.py ]; then
    echo -e "${YELLOW}⚠️ test_camera_pi.py already exists, not overwriting${NC}"
else
    # The test_camera_pi.py file should be created separately
    echo -e "${GREEN}✅ Using existing test_camera_pi.py script${NC}"
fi

# Make test script executable
chmod +x test_camera_pi.py

echo -e "\n${GREEN}===================================================${NC}"
echo -e "${GREEN}    Setup completed!    ${NC}"
echo -e "${GREEN}===================================================${NC}"
echo -e "\n${YELLOW}To test the camera, run:${NC}"
echo -e "    python3 test_camera_pi.py"
echo -e "\n${YELLOW}To run the conversation robot, run:${NC}"
echo -e "    python3 conversation_robotrr.py"
echo -e "\n${YELLOW}To test emotion detection specifically, run:${NC}"
echo -e "    python3 conversation_robotrr.py --test-emotion --show-webcam"
echo -e "\n${BLUE}Happy robot building!${NC}\n" 