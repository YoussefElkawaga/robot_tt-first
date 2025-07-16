#!/bin/bash
# Setup script for Conversation Robot on Raspberry Pi 5
# This script installs all necessary dependencies for the emotion detection system

echo "==== Conversation Robot Setup Script for Raspberry Pi ===="
echo "This script will install all necessary dependencies for emotion detection."
echo "It may take some time to complete."
echo

# Function to check if we're running on a Raspberry Pi
check_raspberry_pi() {
    if [ -f /sys/firmware/devicetree/base/model ]; then
        model=$(cat /sys/firmware/devicetree/base/model)
        if [[ $model == *"Raspberry Pi"* ]]; then
            echo "✅ Detected: $model"
            return 0
        fi
    fi
    echo "❌ This script is designed for Raspberry Pi only."
    return 1
}

# Update package lists
update_packages() {
    echo "Updating package lists..."
    sudo apt-get update
    echo "Upgrading packages..."
    sudo apt-get upgrade -y
}

# Install system dependencies
install_system_deps() {
    echo "Installing system dependencies..."
    sudo apt-get install -y python3-pip python3-dev python3-opencv
    sudo apt-get install -y libatlas-base-dev libhdf5-dev libhdf5-serial-dev
    sudo apt-get install -y libjasper-dev libqtgui4 libqt4-test
    sudo apt-get install -y portaudio19-dev python3-pyaudio
}

# Install Python dependencies
install_python_deps() {
    echo "Installing Python dependencies..."
    pip3 install --upgrade pip
    pip3 install numpy==1.22.0
    pip3 install opencv-python
    pip3 install opencv-contrib-python
    
    # Install TensorFlow (compatible version for Raspberry Pi)
    pip3 install tensorflow==2.9.0
    
    # Install MTCNN with specific version for better compatibility
    pip3 install mtcnn==0.1.0
    
    # Install FER
    pip3 install fer
    
    # Install other required packages
    pip3 install python-dotenv
    pip3 install SpeechRecognition
    pip3 install pyaudio
    pip3 install pyttsx3
    pip3 install google-generativeai
    pip3 install elevenlabs
}

# Configure camera module
setup_camera() {
    echo "Setting up camera module..."
    
    # Enable camera in config
    if [ -f /boot/config.txt ]; then
        # Check if camera is already enabled
        if grep -q "^start_x=1" /boot/config.txt; then
            echo "Camera already enabled in config.txt"
        else
            echo "Enabling camera in config.txt..."
            sudo sed -i 's/^start_x=0/start_x=1/' /boot/config.txt
            if ! grep -q "^start_x=" /boot/config.txt; then
                echo "Adding start_x=1 to config.txt..."
                echo "start_x=1" | sudo tee -a /boot/config.txt
            fi
        fi
        
        # Ensure gpu_mem is at least 128
        if grep -q "^gpu_mem=" /boot/config.txt; then
            current_gpu_mem=$(grep "^gpu_mem=" /boot/config.txt | cut -d'=' -f2)
            if [ "$current_gpu_mem" -lt 128 ]; then
                echo "Setting gpu_mem to 128 in config.txt..."
                sudo sed -i 's/^gpu_mem=.*/gpu_mem=128/' /boot/config.txt
            fi
        else
            echo "Adding gpu_mem=128 to config.txt..."
            echo "gpu_mem=128" | sudo tee -a /boot/config.txt
        fi
    fi
    
    # Load the V4L2 driver for Pi Camera
    sudo modprobe bcm2835-v4l2
    
    # Add the driver to modules to load at boot
    if ! grep -q "bcm2835-v4l2" /etc/modules; then
        echo "Adding bcm2835-v4l2 to /etc/modules..."
        echo "bcm2835-v4l2" | sudo tee -a /etc/modules
    fi
}

# Test camera
test_camera() {
    echo "Testing camera..."
    if [ -c /dev/video0 ]; then
        echo "✅ Camera device found at /dev/video0"
        
        # Try to capture a test image
        echo "Capturing test image..."
        python3 -c "
import cv2
import time
cap = cv2.VideoCapture(0)
time.sleep(2)  # Give camera time to initialize
ret, frame = cap.read()
if ret:
    cv2.imwrite('camera_test.jpg', frame)
    print('✅ Test image captured successfully as camera_test.jpg')
else:
    print('❌ Failed to capture test image')
cap.release()
"
    else
        echo "❌ Camera device not found at /dev/video0"
        echo "Please check your camera connection and try again."
    fi
}

# Main execution
main() {
    # Check if we're on a Raspberry Pi
    if ! check_raspberry_pi; then
        echo "This script is designed for Raspberry Pi. Exiting."
        exit 1
    fi
    
    echo "Starting setup..."
    update_packages
    install_system_deps
    install_python_deps
    setup_camera
    test_camera
    
    echo
    echo "==== Setup Complete ===="
    echo "You may need to reboot your Raspberry Pi for all changes to take effect."
    echo "After rebooting, run the conversation_robotrr.py script to start the robot."
    echo "To reboot now, run: sudo reboot"
}

# Run the main function
main 