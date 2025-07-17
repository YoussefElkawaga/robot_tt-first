#!/bin/bash

# Robot Startup Service Installer for Ubuntu on Raspberry Pi 5
# This script sets up conversation_robotrr (1).py as a systemd service
# that will start automatically at boot

# Exit on error
set -e

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Display header
echo -e "${GREEN}====================================================${NC}"
echo -e "${GREEN}  Ubuntu Conversation Robot Service Installer       ${NC}"
echo -e "${GREEN}====================================================${NC}"

# Check if running on Ubuntu
if ! grep -q "Ubuntu" /etc/os-release; then
    echo -e "${YELLOW}Warning: This script is designed for Ubuntu. Your system may be different.${NC}"
    echo -e "${YELLOW}Do you wish to continue anyway? (y/n)${NC}"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo -e "${RED}Installation aborted.${NC}"
        exit 1
    fi
fi

# Get the current directory or use the provided path
SCRIPT_DIR="$PWD"
PYTHON_PATH=$(which python3)

# Check if files exist
if [ ! -f "$SCRIPT_DIR/conversation_robotrr (1).py" ]; then
    echo -e "${RED}Error: conversation_robotrr (1).py not found in $SCRIPT_DIR${NC}"
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/arduino_helper.py" ]; then
    echo -e "${RED}Error: arduino_helper.py not found in $SCRIPT_DIR${NC}"
    exit 1
fi

echo -e "${GREEN}Found required Python scripts in $SCRIPT_DIR${NC}"

# Install required Ubuntu packages if missing
echo -e "${YELLOW}Checking for required Ubuntu packages...${NC}"
if ! dpkg -l | grep -q python3-pip; then
    echo -e "${YELLOW}Installing python3-pip...${NC}"
    sudo apt-get update
    sudo apt-get install -y python3-pip
fi

if ! dpkg -l | grep -q python3-venv; then
    echo -e "${YELLOW}Installing python3-venv...${NC}"
    sudo apt-get install -y python3-venv
fi

# Create bash wrapper script
echo -e "${YELLOW}Creating wrapper script...${NC}"
cat > "$SCRIPT_DIR/start_robot.sh" << EOL
#!/bin/bash
# Wrapper script to start the conversation robot on Ubuntu
cd "$SCRIPT_DIR"

# Wait for network to be fully initialized (important for Ubuntu on Raspberry Pi)
sleep 10

# Source virtual environment if it exists
if [ -f "$SCRIPT_DIR/robot-env/bin/activate" ]; then
    source "$SCRIPT_DIR/robot-env/bin/activate"
fi

# Ubuntu udev rules might be applied, but let's make sure permissions are set
for PORT in /dev/ttyACM* /dev/ttyUSB*; do
    if [ -e "\$PORT" ]; then
        sudo chmod 666 "\$PORT"
    fi
done

# Make sure audio device is ready (sometimes needed on Ubuntu)
if [ -e /usr/bin/pulseaudio ]; then
    pulseaudio --start --exit-idle-time=-1
fi

# Start the robot with full path to avoid Ubuntu path issues
echo "Starting conversation robot at \$(date)" >> "$SCRIPT_DIR/robot.log"
exec $PYTHON_PATH "$SCRIPT_DIR/conversation_robotrr (1).py" >> "$SCRIPT_DIR/robot.log" 2>&1
EOL

# Make wrapper script executable
chmod +x "$SCRIPT_DIR/start_robot.sh"
echo -e "${GREEN}Created wrapper script: $SCRIPT_DIR/start_robot.sh${NC}"

# Create systemd service file
echo -e "${YELLOW}Creating systemd service file...${NC}"
SERVICE_FILE="/tmp/conversation-robot.service"

cat > $SERVICE_FILE << EOL
[Unit]
Description=Conversation Robot Service
After=network.target pulseaudio.service
Wants=network-online.target
StartLimitIntervalSec=500
StartLimitBurst=5

[Service]
Type=simple
User=$USER
ExecStart=$SCRIPT_DIR/start_robot.sh
Restart=on-failure
RestartSec=5
StandardOutput=append:$SCRIPT_DIR/robot.log
StandardError=append:$SCRIPT_DIR/robot.log
Environment="DISPLAY=:0"
Environment="PYTHONUNBUFFERED=1"

# Ubuntu-specific security settings
NoNewPrivileges=true
ProtectHome=read-only
ProtectSystem=full
# Allow serial port access
DeviceAllow=/dev/ttyACM0 rwm
DeviceAllow=/dev/ttyUSB0 rwm

[Install]
WantedBy=multi-user.target
EOL

# Move service file to systemd directory using sudo
echo -e "${YELLOW}Installing service file (requires sudo)...${NC}"
sudo mv $SERVICE_FILE /etc/systemd/system/conversation-robot.service
sudo chmod 644 /etc/systemd/system/conversation-robot.service

# Create udev rules for Arduino on Ubuntu
echo -e "${YELLOW}Setting up udev rules for Arduino access...${NC}"
UDEV_FILE="/tmp/99-arduino-permissions.rules"

cat > $UDEV_FILE << EOL
# Arduino permissions for Ubuntu
SUBSYSTEM=="tty", KERNEL=="ttyACM[0-9]*", GROUP="dialout", MODE="0666"
SUBSYSTEM=="tty", KERNEL=="ttyUSB[0-9]*", GROUP="dialout", MODE="0666"
EOL

sudo mv $UDEV_FILE /etc/udev/rules.d/99-arduino-permissions.rules
sudo udevadm control --reload-rules
sudo udevadm trigger

# Add current user to the dialout group (needed on Ubuntu)
echo -e "${YELLOW}Adding user to dialout group...${NC}"
sudo usermod -a -G dialout $USER
echo -e "${YELLOW}NOTE: You may need to log out and back in for the group changes to take effect${NC}"

# Reload systemd to recognize the new service
echo -e "${YELLOW}Reloading systemd...${NC}"
sudo systemctl daemon-reload

# Enable the service to start at boot
echo -e "${YELLOW}Enabling conversation-robot service to start at boot...${NC}"
sudo systemctl enable conversation-robot.service

# Start the service now
echo -e "${YELLOW}Starting conversation-robot service...${NC}"
sudo systemctl start conversation-robot.service

# Check service status
echo -e "${YELLOW}Checking service status...${NC}"
sudo systemctl status conversation-robot.service

echo -e "${GREEN}====================================================${NC}"
echo -e "${GREEN} Conversation Robot service has been installed!      ${NC}"
echo -e "${GREEN} Configured specifically for Ubuntu on Raspberry Pi  ${NC}"
echo -e "${GREEN}====================================================${NC}"
echo -e "${YELLOW}Service commands:${NC}"
echo -e "- Check status:    ${GREEN}sudo systemctl status conversation-robot.service${NC}"
echo -e "- View logs:       ${GREEN}cat $SCRIPT_DIR/robot.log${NC}"
echo -e "- Follow logs:     ${GREEN}tail -f $SCRIPT_DIR/robot.log${NC}"
echo -e "- Restart service: ${GREEN}sudo systemctl restart conversation-robot.service${NC}"
echo -e "- Stop service:    ${GREEN}sudo systemctl stop conversation-robot.service${NC}"
echo -e "- Disable at boot: ${GREEN}sudo systemctl disable conversation-robot.service${NC}"
echo -e "\n${YELLOW}IMPORTANT: For the Arduino permissions to work properly,${NC}"
echo -e "${YELLOW}you may need to reboot your Raspberry Pi: ${GREEN}sudo reboot${NC}"