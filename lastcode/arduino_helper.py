import serial
import time
import os
import glob
import subprocess
import sys

def find_arduino_port():
    """Try to find the Arduino port automatically"""
    # Common Arduino port patterns
    port_patterns = ['/dev/ttyACM*', '/dev/ttyUSB*']
    
    # Check each pattern
    for pattern in port_patterns:
        ports = glob.glob(pattern)
        if ports:
            return ports[0]  # Return the first match
    
    return '/dev/ttyACM0'  # Default fallback

def fix_permissions(port):
    """Try to fix port permissions automatically"""
    try:
        # Try to make the port accessible without sudo
        print(f"Attempting to fix permissions for {port}...")
        # Use NOPASSWD sudo for permission fixing
        # This assumes you've set up sudo permissions correctly
        result = subprocess.run(['sudo', '-n', 'chmod', '666', port], 
                              capture_output=True, text=True)
                              
        if result.returncode != 0:
            print("Permission fix failed with sudo -n, trying with regular sudo...")
            # Regular sudo might prompt for password
            os.system(f"sudo chmod 666 {port}")
            
        return True
    except Exception as e:
        print(f"Error fixing permissions: {e}")
        return False

def send_arduino_command(command, port=None, baud_rate=9600):
    """Send a command to Arduino with automatic permission handling"""
    # Try to find the Arduino port automatically if not specified
    arduino_port = port or find_arduino_port()
    
    try:
        # First try without permission fixing
        try:
            arduino = serial.Serial(arduino_port, baud_rate, timeout=1)
        except Exception as e:
            # Check if it's a permission error
            if "permission" in str(e).lower() or "access" in str(e).lower():
                print("Permission error detected, attempting to fix...")
                # If permission error, try to fix permissions
                fix_permissions(arduino_port)
                time.sleep(0.5)  # Wait for permission change to take effect
                arduino = serial.Serial(arduino_port, baud_rate, timeout=1)
            else:
                # Re-raise if it's not a permission error
                raise
        
        print(f"✅ Connected to Arduino on {arduino_port}")
        time.sleep(2)  # Wait for Arduino to reset after connection
        
        # Add newline character for Arduino's readStringUntil('\n') if not present
        if not command.endswith('\n'):
            command += '\n'
            
        arduino.write(command.encode('utf-8'))
        arduino.flush()  # Ensure data is sent
        print(f"✓ Sent: {command.strip()}")
        
        # Wait for a moment to see if there's any response
        time.sleep(1)
        if arduino.in_waiting:
            response = arduino.readline().decode('utf-8', errors='ignore').strip()
            print(f"Arduino response: {response}")
        
        arduino.close()
        print("✅ Connection closed")
        return True
        
    except Exception as e:
        print(f"⚠️ Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Run this command to add your user to the dialout group:")
        print("   sudo usermod -a -G dialout $USER")
        print("   (log out and log back in for changes to take effect)")
        print("2. If that doesn't work, try setting up sudo without password for Arduino commands:")
        print("   sudo visudo -f /etc/sudoers.d/arduino-permissions")
        print("   And add: yourusername ALL=(ALL) NOPASSWD: /bin/chmod 666 /dev/ttyACM*, /bin/chmod 666 /dev/ttyUSB*")
        print("3. Check if your Arduino is properly connected")
        print("4. Verify the correct port with: ls -l /dev/tty*")
        print("5. Install pyserial if needed: pip install pyserial")
        return False

# If script is run directly, use command line arguments
if __name__ == "__main__":
    # Check if command was provided
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        # Optional port argument
        port = None
        if len(sys.argv) > 2:
            port = sys.argv[2]
            
        # Execute command
        send_arduino_command(command, port)
    else:
        print("Usage: python arduino_helper.py <command> [port]")
        print("Example: python arduino_helper.py 'shake_hand' '/dev/ttyACM0'") 