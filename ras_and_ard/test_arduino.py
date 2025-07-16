#!/usr/bin/env python3
"""
Simple test script for Arduino connection on Raspberry Pi
This script attempts to connect to an Arduino and send commands
"""

import os
import time
import glob
import subprocess

def test_arduino_connection():
    """Test direct connection to Arduino on Raspberry Pi"""
    print("\n=== ARDUINO CONNECTION TEST ===")
    
    try:
        # First try to fix permissions
        try:
            print("Attempting to fix permissions for Arduino port...")
            subprocess.run(['sudo', 'chmod', '666', '/dev/ttyACM0'], check=False)
            print("✅ Permission fix applied")
        except:
            print("Note: Could not apply permission fix automatically")
        
        # Check for Arduino in USB devices
        try:
            print("\nChecking USB devices...")
            usb_output = subprocess.check_output(['lsusb']).decode('utf-8')
            print(usb_output)
            
            # Check if Arduino is visible
            if any(keyword in usb_output.lower() for keyword in ['arduino', 'uno', 'mega', 'leonardo', 'micro']):
                print("✅ Arduino detected in USB devices")
            else:
                print("⚠️ No Arduino detected in USB devices. Check your connection.")
        except:
            print("Note: Could not check USB devices")
        
        # List available serial ports
        print("\nChecking available serial ports...")
        available_ports = []
        
        # Check standard Arduino ports first
        if os.path.exists('/dev/ttyACM0'):
            available_ports.append('/dev/ttyACM0')
            print("✅ Found standard Arduino port: /dev/ttyACM0")
        
        # Find all tty devices
        try:
            all_ports = glob.glob('/dev/tty*')
            arduino_ports = [p for p in all_ports if ('ACM' in p or 'USB' in p) and p not in available_ports]
            if arduino_ports:
                available_ports.extend(arduino_ports)
                print(f"✅ Found additional Arduino ports: {', '.join(arduino_ports)}")
            
            print(f"\nAll available TTY devices: {', '.join(all_ports)}")
        except Exception as e:
            print(f"Note: Error listing ports: {e}")
        
        # Select port to use
        if not available_ports:
            print("❌ No Arduino ports found. Please check your connection.")
            return False
        
        arduino_port = available_ports[0]
        print(f"\nUsing port: {arduino_port}")
        
        # Import serial here to avoid issues if not installed
        try:
            import serial
        except ImportError:
            print("❌ PySerial not installed. Installing now...")
            subprocess.check_call(['pip3', 'install', 'pyserial'])
            import serial
        
        # Connect to Arduino
        print(f"\nConnecting to Arduino on {arduino_port}...")
        arduino = serial.Serial(arduino_port, 9600, timeout=1)
        print("✅ Connection established")
        
        # Wait for Arduino to reset
        print("Waiting for Arduino to initialize...")
        time.sleep(2)
        
        # Test commands
        test_commands = ["idle", "talk", "shake_hand", "happy", "idle"]
        
        print("\nSending test commands:")
        for cmd in test_commands:
            # Add newline for Arduino's Serial.readStringUntil('\n')
            message = cmd + "\n"
            
            # Send command
            print(f"\nSending: '{cmd}'")
            arduino.write(message.encode('utf-8'))
            arduino.flush()  # Ensure data is sent
            
            # Wait for response
            time.sleep(1)
            
            # Check for response
            if arduino.in_waiting:
                response = arduino.readline().decode('utf-8', errors='ignore').strip()
                print(f"✅ Arduino response: {response}")
            else:
                print("ℹ️ No response (this is normal for many Arduino sketches)")
            
            # Wait between commands
            time.sleep(2)
        
        # Close the connection
        arduino.close()
        print("\n✅ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure Arduino is connected via USB")
        print("2. Try running: sudo chmod 666 /dev/ttyACM0")
        print("3. Check if Arduino is visible with: lsusb")
        print("4. Try unplugging and reconnecting the Arduino")
        print("5. Make sure the Arduino sketch is uploaded and working")
        return False

def send_single_command(command):
    """Send a single command to the Arduino"""
    print(f"\n=== Sending command: {command} ===")
    
    try:
        # Find Arduino port
        arduino_port = None
        if os.path.exists('/dev/ttyACM0'):
            arduino_port = '/dev/ttyACM0'
        else:
            # Try to find other Arduino ports
            arduino_ports = [p for p in glob.glob('/dev/tty*') if 'ACM' in p or 'USB' in p]
            if arduino_ports:
                arduino_port = arduino_ports[0]
        
        if not arduino_port:
            print("❌ No Arduino port found")
            return False
        
        # Try to fix permissions
        try:
            subprocess.run(['sudo', 'chmod', '666', arduino_port], check=False)
        except:
            pass
        
        # Import serial
        import serial
        
        # Connect to Arduino
        print(f"Connecting to Arduino on {arduino_port}...")
        arduino = serial.Serial(arduino_port, 9600, timeout=1)
        time.sleep(2)  # Wait for Arduino to reset
        
        # Send command
        message = command
        if not message.endswith('\n'):
            message += '\n'
            
        print(f"Sending: '{command}'")
        arduino.write(message.encode('utf-8'))
        arduino.flush()
        
        # Wait for response
        time.sleep(1)
        if arduino.in_waiting:
            response = arduino.readline().decode('utf-8', errors='ignore').strip()
            print(f"✅ Arduino response: {response}")
        
        # Close the connection
        arduino.close()
        print(f"✅ Command '{command}' sent successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to send command: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    # Check if a specific command was provided
    if len(sys.argv) > 1:
        command = sys.argv[1]
        send_single_command(command)
    else:
        # Run the full test
        test_arduino_connection() 