#!/usr/bin/env python3
"""
Arduino Connection Test Script for Raspberry Pi
This script helps diagnose and test Arduino connections on a Raspberry Pi.
"""

import serial
import time
import sys
import glob

def list_serial_ports():
    """List all available serial ports"""
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        # This excludes your current terminal "/dev/tty"
        ports = glob.glob('/dev/tty[A-Za-z]*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/tty.*')
    else:
        raise EnvironmentError('Unsupported platform')

    result = []
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass
    return result

def find_arduino_port():
    """Try to automatically find the Arduino port"""
    ports = list_serial_ports()
    print(f"Available ports: {ports}")
    
    # For Raspberry Pi, prioritize ACM ports
    if sys.platform.startswith('linux'):
        for port in ports:
            if 'ACM' in port:
                return port
    
    # Return first available port or None if none found
    return ports[0] if ports else None

def test_arduino_command(command, port=None, baud_rate=9600, add_newline=True):
    """Test sending a command to Arduino"""
    if port is None:
        port = find_arduino_port()
        if not port:
            print("❌ No serial ports found. Is Arduino connected?")
            return False
    
    try:
        print(f"Opening port {port} at {baud_rate} baud...")
        arduino = serial.Serial(port, baud_rate, timeout=1)
        time.sleep(2)  # Wait for connection to stabilize
        
        # Add newline if requested (for Arduino's readStringUntil('\n'))
        if add_newline and not command.endswith('\n'):
            command += '\n'
        
        print(f"Sending command: '{command.strip()}'")
        arduino.write(command.encode('utf-8'))
        arduino.flush()  # Ensure data is sent
        
        # Wait for response
        time.sleep(0.5)
        if arduino.in_waiting:
            response = arduino.readline().decode('utf-8', errors='ignore').strip()
            print(f"Response: {response}")
        
        arduino.close()
        print("✅ Command sent successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("Arduino Communication Test")
    print("--------------------------")
    
    # List available ports
    print("\nScanning for available ports...")
    ports = list_serial_ports()
    if ports:
        print(f"Found {len(ports)} ports: {', '.join(ports)}")
    else:
        print("No ports found. Make sure Arduino is connected.")
        sys.exit(1)
    
    # Try to find Arduino port
    arduino_port = find_arduino_port()
    if arduino_port:
        print(f"Using port: {arduino_port}")
    else:
        arduino_port = input("Enter Arduino port manually: ")
    
    # Menu for testing commands
    while True:
        print("\nTest Commands:")
        print("1. idle")
        print("2. talk")
        print("3. talk with duration (talk:5000)")
        print("4. happy")
        print("5. shake_hand")
        print("6. Custom command")
        print("7. Exit")
        
        choice = input("\nEnter choice (1-7): ")
        
        if choice == '1':
            test_arduino_command("idle", arduino_port)
        elif choice == '2':
            test_arduino_command("talk", arduino_port)
        elif choice == '3':
            duration = input("Enter talk duration in ms (e.g. 5000): ")
            test_arduino_command(f"talk:{duration}", arduino_port)
        elif choice == '4':
            test_arduino_command("happy", arduino_port)
        elif choice == '5':
            test_arduino_command("shake_hand", arduino_port)
        elif choice == '6':
            cmd = input("Enter custom command: ")
            test_arduino_command(cmd, arduino_port)
        elif choice == '7':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Try again.")
        
        time.sleep(1) 