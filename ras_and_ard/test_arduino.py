#!/usr/bin/env python3
"""
Arduino Connection Test Script for Raspberry Pi
This script helps diagnose and test Arduino connections on a Raspberry Pi.
"""

import os
import sys
import time
import platform
import glob
import subprocess
import serial
import stat

def print_header(text):
    """Print a header with decoration"""
    print("\n" + "="*60)
    print(f" {text} ".center(60, "="))
    print("="*60 + "\n")

def check_if_raspberry_pi():
    """Check if running on a Raspberry Pi"""
    is_raspberry_pi = False
    pi_model = "Unknown"
    
    if platform.system() == 'Linux':
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                if 'Raspberry Pi' in cpuinfo or 'BCM' in cpuinfo:
                    is_raspberry_pi = True
                    # Try to extract model
                    for line in cpuinfo.split('\n'):
                        if 'Model' in line:
                            pi_model = line.split(':', 1)[1].strip()
                            break
        except Exception as e:
            print(f"Error checking if Raspberry Pi: {e}")
    
    return is_raspberry_pi, pi_model

def check_user_permissions():
    """Check if user has permissions for serial ports"""
    try:
        # Check if user is in dialout group
        groups_output = subprocess.check_output(['groups']).decode('utf-8')
        if 'dialout' in groups_output:
            print("✅ User is in the 'dialout' group - should have permission to access serial ports")
        else:
            print("❌ User is NOT in the 'dialout' group - may have permission issues with serial ports")
            print("   Run the following command and then log out and back in:")
            print("   sudo usermod -a -G dialout $USER")
    except Exception as e:
        print(f"Error checking user permissions: {e}")

def list_usb_devices():
    """List USB devices connected to the system"""
    print_header("USB DEVICES")
    
    try:
        usb_output = subprocess.check_output(['lsusb']).decode('utf-8')
        if not usb_output.strip():
            print("No USB devices detected")
        else:
            print(usb_output)
            
            # Highlight Arduino devices
            arduino_found = False
            for line in usb_output.split('\n'):
                if any(keyword in line.lower() for keyword in ['arduino', 'uno', 'mega', 'leonardo', 'micro']):
                    print(f"🔍 ARDUINO DEVICE FOUND: {line}")
                    arduino_found = True
            
            if not arduino_found:
                print("\n❌ No Arduino devices detected in USB list")
                print("   Make sure your Arduino is properly connected")
    except Exception as e:
        print(f"Error listing USB devices: {e}")

def check_serial_ports():
    """Check available serial ports"""
    print_header("SERIAL PORTS")
    
    try:
        # Check for serial ports
        serial_ports = glob.glob('/dev/tty*')
        if not serial_ports:
            print("No serial ports found")
            return []
        
        print("Available serial ports:")
        arduino_ports = []
        
        # Common Arduino port patterns
        arduino_patterns = ['/dev/ttyACM*', '/dev/ttyUSB*']
        for pattern in arduino_patterns:
            matching_ports = glob.glob(pattern)
            for port in matching_ports:
                arduino_ports.append(port)
                print(f"🔍 POTENTIAL ARDUINO PORT: {port}")
        
        # Show all other serial ports
        other_ports = [p for p in serial_ports if p not in arduino_ports]
        if other_ports:
            print("\nOther serial ports:")
            for port in other_ports:
                print(f"  {port}")
        
        # Check permissions on potential Arduino ports
        if arduino_ports:
            print("\nChecking permissions on potential Arduino ports:")
            for port in arduino_ports:
                try:
                    port_stat = os.stat(port)
                    mode = port_stat.st_mode
                    perms = oct(mode & 0o777)
                    
                    # Check if current user can read/write
                    readable = mode & stat.S_IRUSR
                    writable = mode & stat.S_IWUSR
                    
                    if readable and writable:
                        print(f"✅ {port}: Permissions {perms} - Current user can read/write")
                    else:
                        print(f"❌ {port}: Permissions {perms} - Current user may NOT have read/write access")
                        print(f"   Try: sudo chmod 666 {port}")
                except Exception as e:
                    print(f"Error checking permissions for {port}: {e}")
        
        return arduino_ports
    except Exception as e:
        print(f"Error checking serial ports: {e}")
        return []

def check_dmesg_for_arduino():
    """Check dmesg for Arduino connections"""
    print_header("KERNEL MESSAGES (dmesg)")
    
    try:
        # Look at recent kernel messages for Arduino connection
        dmesg_output = subprocess.check_output(['dmesg', '|', 'grep', '-i', 'tty\\|usb\\|acm\\|arduino', '|', 'tail', '-n', '20'], 
                                             shell=True).decode('utf-8', errors='ignore')
        
        if not dmesg_output.strip():
            print("No relevant kernel messages found")
        else:
            print("Recent relevant kernel messages:")
            print(dmesg_output)
            
            # Highlight Arduino-related messages
            for line in dmesg_output.split('\n'):
                if 'arduino' in line.lower():
                    print(f"🔍 ARDUINO REFERENCE: {line}")
                elif 'ttyACM' in line or 'ttyUSB' in line:
                    print(f"🔍 TTY ASSIGNMENT: {line}")
    except Exception as e:
        print(f"Error checking dmesg: {e}")

def test_arduino_connection(port):
    """Test connection to Arduino on specified port"""
    print_header(f"TESTING ARDUINO CONNECTION ON {port}")
    
    try:
        print(f"Attempting to connect to {port} at 9600 baud...")
        
        # Try to connect to the port
        ser = serial.Serial(port, 9600, timeout=1)
        print(f"✅ Successfully opened port {port}")
        
        # Wait for Arduino to reset
        time.sleep(2)
        
        # Flush any pending data
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        
        # Send test commands
        test_commands = ["idle\n", "talk\n", "idle\n"]
        
        for cmd in test_commands:
            print(f"Sending command: {cmd.strip()}")
            ser.write(cmd.encode())
            ser.flush()
            time.sleep(1)
            
            # Check for response
            if ser.in_waiting:
                response = ser.readline().decode('utf-8', errors='ignore').strip()
                print(f"✅ Received response: {response}")
            else:
                print("ℹ️ No response received (this may be normal depending on your Arduino code)")
        
        # Close the connection
        ser.close()
        print(f"✅ Connection test completed successfully on {port}")
        return True
    except Exception as e:
        print(f"❌ Error testing Arduino connection: {e}")
        
        # Check for permission errors
        if "permission" in str(e).lower():
            print("\nThis appears to be a permission error. Try the following:")
            print("1. Run: sudo usermod -a -G dialout $USER")
            print("2. Log out and log back in")
            print("3. If that doesn't work, try: sudo chmod 666 " + port)
        
        return False

def main():
    """Main function to run all tests"""
    print_header("ARDUINO CONNECTION TEST")
    
    # Check if running on Raspberry Pi
    is_pi, pi_model = check_if_raspberry_pi()
    if is_pi:
        print(f"✅ Running on Raspberry Pi: {pi_model}")
    else:
        print("ℹ️ Not running on a Raspberry Pi")
    
    # Check user permissions
    check_user_permissions()
    
    # List USB devices
    list_usb_devices()
    
    # Check available serial ports
    arduino_ports = check_serial_ports()
    
    # Check dmesg for Arduino connections
    check_dmesg_for_arduino()
    
    # Test connection to each potential Arduino port
    if arduino_ports:
        print_header("CONNECTION TESTS")
        
        success = False
        for port in arduino_ports:
            if test_arduino_connection(port):
                success = True
                print(f"\n✅ Successfully connected to Arduino on {port}")
                print(f"You should use this port in your .env file or environment variables:")
                print(f"ARDUINO_PORT={port}")
                break
        
        if not success:
            print("\n❌ Failed to connect to Arduino on any port")
            print("Please check your connections and permissions")
    else:
        print("\n❌ No potential Arduino ports found to test")
        print("Make sure your Arduino is connected via USB")
    
    print_header("TEST COMPLETE")

if __name__ == "__main__":
    main() 