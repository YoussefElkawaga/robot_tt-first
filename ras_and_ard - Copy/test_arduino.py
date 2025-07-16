#!/usr/bin/env python3
import serial
import time
import os
import sys
import platform
import glob

def list_serial_ports():
    """List all available serial ports"""
    try:
        import serial.tools.list_ports
        ports = serial.tools.list_ports.comports()
        
        if not ports:
            print("No serial ports found. Make sure your Arduino is connected.")
            return []
        
        print("\nAvailable serial ports:")
        port_list = []
        for i, port in enumerate(ports):
            description = f"{port.device}"
            if port.description:
                description += f" - {port.description}"
            if port.manufacturer:
                description += f" ({port.manufacturer})"
            if "Arduino" in port.description or "Arduino" in str(port.manufacturer):
                description += " [ARDUINO DETECTED]"
            
            print(f"  {i+1}. {description}")
            port_list.append(port.device)
        
        return port_list
    except ImportError:
        print("Serial tools not available. Cannot list serial ports.")
        return []
    except Exception as e:
        print(f"Error listing serial ports: {e}")
        return []

def find_arduino_port():
    """Find the most likely Arduino port"""
    # Default ports by platform
    default_port = "COM3" if platform.system() == "Windows" else "/dev/ttyACM0"
    
    # Try to find Arduino ports
    try:
        import serial.tools.list_ports
        ports = serial.tools.list_ports.comports()
        
        # First look for ports that explicitly identify as Arduino
        for port in ports:
            if port.manufacturer and "arduino" in port.manufacturer.lower():
                print(f"Found Arduino by manufacturer: {port.device}")
                return port.device
            if port.description and "arduino" in port.description.lower():
                print(f"Found Arduino by description: {port.device}")
                return port.device
        
        # If we're on a Raspberry Pi, look for /dev/serial/by-id paths
        if platform.system() == 'Linux':
            try:
                by_id_path = "/dev/serial/by-id"
                if os.path.exists(by_id_path):
                    arduino_paths = [os.path.join(by_id_path, f) for f in os.listdir(by_id_path) 
                                    if "arduino" in f.lower()]
                    if arduino_paths:
                        print(f"Found Arduino at stable path: {arduino_paths[0]}")
                        return arduino_paths[0]
            except:
                pass
            
            # Check common Raspberry Pi ports
            pi_ports = ['/dev/ttyACM0', '/dev/ttyACM1', '/dev/ttyUSB0', '/dev/ttyUSB1']
            for port in pi_ports:
                if os.path.exists(port):
                    print(f"Found potential Arduino port: {port}")
                    return port
        
        # If no Arduino found, return first available port or default
        if ports:
            print(f"No Arduino specifically identified. Using first available port: {ports[0].device}")
            return ports[0].device
    except:
        pass
    
    print(f"Using default port: {default_port}")
    return default_port

def test_arduino_connection(port=None, baud_rate=9600):
    """Test Arduino connection by sending commands"""
    if port is None:
        port = find_arduino_port()
    
    print(f"\nTesting Arduino connection on {port} at {baud_rate} baud...")
    
    try:
        # Open the serial connection
        arduino = serial.Serial(port, baud_rate, timeout=1)
        print("✅ Serial connection opened successfully")
        
        # Wait for Arduino to reset after connection
        time.sleep(2)
        
        # Send test commands
        test_commands = ["idle\n", "talk\n", "shake_hand\n", "happy\n", "idle\n"]
        
        for cmd in test_commands:
            print(f"Sending command: {cmd.strip()}")
            arduino.write(cmd.encode('utf-8'))
            time.sleep(1)  # Wait for Arduino to process
            
            # Check for response
            if arduino.in_waiting:
                response = arduino.readline().decode('utf-8', errors='ignore').strip()
                print(f"Arduino response: {response}")
            
            # Wait between commands
            time.sleep(2)
        
        # Close the connection
        arduino.close()
        print("✅ Arduino test completed successfully!")
        return True
    
    except Exception as e:
        print(f"❌ Error: {e}")
        
        # If permission error on Linux, suggest fix
        if "permission" in str(e).lower() and platform.system() == "Linux":
            print("\nPermission error detected. Try the following:")
            print(f"1. Run: sudo chmod 666 {port}")
            print("2. Or add your user to the dialout group:")
            print("   sudo usermod -a -G dialout $USER")
            print("   (logout and login again for changes to take effect)")
        
        return False

def main():
    """Main function"""
    print("Arduino Connection Test Script")
    print("=============================")
    
    # List available ports
    available_ports = list_serial_ports()
    
    # If command line arguments are provided, use them
    if len(sys.argv) > 1:
        port = sys.argv[1]
        baud_rate = int(sys.argv[2]) if len(sys.argv) > 2 else 9600
        test_arduino_connection(port, baud_rate)
    else:
        # Let user select a port or use auto-detection
        if available_ports:
            print("\nOptions:")
            print("1. Auto-detect Arduino port")
            for i, port in enumerate(available_ports):
                print(f"{i+2}. Use {port}")
            
            try:
                choice = input("\nEnter your choice (default: 1): ").strip()
                if not choice or choice == "1":
                    port = find_arduino_port()
                else:
                    idx = int(choice) - 2
                    if 0 <= idx < len(available_ports):
                        port = available_ports[idx]
                    else:
                        print("Invalid choice. Using auto-detection.")
                        port = find_arduino_port()
                
                test_arduino_connection(port)
            except Exception as e:
                print(f"Error: {e}")
                print("Using auto-detection.")
                test_arduino_connection()
        else:
            # No ports found, try auto-detection
            test_arduino_connection()

if __name__ == "__main__":
    main() 