import serial
import time
import sys

def main():
    print("Testing Arduino Talk Duration Command")
    print("------------------------------------")
    
    # Default Arduino port based on platform
    if sys.platform.startswith('win'):
        default_port = 'COM3'
    else:
        default_port = '/dev/ttyACM0'
    
    # Get port from command line or use default
    port = sys.argv[1] if len(sys.argv) > 1 else default_port
    baud_rate = 9600
    
    print(f"Using port: {port} at {baud_rate} baud")
    
    try:
        # Connect to Arduino
        print("Opening serial connection...")
        arduino = serial.Serial(port, baud_rate, timeout=1)
        time.sleep(2)  # Wait for connection to stabilize
        
        # Clear any pending data
        arduino.reset_input_buffer()
        arduino.reset_output_buffer()
        
        # Send commands with different durations
        commands = [
            ("idle", "Setting to idle pose"),
            ("talk", "Simple talk command (default duration)"),
            ("talk:3000", "Talk with 3 second duration"),
            ("talk:5000", "Talk with 5 second duration"),
            ("idle", "Back to idle pose"),
            ("shake_hand", "Testing handshake command"),
            ("idle", "Back to idle pose"),
            ("happy", "Testing happy dance")
        ]
        
        for cmd, desc in commands:
            print(f"\n{desc}")
            print(f"Sending: {cmd}")
            
            # Add newline if not present
            if not cmd.endswith('\n'):
                cmd += '\n'
            
            # Send command
            arduino.write(cmd.encode('utf-8'))
            arduino.flush()
            
            # Wait for response
            time.sleep(0.5)
            if arduino.in_waiting:
                response = arduino.readline().decode('utf-8', errors='ignore').strip()
                print(f"Response: {response}")
            
            # Wait longer for commands that take time to execute
            if "talk:5000" in cmd:
                print("Waiting for 5 second talk animation...")
                time.sleep(5)
            elif "talk:3000" in cmd:
                print("Waiting for 3 second talk animation...")
                time.sleep(3)
            elif "shake_hand" in cmd or "happy" in cmd:
                print("Waiting for animation to complete...")
                time.sleep(3)
            else:
                time.sleep(1)
        
        # Close connection
        arduino.close()
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure Arduino is connected")
        print("2. Check if the port is correct")
        print("3. Make sure the Arduino has the updated code with talk duration support")
        print("4. Try running with a specific port: python test_talk_duration.py /dev/ttyACM0")
        return False
    
    return True

if __name__ == "__main__":
    main() 