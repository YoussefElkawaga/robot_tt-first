#!/usr/bin/env python3
"""
Raspberry Pi Camera Test Script
This script tests camera functionality on Raspberry Pi with detailed diagnostics
"""

import os
import sys
import time
import platform
import glob
import cv2
import numpy as np

def check_raspberry_pi():
    """Check if we're running on a Raspberry Pi and which model"""
    is_pi = False
    pi_model = "Unknown"
    is_pi5 = False
    
    if platform.system() == 'Linux' and os.path.exists('/sys/firmware/devicetree/base/model'):
        try:
            with open('/sys/firmware/devicetree/base/model', 'r') as f:
                model = f.read()
                if 'Raspberry Pi' in model:
                    is_pi = True
                    pi_model = model.strip(chr(0))
                    is_pi5 = '5' in pi_model
                    print(f"✅ Detected: {pi_model}")
        except Exception as e:
            print(f"❌ Error reading device model: {e}")
    
    if not is_pi:
        print("ℹ️ Not running on a Raspberry Pi")
    
    return is_pi, pi_model, is_pi5

def check_camera_modules():
    """Check for camera modules and drivers"""
    print("\nChecking camera modules and drivers:")
    
    # Check for V4L2 devices
    print("\nChecking for video devices:")
    video_devices = glob.glob('/dev/video*')
    if video_devices:
        print(f"✅ Found {len(video_devices)} video devices: {', '.join(video_devices)}")
    else:
        print("❌ No video devices found in /dev/")
    
    # Check if we're on a Raspberry Pi
    is_pi, pi_model, is_pi5 = check_raspberry_pi()
    
    if is_pi:
        if is_pi5:
            check_pi5_camera()
        else:
            check_older_pi_camera()
    
    return video_devices

def check_pi5_camera():
    """Check Raspberry Pi 5 specific camera setup"""
    print("\nChecking libcamera setup (for Raspberry Pi 5):")
    
    # Check if libcamera-hello is available
    try:
        result = os.system("which libcamera-hello >/dev/null 2>&1")
        if result == 0:
            print("✅ libcamera-hello is installed")
            
            # Try to run libcamera-hello with minimal output
            print("Running libcamera-hello to test camera...")
            test_result = os.system("libcamera-hello --timeout 1000 --preview 0 >/dev/null 2>&1")
            if test_result == 0:
                print("✅ libcamera-hello test successful")
            else:
                print(f"⚠️ libcamera-hello test returned error code {test_result}")
        else:
            print("⚠️ libcamera-hello not found")
    except Exception as e:
        print(f"❌ Error testing libcamera: {e}")
    
    # Check for v4l2 compatibility layer
    try:
        with open('/proc/modules', 'r') as modules:
            module_content = modules.read()
            if 'v4l2_compat' in module_content:
                print("✅ v4l2-compat module is loaded")
            else:
                print("⚠️ v4l2-compat module is not loaded")
                print("Attempting to load the module...")
                os.system("sudo modprobe v4l2-compat >/dev/null 2>&1")
                time.sleep(1)
                
                # Check again
                with open('/proc/modules', 'r') as modules:
                    if 'v4l2_compat' in modules.read():
                        print("✅ Successfully loaded v4l2-compat module")
                    else:
                        print("❌ Failed to load v4l2-compat module")
    except Exception as e:
        print(f"❌ Error checking for v4l2-compat module: {e}")

def check_older_pi_camera():
    """Check older Raspberry Pi camera setup"""
    print("\nChecking bcm2835-v4l2 module (for older Raspberry Pi models):")
    try:
        # Check if module is loaded
        with open('/proc/modules', 'r') as modules:
            module_content = modules.read()
            if 'bcm2835_v4l2' in module_content:
                print("✅ bcm2835-v4l2 module is loaded")
            else:
                print("⚠️ bcm2835-v4l2 module is not loaded")
                print("Attempting to load the module...")
                os.system("sudo modprobe bcm2835-v4l2")
                time.sleep(1)
                
                # Check again
                with open('/proc/modules', 'r') as modules:
                    if 'bcm2835_v4l2' in modules.read():
                        print("✅ Successfully loaded bcm2835-v4l2 module")
                    else:
                        print("❌ Failed to load bcm2835-v4l2 module")
    except Exception as e:
        print(f"❌ Error checking for bcm2835-v4l2 module: {e}")

def test_camera_backends():
    """Test different camera backends and configurations"""
    print("\nTesting camera access with different backends:")
    
    # Check if we're on a Raspberry Pi 5
    is_pi, pi_model, is_pi5 = check_raspberry_pi()
    
    backends_to_try = []
    
    # Default backend
    backends_to_try.append(("Default", lambda i: cv2.VideoCapture(i)))
    
    # V4L2 backend
    backends_to_try.append(("V4L2", lambda i: cv2.VideoCapture(i, cv2.CAP_V4L2)))
    
    # GStreamer backend if available
    try:
        # Check if OpenCV has GStreamer support
        if 'GStreamer' in cv2.getBuildInformation():
            backends_to_try.append(("GStreamer", lambda i: cv2.VideoCapture(i, cv2.CAP_GSTREAMER)))
            
            # For Pi 5, add a specific GStreamer pipeline
            if is_pi5:
                gst_pipeline = "libcamerasrc ! video/x-raw, width=640, height=480 ! videoconvert ! appsink"
                backends_to_try.append(("GStreamer Pipeline", lambda _: cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)))
    except:
        print("Could not check for GStreamer support")
    
    # Try each backend
    results = []
    
    for name, create_cap in backends_to_try:
        print(f"\nTrying {name} backend:")
        
        # Try camera indices 0-3
        for i in range(4):
            try:
                print(f"  Testing camera index {i}...")
                cap = create_cap(i)
                
                if cap.isOpened():
                    print(f"  ✅ Successfully opened camera with {name} backend at index {i}")
                    
                    # Try to read a frame
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        h, w = frame.shape[:2]
                        print(f"  ✅ Successfully read frame with shape: {w}x{h}")
                        
                        # Save a test image
                        filename = f"camera_test_{name.lower().replace(' ', '_')}_{i}.jpg"
                        cv2.imwrite(filename, frame)
                        print(f"  ✅ Saved test image to {filename}")
                        
                        # Get camera properties
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
                        contrast = cap.get(cv2.CAP_PROP_CONTRAST)
                        print(f"  ℹ️ Camera properties - FPS: {fps}, Brightness: {brightness}, Contrast: {contrast}")
                        
                        results.append({
                            "backend": name,
                            "index": i,
                            "resolution": (w, h),
                            "fps": fps,
                            "success": True
                        })
                    else:
                        print(f"  ❌ Failed to read frame from camera")
                        results.append({
                            "backend": name,
                            "index": i,
                            "success": False,
                            "error": "Could not read frame"
                        })
                    
                    # Release the camera
                    cap.release()
                else:
                    print(f"  ❌ Failed to open camera")
                    results.append({
                        "backend": name,
                        "index": i,
                        "success": False,
                        "error": "Could not open camera"
                    })
                    cap.release()
            except Exception as e:
                print(f"  ❌ Error with {name} backend at index {i}: {e}")
                results.append({
                    "backend": name,
                    "index": i,
                    "success": False,
                    "error": str(e)
                })
    
    return results

def test_camera_configurations():
    """Test different camera configurations"""
    print("\nTesting different camera configurations:")
    
    # Check if we're on a Raspberry Pi
    is_pi, pi_model, is_pi5 = check_raspberry_pi()
    
    # Find a working camera setup first
    working_backend = None
    working_index = 0
    
    # Try V4L2 first
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if cap.isOpened() and cap.read()[0]:
            working_backend = cv2.CAP_V4L2
            working_index = 0
            cap.release()
    except:
        pass
    
    # If V4L2 didn't work, try default
    if working_backend is None:
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened() and cap.read()[0]:
                working_backend = None  # Default backend
                working_index = 0
                cap.release()
        except:
            pass
    
    # If still no working camera, try other indices
    if working_backend is None:
        for i in range(1, 4):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened() and cap.read()[0]:
                    working_backend = None  # Default backend
                    working_index = i
                    cap.release()
                    break
            except:
                pass
    
    if working_backend is None:
        print("❌ Could not find a working camera configuration")
        return
    
    print(f"✅ Found working camera: Backend={working_backend or 'Default'}, Index={working_index}")
    
    # Now test different resolutions
    resolutions = [
        (320, 240),
        (640, 480),
        (800, 600),
        (1280, 720)
    ]
    
    for width, height in resolutions:
        print(f"\nTesting resolution {width}x{height}:")
        
        try:
            # Create capture with the working backend
            if working_backend:
                cap = cv2.VideoCapture(working_index, working_backend)
            else:
                cap = cv2.VideoCapture(working_index)
            
            if not cap.isOpened():
                print(f"❌ Failed to open camera")
                continue
            
            # Set resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            # Read a frame
            ret, frame = cap.read()
            if ret and frame is not None:
                actual_h, actual_w = frame.shape[:2]
                print(f"✅ Got frame with resolution: {actual_w}x{actual_h}")
                
                # Save test image
                filename = f"camera_test_{actual_w}x{actual_h}.jpg"
                cv2.imwrite(filename, frame)
                print(f"✅ Saved test image to {filename}")
            else:
                print(f"❌ Failed to read frame at resolution {width}x{height}")
            
            cap.release()
        except Exception as e:
            print(f"❌ Error testing resolution {width}x{height}: {e}")
    
    # Test different FPS settings
    fps_values = [15, 30]
    
    for fps in fps_values:
        print(f"\nTesting FPS setting: {fps}")
        
        try:
            # Create capture with the working backend
            if working_backend:
                cap = cv2.VideoCapture(working_index, working_backend)
            else:
                cap = cv2.VideoCapture(working_index)
            
            if not cap.isOpened():
                print(f"❌ Failed to open camera")
                continue
            
            # Set FPS
            cap.set(cv2.CAP_PROP_FPS, fps)
            
            # Read a frame
            ret, frame = cap.read()
            if ret and frame is not None:
                actual_fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"✅ Set FPS to {fps}, actual FPS reported as: {actual_fps}")
            else:
                print(f"❌ Failed to read frame at FPS {fps}")
            
            cap.release()
        except Exception as e:
            print(f"❌ Error testing FPS {fps}: {e}")

def run_camera_test():
    """Run a comprehensive camera test"""
    print("=" * 50)
    print("RASPBERRY PI CAMERA TEST")
    print("=" * 50)
    
    # Check Raspberry Pi model
    is_pi, pi_model, is_pi5 = check_raspberry_pi()
    
    # Check for camera modules and drivers
    video_devices = check_camera_modules()
    
    if not video_devices:
        print("\n❌ No video devices found. Please check your camera connection.")
        print("If using Raspberry Pi Camera Module:")
        print("1. Make sure the camera is properly connected")
        print("2. Enable the camera in raspi-config")
        print("3. For Pi 5, ensure libcamera is installed")
        print("4. For older Pi models, ensure bcm2835-v4l2 module is loaded")
        return False
    
    # Test different camera backends
    backend_results = test_camera_backends()
    
    # Check if any backend was successful
    success = any(result["success"] for result in backend_results)
    
    if success:
        print("\n✅ At least one camera configuration works!")
        
        # Test different camera configurations
        test_camera_configurations()
        
        # Print summary of working configurations
        print("\n" + "=" * 50)
        print("CAMERA TEST SUMMARY")
        print("=" * 50)
        
        working_configs = [r for r in backend_results if r["success"]]
        
        if working_configs:
            print(f"Working camera configurations:")
            for config in working_configs:
                if "resolution" in config:
                    print(f"- Backend: {config['backend']}, Index: {config['index']}, Resolution: {config['resolution'][0]}x{config['resolution'][1]}, FPS: {config['fps']}")
                else:
                    print(f"- Backend: {config['backend']}, Index: {config['index']}")
            
            # Recommend the best configuration
            print("\nRECOMMENDED CONFIGURATION:")
            # Prefer V4L2 backend for better compatibility
            v4l2_configs = [c for c in working_configs if c["backend"] == "V4L2"]
            if v4l2_configs:
                best_config = v4l2_configs[0]
            else:
                best_config = working_configs[0]
            
            print(f"Backend: {best_config['backend']}, Camera Index: {best_config['index']}")
            if is_pi5:
                print("For Raspberry Pi 5, add these lines to your code:")
                print("self.emotion_cap = cv2.VideoCapture(0, cv2.CAP_V4L2)")
            else:
                print("For your Raspberry Pi model, add these lines to your code:")
                print("os.system('sudo modprobe bcm2835-v4l2')")
                print("time.sleep(1)")
                print("self.emotion_cap = cv2.VideoCapture(0)")
        
        return True
    else:
        print("\n❌ All camera configurations failed.")
        print("Please check your camera connection and drivers.")
        return False

def test_live_preview():
    """Run a live preview of the camera if possible"""
    print("\nStarting live camera preview (press 'q' to exit):")
    
    # Try to find a working camera configuration
    backends_to_try = [
        (None, 0),  # Default backend, index 0
        (cv2.CAP_V4L2, 0),  # V4L2 backend, index 0
        (None, 1),  # Default backend, index 1
        (cv2.CAP_V4L2, 1),  # V4L2 backend, index 1
    ]
    
    cap = None
    for backend, index in backends_to_try:
        try:
            if backend is None:
                cap = cv2.VideoCapture(index)
            else:
                cap = cv2.VideoCapture(index, backend)
            
            if cap.isOpened() and cap.read()[0]:
                print(f"✅ Using {'default' if backend is None else 'V4L2'} backend with camera index {index}")
                break
        except:
            if cap:
                cap.release()
            cap = None
    
    if not cap or not cap.isOpened():
        print("❌ Could not open any camera for live preview")
        return False
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            # Add timestamp to the frame
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(
                frame,
                timestamp,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
            
            # Add resolution info
            h, w = frame.shape[:2]
            cv2.putText(
                frame,
                f"Resolution: {w}x{h}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
            
            # Show the frame
            cv2.imshow("Camera Preview", frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"❌ Error in live preview: {e}")
    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()
    
    return True

if __name__ == "__main__":
    print("Raspberry Pi Camera Test Script")
    print("This script will test your camera setup and provide diagnostic information")
    
    success = run_camera_test()
    
    if success:
        print("\nCamera test completed successfully!")
        
        # Ask if user wants to see a live preview
        try:
            response = input("\nDo you want to see a live camera preview? (y/n): ")
            if response.lower() in ['y', 'yes']:
                test_live_preview()
        except:
            pass
    else:
        print("\nCamera test failed. Please check your camera setup.")
    
    print("\nTest complete. Exiting.") 