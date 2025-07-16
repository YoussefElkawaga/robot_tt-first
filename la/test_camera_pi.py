#!/usr/bin/env python3
"""
Raspberry Pi Camera Test Script
Tests camera functionality and emotion detection
"""

import cv2
import numpy as np
import time
import sys
import os

def detect_raspberry_pi():
    """Detect if running on Raspberry Pi"""
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
            return 'BCM' in cpuinfo or 'ARM' in cpuinfo
    except:
        return False

def test_camera_basic():
    """Test basic camera functionality"""
    print("🎥 Testing basic camera functionality...")
    
    # Try different camera indices
    camera_indices = [0, 1, 2]
    working_cameras = []
    
    for idx in camera_indices:
        print(f"  Testing camera index {idx}...")
        try:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"  ✅ Camera {idx} working! Resolution: {frame.shape}")
                    working_cameras.append(idx)
                else:
                    print(f"  ❌ Camera {idx} opened but cannot read frames")
                cap.release()
            else:
                print(f"  ❌ Cannot open camera {idx}")
        except Exception as e:
            print(f"  ❌ Camera {idx} error: {e}")
    
    return working_cameras

def test_camera_settings(camera_idx=0):
    """Test camera with different settings"""
    print(f"🔧 Testing camera {camera_idx} with different settings...")
    
    try:
        cap = cv2.VideoCapture(camera_idx)
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return False
        
        # Test different resolutions
        resolutions = [
            (320, 240),   # Pi optimized
            (640, 480),   # Standard
            (160, 120),   # Very low for processing
        ]
        
        for width, height in resolutions:
            print(f"  Testing resolution {width}x{height}...")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            ret, frame = cap.read()
            if ret:
                actual_shape = frame.shape
                print(f"  ✅ Got frame: {actual_shape}")
            else:
                print(f"  ❌ Failed to get frame at {width}x{height}")
        
        # Test FPS settings
        fps_values = [15, 30]
        for fps in fps_values:
            print(f"  Testing FPS: {fps}")
            cap.set(cv2.CAP_PROP_FPS, fps)
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"  Set FPS: {fps}, Actual FPS: {actual_fps}")
        
        cap.release()
        return True
        
    except Exception as e:
        print(f"❌ Camera settings test failed: {e}")
        return False

def test_emotion_detection():
    """Test emotion detection with FER"""
    print("🎭 Testing emotion detection...")
    
    try:
        from fer import FER
        print("  ✅ FER library imported successfully")
        
        # Initialize FER
        is_pi = detect_raspberry_pi()
        if is_pi:
            detector = FER(mtcnn=False)  # Lightweight for Pi
            print("  ✅ FER initialized (Pi mode - no MTCNN)")
        else:
            detector = FER(mtcnn=True)
            print("  ✅ FER initialized (full mode with MTCNN)")
        
        # Test with camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("  ❌ Cannot open camera for emotion detection test")
            return False
        
        # Set Pi-optimized settings
        if is_pi:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            cap.set(cv2.CAP_PROP_FPS, 15)
        
        print("  📸 Capturing frames for emotion detection test...")
        emotions_detected = []
        
        for i in range(10):  # Test 10 frames
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Resize for faster processing on Pi
            if is_pi:
                frame = cv2.resize(frame, (160, 120))
            
            try:
                result = detector.detect_emotions(frame)
                if result:
                    emotions = result[0]["emotions"]
                    dominant = max(emotions, key=emotions.get)
                    score = emotions[dominant]
                    emotions_detected.append((dominant, score))
                    print(f"    Frame {i+1}: {dominant} ({score:.2f})")
                else:
                    print(f"    Frame {i+1}: No face detected")
            except Exception as e:
                print(f"    Frame {i+1}: Error - {e}")
            
            time.sleep(0.1)
        
        cap.release()
        
        if emotions_detected:
            print(f"  ✅ Emotion detection working! Detected {len(emotions_detected)} emotions")
            return True
        else:
            print("  ⚠️ No emotions detected - may need better lighting or face positioning")
            return False
            
    except ImportError:
        print("  ❌ FER library not installed")
        print("  Install with: pip install fer")
        return False
    except Exception as e:
        print(f"  ❌ Emotion detection test failed: {e}")
        return False

def test_camera_performance():
    """Test camera performance (FPS measurement)"""
    print("⚡ Testing camera performance...")
    
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return False
        
        # Set Pi-optimized settings
        is_pi = detect_raspberry_pi()
        if is_pi:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            cap.set(cv2.CAP_PROP_FPS, 15)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Measure FPS
        frame_count = 0
        start_time = time.time()
        test_duration = 5  # seconds
        
        print(f"  Measuring FPS for {test_duration} seconds...")
        
        while time.time() - start_time < test_duration:
            ret, frame = cap.read()
            if ret:
                frame_count += 1
            else:
                print("  ⚠️ Failed to read frame")
        
        end_time = time.time()
        actual_duration = end_time - start_time
        fps = frame_count / actual_duration
        
        print(f"  ✅ Captured {frame_count} frames in {actual_duration:.2f} seconds")
        print(f"  ✅ Actual FPS: {fps:.2f}")
        
        cap.release()
        
        # Performance assessment
        if is_pi:
            if fps >= 10:
                print("  ✅ Good performance for Raspberry Pi")
            elif fps >= 5:
                print("  ⚠️ Acceptable performance for Raspberry Pi")
            else:
                print("  ❌ Poor performance - consider optimizations")
        else:
            if fps >= 20:
                print("  ✅ Good performance")
            elif fps >= 10:
                print("  ⚠️ Acceptable performance")
            else:
                print("  ❌ Poor performance")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🍓 Raspberry Pi Camera Test")
    print("=" * 30)
    
    is_pi = detect_raspberry_pi()
    if is_pi:
        print("✅ Running on Raspberry Pi")
    else:
        print("💻 Running on standard computer")
    
    print()
    
    # Test 1: Basic camera functionality
    working_cameras = test_camera_basic()
    if not working_cameras:
        print("❌ No working cameras found!")
        print("\n🔧 Troubleshooting tips:")
        print("• Check camera connection")
        print("• Enable camera interface: sudo raspi-config")
        print("• Check permissions: sudo usermod -a -G video $USER")
        print("• List devices: v4l2-ctl --list-devices")
        return
    
    print(f"✅ Found {len(working_cameras)} working camera(s): {working_cameras}")
    print()
    
    # Test 2: Camera settings
    test_camera_settings(working_cameras[0])
    print()
    
    # Test 3: Performance test
    test_camera_performance()
    print()
    
    # Test 4: Emotion detection
    test_emotion_detection()
    print()
    
    print("🎯 Test Summary:")
    print("===============")
    print(f"• Cameras found: {len(working_cameras)}")
    print("• Settings test: Completed")
    print("• Performance test: Completed")
    print("• Emotion detection test: Completed")
    
    if is_pi:
        print("\n🍓 Raspberry Pi Tips:")
        print("• Use lower resolutions for better performance")
        print("• Process every N frames to reduce CPU usage")
        print("• Consider USB camera for better quality")
        print("• Ensure good lighting for emotion detection")
    
    print("\n✅ Camera testing complete!")

if __name__ == "__main__":
    main()