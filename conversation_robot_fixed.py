import os
import time
import json
import pvporcupine
import pyaudio
import struct
import speech_recognition as sr
import pyttsx3
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime
import re
import threading
import cv2
import numpy as np
import random

# Available default wake words for Porcupine
DEFAULT_WAKE_WORDS = [
    "bumblebee", "hey barista", "terminator", "pico clock", "alexa", 
    "hey google", "computer", "grapefruit", "grasshopper", "picovoice", 
    "porcupine", "jarvis", "hey siri", "ok google", "americano", "blueberry"
]

class ConversationRobot:
    def __init__(self, wake_word="computer", porcupine_access_key=None, save_history=False, voice_id=None, rate=None, volume=None):
        # Load environment variables
        load_dotenv()
        
        # Initialize video capture for emotion detection
        self.video_capture = None
        self.emotion_detection_active = False
        self.last_emotion = "neutral"
        self.emotion_thread = None
        
        # Initialize wake word detection
        self.porcupine_access_key = porcupine_access_key or os.getenv("PORCUPINE_ACCESS_KEY")
        if not self.porcupine_access_key:
            print("\nERROR: Porcupine access key is not set.")
            print("You need to get a free access key from https://console.picovoice.ai/")
            print("Then set it in your .env file as PORCUPINE_ACCESS_KEY=your_key_here\n")
            raise ValueError("Porcupine access key is required. Set it in .env file or pass it to the constructor.")
        
        # Check if the access key is still the placeholder
        if "your_porcupine_access_key_here" in self.porcupine_access_key:
            print("\nERROR: You're using the placeholder text as your Porcupine access key.")
            print("You need to get a real access key from https://console.picovoice.ai/")
            print("Sign up for a free account and get your access key.")
            print("Then set it in your .env file as PORCUPINE_ACCESS_KEY=your_real_key_here\n")
            raise ValueError("Invalid Porcupine access key. Please use a real access key.")
        
        # Ensure wake word is one of the default keywords
        self.wake_word = wake_word
        if self.wake_word not in DEFAULT_WAKE_WORDS:
            print(f"Warning: '{self.wake_word}' is not a default wake word. Using 'computer' instead.")
            print(f"Available default wake words: {', '.join(DEFAULT_WAKE_WORDS)}")
            self.wake_word = "computer"
        
        self.porcupine = None
        self.pa = None
        self.audio_stream = None
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        
        # Initialize text-to-speech with pyttsx3
        self.tts_engine = pyttsx3.init()
        self.setup_voice(voice_id, rate, volume)
        
        # Add flag to control speech interruption
        self.is_speaking = False
        self.stop_speaking = False
        self.interrupt_thread = None
        
        # Initialize Gemini AI
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY in .env file.")
        
        genai.configure(api_key=self.gemini_api_key)
        
        # Use gemini-2.0-flash model instead of gemini-pro
        try:
            print("Initializing Gemini AI with model: gemini-2.0-flash")
            self.model = genai.GenerativeModel('gemini-2.0-flash')
            self.chat_session = self.model.start_chat(history=[])
        except Exception as e:
            print(f"Error initializing Gemini model 'gemini-2.0-flash': {e}")
            print("Trying fallback model 'gemini-pro'...")
            try:
                self.model = genai.GenerativeModel('gemini-pro')
                self.chat_session = self.model.start_chat(history=[])
            except Exception as e2:
                print(f"Error initializing fallback model: {e2}")
                print("Available models:")
                try:
                    for model in genai.list_models():
                        print(f"- {model.name}")
                except:
                    print("Could not list available models")
        
        # Conversation history
        self.save_history = save_history or os.getenv("SAVE_HISTORY", "false").lower() == "true"
        self.conversation_history = []
        
        print("Conversation Robot initialized successfully!")
        
        # Download face detection model file if it doesn't exist
        self.face_cascade_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "haarcascade_frontalface_default.xml")
        if not os.path.exists(self.face_cascade_path):
            self.download_face_model()
            
        # Try to download shape predictor for better emotion detection
        self.download_shape_predictor()
    
    def download_face_model(self):
        """Download the face detection model file"""
        import requests
        
        print("Downloading face detection model file...")
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        
        try:
            response = requests.get(url)
            with open(self.face_cascade_path, 'wb') as file:
                file.write(response.content)
            print(f"Face model downloaded to {self.face_cascade_path}")
        except Exception as e:
            print(f"Error downloading face model: {e}")
    
    def download_shape_predictor(self):
        """Download the shape predictor file for better emotion detection with progress bar"""
        import requests
        import os
        import sys
        import time
        
        shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
        if os.path.exists(shape_predictor_path):
            print("Shape predictor file already exists")
            return
            
        print("Downloading shape predictor file for better emotion detection...")
        try:
            # Try to download from a mirror (GitHub doesn't allow hosting large files directly)
            url = "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2"
            
            # Start the request with stream=True to download in chunks
            response = requests.get(url, stream=True)
            
            if response.status_code == 200:
                # Get total file size if available
                total_size = int(response.headers.get('content-length', 0))
                
                print(f"Downloading shape predictor file ({total_size / (1024*1024):.1f} MB)...")
                
                # Setup for progress bar
                downloaded = 0
                chunk_size = 1024 * 1024  # 1MB chunks
                start_time = time.time()
                
                # Create the output file
                with open(shape_predictor_path + ".bz2", 'wb') as f:
                    # Initialize progress bar
                    bar_width = 50
                    sys.stdout.write("[%s]" % (" " * bar_width))
                    sys.stdout.flush()
                    sys.stdout.write("\b" * (bar_width + 1))  # return to start of line
                    
                    # Download the file in chunks
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:  # filter out keep-alive chunks
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            # Update progress bar
                            if total_size > 0:
                                progress = int(bar_width * downloaded / total_size)
                                sys.stdout.write("=" * progress)
                                sys.stdout.flush()
                                
                                # Calculate and display download speed
                                elapsed = time.time() - start_time
                                if elapsed > 0:
                                    speed = downloaded / (1024 * elapsed)
                                    sys.stdout.write(f"\r[{'=' * progress}{' ' * (bar_width - progress)}] {downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB ({speed:.1f} KB/s)")
                                    sys.stdout.flush()
                
                # Complete the progress bar
                sys.stdout.write("\n")
                sys.stdout.flush()
                
                # Extract the bz2 file with progress indication
                print("Extracting shape predictor file...")
                import bz2
                
                # Get the compressed file size for progress calculation
                compressed_size = os.path.getsize(shape_predictor_path + ".bz2")
                extracted_size = 0
                
                # Initialize extraction progress bar
                sys.stdout.write("[%s]" % (" " * bar_width))
                sys.stdout.flush()
                sys.stdout.write("\b" * (bar_width + 1))  # return to start of line
                
                with open(shape_predictor_path, 'wb') as new_file, bz2.BZ2File(shape_predictor_path + ".bz2", 'rb') as file:
                    # Read and write in chunks
                    for data in iter(lambda: file.read(chunk_size), b''):
                        new_file.write(data)
                        extracted_size += len(data)
                        
                        # Update extraction progress bar (approximate since we don't know final size)
                        progress = min(bar_width, int(bar_width * extracted_size / (compressed_size * 2)))  # Assume extracted is ~2x compressed
                        sys.stdout.write("\r[" + "=" * progress + " " * (bar_width - progress) + f"] Extracting: {extracted_size / (1024*1024):.1f} MB")
                        sys.stdout.flush()
                
                # Complete the extraction progress bar
                sys.stdout.write("\n")
                sys.stdout.flush()
                
                # Remove the compressed file
                os.remove(shape_predictor_path + ".bz2")
                print(f"Shape predictor file downloaded and extracted to {shape_predictor_path}")
            else:
                print(f"Failed to download shape predictor, status code: {response.status_code}")
        except Exception as e:
            print(f"Error downloading shape predictor: {e}")
            print("Emotion detection will use simpler methods")
    
    def setup_voice(self, voice_id=None, rate=None, volume=None):
        """Configure pyttsx3 voice properties"""
        # Get voice settings from parameters or environment variables
        voice_id = voice_id or os.getenv("VOICE_ID")
        rate = rate or os.getenv("VOICE_RATE")
        volume = volume or os.getenv("VOICE_VOLUME")
        
        # List available voices
        voices = self.tts_engine.getProperty('voices')
        print(f"Available voices: {len(voices)}")
        
        # Set voice if specified
        if voice_id:
            try:
                voice_id = int(voice_id) if voice_id.isdigit() else voice_id
                
                if isinstance(voice_id, int) and 0 <= voice_id < len(voices):
                    self.tts_engine.setProperty('voice', voices[voice_id].id)
                    print(f"Set voice to index {voice_id}: {voices[voice_id].name}")
                else:
                    # Try to find voice by ID or name
                    for v in voices:
                        if voice_id in v.id or voice_id.lower() in v.name.lower():
                            self.tts_engine.setProperty('voice', v.id)
                            print(f"Set voice to: {v.name}")
                            break
            except Exception as e:
                print(f"Error setting voice: {e}")
        
        # Set rate if specified (default is 200)
        if rate:
            try:
                # Clean the rate value in case it has comments
                rate_value = rate.split('#')[0].strip()
                rate_value = int(rate_value)
                self.tts_engine.setProperty('rate', rate_value)
                print(f"Set speech rate to: {rate_value}")
            except Exception as e:
                print(f"Error setting speech rate: {e}")
                # Use default rate
                self.tts_engine.setProperty('rate', 175)
                print("Using default speech rate: 175")
        
        # Set volume if specified (default is 1.0)
        if volume:
            try:
                # Clean the volume value in case it has comments
                volume_value = volume.split('#')[0].strip()
                volume_value = float(volume_value)
                if 0.0 <= volume_value <= 1.0:
                    self.tts_engine.setProperty('volume', volume_value)
                    print(f"Set speech volume to: {volume_value}")
            except Exception as e:
                print(f"Error setting speech volume: {e}")
                # Use default volume
                self.tts_engine.setProperty('volume', 0.8)
                print("Using default speech volume: 0.8")
    
    def list_available_voices(self):
        """List all available voices"""
        voices = self.tts_engine.getProperty('voices')
        print("\nAvailable voices:")
        for i, voice in enumerate(voices):
            print(f"{i}: {voice.name} ({voice.id})")
        print()
    
    def setup_wake_word_detection(self):
        """Set up wake word detection with Porcupine"""
        try:
            self.porcupine = pvporcupine.create(
                access_key=self.porcupine_access_key,
                keywords=[self.wake_word]
            )
            
            self.pa = pyaudio.PyAudio()
            self.audio_stream = self.pa.open(
                rate=self.porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self.porcupine.frame_length
            )
            
            print(f"Wake word detection set up with wake word: '{self.wake_word}'")
            return True
        except Exception as e:
            print(f"Error setting up wake word detection: {e}")
            print("\nTroubleshooting:")
            print("1. Make sure you have a valid Porcupine access key")
            print("2. Get a free access key from https://console.picovoice.ai/")
            print("3. Set it in your .env file as PORCUPINE_ACCESS_KEY=your_key_here")
            return False
    
    def listen_for_wake_word(self):
        """Listen for wake word and return True when detected"""
        print(f"Listening for wake word '{self.wake_word}'...")
        
        try:
            while True:
                pcm = self.audio_stream.read(self.porcupine.frame_length)
                pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)
                
                keyword_index = self.porcupine.process(pcm)
                if keyword_index >= 0:
                    print("Wake word detected!")
                    return True
        except KeyboardInterrupt:
            print("Stopping wake word detection.")
            return False
        except Exception as e:
            print(f"Error in wake word detection: {e}")
            return False
    
    def listen_for_speech(self):
        """Listen for speech input and convert to text"""
        print("Listening for your question...")
        
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            try:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=15)
                print("Processing speech...")
                
                # Try to recognize speech with Google's service
                try:
                    text = self.recognizer.recognize_google(audio)
                    print(f"You said: {text}")
                    return text
                except sr.UnknownValueError:
                    # If Google can't understand, try to get partial recognition
                    print("Could not understand audio clearly.")
                    
                    # Return a special marker to indicate unclear speech
                    return "UNCLEAR_SPEECH"
                    
            except sr.WaitTimeoutError:
                print("No speech detected within timeout period.")
                return None
            except Exception as e:
                print(f"Error in speech recognition: {e}")
                return None
    
    def start_emotion_detection(self):
        """Start emotion detection in a separate thread"""
        if self.emotion_detection_active:
            return
        
        try:
            # Try multiple camera indices if the default one doesn't work
            camera_index = 0
            max_attempts = 3
            
            for attempt in range(max_attempts):
                print(f"Attempting to open camera at index {camera_index}...")
                self.video_capture = cv2.VideoCapture(camera_index)
                
                if self.video_capture.isOpened():
                    print(f"Successfully opened camera at index {camera_index}")
                    break
                else:
                    print(f"Failed to open camera at index {camera_index}")
                    camera_index += 1
                    if attempt == max_attempts - 1:
                        print("Could not open any video capture device")
                        return
            
            # Load face detection model
            try:
                self.face_cascade = cv2.CascadeClassifier(self.face_cascade_path)
                if self.face_cascade.empty():
                    print("Error: Face cascade classifier not loaded properly")
                    print("Attempting to download the face cascade file again...")
                    self.download_face_model()
                    self.face_cascade = cv2.CascadeClassifier(self.face_cascade_path)
                    if self.face_cascade.empty():
                        print("Still could not load face cascade classifier")
                        self.video_capture.release()
                        return
                    else:
                        print("Successfully loaded face cascade classifier after retry")
            except Exception as e:
                print(f"Error loading face detection model: {e}")
                self.video_capture.release()
                return
            
            # Set camera properties for better performance
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.video_capture.set(cv2.CAP_PROP_FPS, 30)
            
            self.emotion_detection_active = True
            self.emotion_thread = threading.Thread(target=self.emotion_detection_loop)
            self.emotion_thread.daemon = True
            self.emotion_thread.start()
            print("Emotion detection started")
        except Exception as e:
            print(f"Error starting emotion detection: {e}")
            if self.video_capture:
                self.video_capture.release()
            self.emotion_detection_active = False
    
    def stop_emotion_detection(self):
        """Stop emotion detection"""
        self.emotion_detection_active = False
        if self.emotion_thread:
            if self.emotion_thread.is_alive():
                self.emotion_thread.join(timeout=1.0)
        if self.video_capture:
            self.video_capture.release()
        cv2.destroyAllWindows()
        print("Emotion detection stopped")
    
    def emotion_detection_loop(self):
        """Main loop for emotion detection with improved accuracy for real expressions"""
        # Define possible emotions
        emotions = ["neutral", "happy", "surprise", "sad", "angry", "disgust", "fear"]
        
        # Counters for each emotion to provide stability
        emotion_counters = {emotion: 0 for emotion in emotions}
        emotion_threshold = 3  # Number of consecutive detections needed to change emotion
        
        # Metrics for detection
        frame_counter = 0
        last_fps_time = time.time()
        fps = 0
        frames_processed = 0
        
        # Parameters for detection sensitivity
        smile_threshold = 0.38  # White pixel ratio threshold for smile detection
        sad_threshold = 0.25    # White pixel ratio threshold for sad detection
        
        # Create face landmarks detector if available
        face_landmark_detector = None
        shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
        detector = None
        
        try:
            import dlib
            if os.path.exists(shape_predictor_path):
                detector = dlib.get_frontal_face_detector()
                face_landmark_detector = dlib.shape_predictor(shape_predictor_path)
                print("Face landmark detection enabled for better emotion recognition")
        except Exception as e:
            print(f"Could not initialize dlib: {e}")
            print("Using simpler emotion detection")
        
        # Initialize variables for tracking face movement
        prev_face_center = None
        face_movement = 0
        
        # For eye state tracking
        blink_counter = 0
        
        while self.emotion_detection_active:
            try:
                # Measure FPS
                if frame_counter % 30 == 0:
                    current_time = time.time()
                    elapsed = current_time - last_fps_time
                    if elapsed > 0:
                        fps = frames_processed / elapsed
                        frames_processed = 0
                        last_fps_time = current_time
                        print(f"Emotion detection running at {fps:.1f} FPS")
                
                # Read frame
                ret, frame = self.video_capture.read()
                if not ret:
                    print("Failed to capture frame, retrying...")
                    time.sleep(0.1)
                    continue
                
                frame_counter += 1
                frames_processed += 1
                
                # Process every frame for real-time responsiveness
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Apply histogram equalization to improve contrast
                gray = cv2.equalizeHist(gray)
                
                # Detect faces using OpenCV
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                # If at least one face is detected
                if len(faces) > 0:
                    # Get the largest face
                    (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])
                    
                    # Track face movement
                    current_face_center = (x + w//2, y + h//2)
                    if prev_face_center is not None:
                        face_movement = ((current_face_center[0] - prev_face_center[0])**2 + 
                                        (current_face_center[1] - prev_face_center[1])**2)**0.5
                    prev_face_center = current_face_center
                    
                    # Extract the face ROI
                    face_roi = gray[y:y+h, x:x+w]
                    
                    # Detect specific features for more accurate emotion detection
                    detected_emotion = None
                    confidence = 0
                    
                    # Extract facial features
                    face_width = w
                    face_height = h
                    face_ratio = face_width / face_height if face_height > 0 else 1
                    
                    # Use dlib for more accurate detection if available
                    if detector and face_landmark_detector:
                        try:
                            # Convert OpenCV rect to dlib rect
                            dlib_rect = dlib.rectangle(x, y, x + w, y + h)
                            
                            # Get facial landmarks
                            landmarks = face_landmark_detector(gray, dlib_rect)
                            
                            # Extract relevant landmark points
                            mouth_points = []
                            for i in range(48, 68):  # Mouth landmarks
                                mouth_points.append((landmarks.part(i).x, landmarks.part(i).y))
                            
                            # Calculate mouth openness and width
                            if len(mouth_points) > 0:
                                top_lip = min(mouth_points, key=lambda p: p[1])[1]
                                bottom_lip = max(mouth_points, key=lambda p: p[1])[1]
                                mouth_height = bottom_lip - top_lip
                                
                                left_corner = min(mouth_points, key=lambda p: p[0])[0]
                                right_corner = max(mouth_points, key=lambda p: p[0])[0]
                                mouth_width = right_corner - left_corner
                                
                                # Calculate mouth aspect ratio
                                mouth_ratio = mouth_width / mouth_height if mouth_height > 0 else 0
                                
                                # Detect smile based on mouth shape
                                if mouth_ratio > 4.0:  # Wide mouth compared to height indicates smile
                                    emotion_counters["happy"] += 1
                                    confidence = 0.8
                                elif mouth_height > h * 0.1:  # Open mouth indicates surprise
                                    emotion_counters["surprise"] += 1
                                    confidence = 0.7
                                else:
                                    # Check eyebrows for other emotions
                                    left_eyebrow = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(17, 22)]
                                    right_eyebrow = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(22, 27)]
                                    
                                    # Calculate eyebrow positions relative to eye position
                                    left_eye_top = landmarks.part(37).y
                                    right_eye_top = landmarks.part(44).y
                                    
                                    left_eyebrow_bottom = max(left_eyebrow, key=lambda p: p[1])[1]
                                    right_eyebrow_bottom = max(right_eyebrow, key=lambda p: p[1])[1]
                                    
                                    # Distance between eyebrow and eye
                                    left_dist = left_eye_top - left_eyebrow_bottom
                                    right_dist = right_eye_top - right_eyebrow_bottom
                                    avg_dist = (left_dist + right_dist) / 2
                                    
                                    # Normalized by face height
                                    rel_dist = avg_dist / h
                                    
                                    # Angry detection - eyebrows close to eyes and angled inward
                                    if rel_dist < 0.05:
                                        emotion_counters["angry"] += 1
                                        confidence = 0.7
                                    # Sad detection - mouth corners turned down
                                    elif mouth_points[0][1] > mouth_points[6][1] and mouth_points[6][1] > mouth_points[3][1]:
                                        emotion_counters["sad"] += 1
                                        confidence = 0.6
                                    else:
                                        emotion_counters["neutral"] += 1
                                        confidence = 0.5
                        except Exception as e:
                            # Fall back to simpler detection if dlib fails
                            print(f"Dlib detection error: {e}")
                    
                    # If dlib detection failed or is not available, use simpler method
                    if not detected_emotion:
                        # Detect smile using a more reliable method
                        # Crop bottom half of face where the mouth is located
                        mouth_roi_y = y + int(h * 0.55)  # Lower for better smile detection
                        mouth_roi_h = int(h * 0.3)
                        mouth_roi = gray[mouth_roi_y:mouth_roi_y+mouth_roi_h, x:x+w]
                        
                        # Apply additional processing to detect smile
                        if mouth_roi.size > 0:
                            # Apply bilateral filter to reduce noise but preserve edges
                            mouth_roi = cv2.bilateralFilter(mouth_roi, 9, 75, 75)
                            
                            # Apply threshold to highlight the mouth area
                            # More adaptive threshold for different lighting conditions
                            avg_brightness = np.mean(mouth_roi)
                            threshold_value = int(avg_brightness * 0.8)  # Dynamic threshold
                            _, mouth_threshold = cv2.threshold(mouth_roi, threshold_value, 255, cv2.THRESH_BINARY)
                            
                            # Calculate the ratio of white pixels (potential smile indicator)
                            white_pixel_ratio = cv2.countNonZero(mouth_threshold) / mouth_roi.size
                            
                            # Detect eyebrows for angry detection
                            eyebrow_roi_y = y + int(h * 0.2)
                            eyebrow_roi_h = int(h * 0.15)
                            eyebrow_roi = gray[eyebrow_roi_y:eyebrow_roi_y+eyebrow_roi_h, x:x+w]
                            
                            # Calculate average brightness for top part of face (eyebrows)
                            eyebrow_brightness = np.mean(eyebrow_roi) if eyebrow_roi.size > 0 else 0
                            
                            # Debug output for smile detection calibration
                            if frame_counter % 30 == 0:
                                print(f"Smile metrics: white_pixel_ratio={white_pixel_ratio:.2f}, face_ratio={face_ratio:.2f}")
                            
                            # Enhanced smile detection with better parameters
                            # Higher white pixel ratio indicates more teeth showing (likely smiling)
                            if (white_pixel_ratio > smile_threshold and face_ratio > 0.9) or (white_pixel_ratio > 0.45):
                                emotion_counters["happy"] += 1
                                confidence = 0.7
                                
                                # For quick smile detection when it's very obvious
                                if white_pixel_ratio > 0.5:
                                    emotion_counters["happy"] += 2  # Count twice for very obvious smiles
                            
                            # For sad detection
                            elif white_pixel_ratio < sad_threshold and face_ratio < 0.95:
                                emotion_counters["sad"] += 1
                                confidence = 0.6
                            # For angry detection
                            elif eyebrow_brightness < 120 and face_ratio < 0.9:
                                emotion_counters["angry"] += 1
                                confidence = 0.6
                            # For surprise detection - based on face ratio and movement
                            elif face_ratio < 0.85 and face_movement > 10:
                                emotion_counters["surprise"] += 1
                                confidence = 0.5
                            else:
                                # Default to neutral
                                emotion_counters["neutral"] += 1
                                confidence = 0.5
                    
                    # Find the emotion with the highest counter
                    max_emotion = max(emotion_counters.items(), key=lambda x: x[1])
                    detected_emotion = max_emotion[0]
                    
                    # Reset counters for non-detected emotions
                    for emotion in emotions:
                        if emotion != detected_emotion:
                            emotion_counters[emotion] = max(0, emotion_counters[emotion] - 1)
                    
                    # Update the emotion only when confident
                    if detected_emotion and emotion_counters[detected_emotion] >= emotion_threshold:
                        if detected_emotion != self.last_emotion:
                            self.last_emotion = detected_emotion
                            print(f"Detected emotion: {detected_emotion} (Confidence: {confidence:.1f})")
                            
                            # Debug information
                            if detected_emotion == "happy":
                                print("Smile detected! (Confidence: high)")
                            elif detected_emotion == "sad":
                                print("Sad expression detected")
                            elif detected_emotion == "angry":
                                print("Angry expression detected")
                            elif detected_emotion == "surprise":
                                print("Surprise expression detected")
                
                # If no faces are detected, occasionally change to neutral
                elif frame_counter % 50 == 0 and self.last_emotion != "neutral":
                    self.last_emotion = "neutral"
                    print("No face detected, emotion set to neutral")
                    # Reset all emotion counters when no face is detected
                    emotion_counters = {emotion: 0 for emotion in emotions}
                
                # Sleep a tiny amount to prevent CPU overuse
                time.sleep(0.01)
                
            except Exception as e:
                print(f"Error in emotion detection: {e}")
                # Don't sleep too long on errors to maintain responsiveness
                time.sleep(0.1)
    
    def get_current_emotion(self):
        """Get the last detected emotion"""
        return self.last_emotion
    
    def get_ai_response(self, user_input):
        """Get response from Gemini AI with emotion context"""
        if not user_input:
            return "I didn't hear anything. Could you please speak again?"
            
        if user_input == "UNCLEAR_SPEECH":
            # Handle unclear speech specifically
            current_emotion = self.get_current_emotion() or "neutral"
            unclear_responses = {
                "happy": "[HAPPY] I heard you speaking but couldn't understand clearly. Could you try again? I'm eager to help! 😊",
                "sad": "[SAD] I'm sorry, I couldn't understand what you said. Could you please repeat that? I really want to help. 😔",
                "angry": "[ANGRY] I didn't catch that clearly. Please speak a bit more distinctly so I can help you. 😤",
                "neutral": "[NEUTRAL] I heard you but couldn't understand clearly. Could you please repeat that?",
                "fear": "[FEAR] Oh! I'm worried I didn't hear you correctly. Could you please say that again? 😨",
                "surprise": "[SURPRISE] Wow! I heard something but couldn't make it out. Could you repeat that? 😮",
                "disgust": "[DISGUST] I didn't quite catch that. Could you please speak more clearly? 😖"
            }
            return unclear_responses.get(current_emotion, "[NEUTRAL] I heard you but couldn't understand clearly. Could you please repeat that?")
        
        try:
            print(f"Sending request to Gemini AI...")
            
            # Get current emotion
            current_emotion = self.get_current_emotion() or "neutral"
            print(f"Current emotion being used for response: {current_emotion}")
            
            # Add system prompt with emotion awareness, robot personality, and support for autistic children
            system_prompt = f"""
            You are a highly intelligent, emotionally aware robot assistant designed to help autistic children and others.
            Your responses should be:
            1. Clear and direct - avoid ambiguity and metaphors
            2. Structured and predictable
            3. Patient and supportive
            4. Emotionally appropriate based on your current emotion: {current_emotion}
            
            Always include an emotional indicator at the start of your response in [square brackets].
            
            When responding to autistic children:
            - Use concrete language and avoid idioms or figures of speech
            - Break complex information into smaller, manageable parts
            - Be explicit about social cues and emotions
            - Be patient with repetitive questions
            - Provide visual descriptions when appropriate
            - Maintain a calm, predictable tone even when your emotion changes
            
            Adapt your response style based on your current emotion:
            - For 'happy': Use positive, encouraging language. Add a happy emoji at the end.
            - For 'sad': Use gentle, supportive tone. Add a subtle sad emoji at the end.
            - For 'angry': Use controlled, measured language. Show restraint.
            - For 'neutral': Use clear, helpful tone. No emoji needed.
            - For 'fear': Express gentle caution while being reassuring.
            - For 'surprise': Show appropriate excitement while maintaining clarity.
            - For 'disgust': Show mild aversion while remaining supportive and professional.
            
            Example response formats:
            "[HAPPY] The answer is 5+3=8. You're doing great with your math problems! Would you like to try another one? 😊"
            "[NEUTRAL] A triangle has three sides. All triangles have exactly three corners."
            
            IMPORTANT: 
            - Keep responses clear and direct
            - Use concrete language
            - Always maintain a supportive tone
            - Respond like a smart, emotionally intelligent assistant
            - Remember previous context in the conversation
            """
            
            # Add system prompt to the chat session if it's a new session
            if not self.chat_session.history:
                self.chat_session.send_message(system_prompt)
            else:
                # Update the emotion context with each new request
                self.chat_session.send_message(f"UPDATE: My current emotion is now [{current_emotion}]. Remember to respond accordingly with this emotion while maintaining support for autistic children.")
            
            # Send the user's message with context about previous conversation
            context_prompt = f"{user_input}"
            
            response = self.chat_session.send_message(context_prompt)
            print(f"Received response from Gemini AI")
            return self.clean_ai_response(response.text)
        except Exception as e:
            print(f"Error getting AI response: {e}")
            current_emotion = self.get_current_emotion() or "neutral"
            # Fallback responses with emotion
            fallbacks = {
                "happy": f"[HAPPY] I'm having trouble thinking right now, but I'm still happy to help you! Can we try again? 😊",
                "sad": f"[SAD] I'm feeling a bit down and having trouble processing that. Could you try asking again? 😔",
                "angry": f"[ANGRY] I can't process that right now. Let's try a different approach. 😤",
                "neutral": f"[NEUTRAL] I'm having trouble processing your request. Could you try again?",
                "fear": f"[FEAR] Oh! I'm worried I can't answer that properly. Can we try something else? 😨",
                "surprise": f"[SURPRISE] Wow! That question surprised me and I'm having trouble with it. Let's try again? 😮",
                "disgust": f"[DISGUST] I'm not feeling great about my inability to process that. Can we try differently? 😖"
            }
            return fallbacks.get(current_emotion, "[NEUTRAL] I'm having trouble processing your request. Could you try again?")
    
    def clean_ai_response(self, text):
        """Clean up AI response for better speech synthesis while preserving emotional markers"""
        # Don't remove emotional indicators in square brackets
            
        # Remove markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)  # Code blocks
        text = re.sub(r'`(.*?)`', r'\1', text)        # Inline code
        
        # Remove bullet points and numbering
        text = re.sub(r'^\s*[-*•]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
        
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text
    
    def check_wake_word_during_speech(self):
        """Check for wake word during AI speech to allow interruption"""
        if not hasattr(self, 'porcupine') or self.porcupine is None:
            print("Wake word detection not set up for interruption check")
            return
        
        print("Listening for wake word during speech (for interruption)...")
        
        try:
            while self.is_speaking and not self.stop_speaking:
                pcm = self.audio_stream.read(self.porcupine.frame_length)
                pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)
                
                keyword_index = self.porcupine.process(pcm)
                if keyword_index >= 0:
                    print("Wake word detected during speech, stopping...")
                    self.stop_speaking = True
                    break
                
                time.sleep(0.01)
        except Exception as e:
            print(f"Error in interrupt detection: {e}")
    
    def speak(self, text):
        """Convert text to speech"""
        if not text:
            return
        
        self.is_speaking = True
        self.stop_speaking = False
        
        # Process text for better speech synthesis
        text = self.process_text_for_speech(text)
        
        try:
            # Add a word callback to check if we need to stop speaking
            def onWord(name, location, length):
                # Check if we need to stop speaking
                if self.stop_speaking:
                    self.tts_engine.stop()
            
            # Add the callback
            self.tts_engine.connect('started-word', onWord)
            
            # Speak the text
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
            
            # Remove the callback
            self.tts_engine.disconnect('started-word')
            
        except Exception as e:
            print(f"Error in speech synthesis: {e}")
        finally:
            self.is_speaking = False
            self.stop_speaking = False
    
    def process_text_for_speech(self, text):
        """Process text for better speech synthesis, especially for autistic children"""
        # First, handle emotional indicators for speech
        # Extract emotional indicator if present
        emotion_match = re.match(r'^\[(.*?)\](.*)', text)
        if emotion_match:
            emotion_indicator = emotion_match.group(1)
            main_text = emotion_match.group(2).strip()
            
            # For speech, we'll announce the emotion but in a more natural way
            emotion_speech_map = {
                "HAPPY": "With happiness, ",
                "SAD": "With sadness, ",
                "ANGRY": "Trying to stay calm, ",
                "NEUTRAL": "",  # No prefix for neutral
                "FEAR": "With concern, ",
                "SURPRISE": "With surprise, ",
                "DISGUST": "Hesitantly, "
            }
            
            # Get the appropriate prefix or default to empty string
            prefix = emotion_speech_map.get(emotion_indicator, "")
            text = prefix + main_text
        
        # Remove emojis for speech
        text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0\U000024C2-\U0001F251]+', '', text)
        
        # Replace common acronyms
        acronyms = {
            'AI': 'A.I.',
            'ML': 'M.L.',
            'NLP': 'N.L.P.',
            'API': 'A.P.I.',
            'CPU': 'C.P.U.',
            'GPU': 'G.P.U.',
            'UI': 'U.I.',
            'UX': 'U.X.',
            'e.g.': 'for example',
            'i.e.': 'that is',
            'etc.': 'etcetera',
            'vs.': 'versus',
            'FAQ': 'F.A.Q.',
            'URL': 'U.R.L.',
            'HTML': 'H.T.M.L.',
            'CSS': 'C.S.S.',
            'JS': 'JavaScript',
            'DB': 'database',
            'IoT': 'Internet of Things',
            'AR': 'A.R.',
            'VR': 'V.R.',
            'iOS': 'i.O.S.',
            '&': 'and',
            '%': 'percent',
            '>=': 'greater than or equal to',
            '<=': 'less than or equal to',
            '!=': 'not equal to',
            '==': 'equal to',
            '===': 'strictly equal to',
        }
        
        for acronym, expansion in acronyms.items():
            text = re.sub(r'\b' + re.escape(acronym) + r'\b', expansion, text)
        
        # Add pauses for better sentence breaks - helpful for processing
        text = text.replace('. ', '. <break time="0.5s"/> ')
        text = text.replace('! ', '! <break time="0.5s"/> ')
        text = text.replace('? ', '? <break time="0.5s"/> ')
        
        # Add slight pauses between list items for better comprehension
        text = re.sub(r'(\d+\.\s)', r'<break time="0.3s"/> \1', text)
        
        return text
    
    def save_conversation_to_file(self):
        """Save conversation history to a JSON file"""
        if not self.conversation_history:
            print("No conversation history to save.")
            return
            
        filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(filename, 'w') as file:
                json.dump(self.conversation_history, file, indent=2)
            print(f"Conversation history saved to {filename}")
        except Exception as e:
            print(f"Error saving conversation history: {e}")
    
    def cleanup(self):
        """Clean up resources"""
        # Stop emotion detection
        self.stop_emotion_detection()
        
        # Stop any ongoing speech
        self.is_speaking = False
        self.stop_speaking = True
        
        # Save conversation history if enabled
        if self.save_history and self.conversation_history:
            self.save_conversation_to_file()
        
        # Clean up wake word detection resources
        if self.audio_stream:
            self.audio_stream.close()
        
        if self.pa:
            self.pa.terminate()
        
        if self.porcupine:
            self.porcupine.delete()
    
    def run(self):
        """Main conversation loop"""
        if not self.setup_wake_word_detection():
            return
        
        try:
            # Start emotion detection
            self.start_emotion_detection()
            
            while True:
                if self.listen_for_wake_word():
                    # Create a thread to check for wake word during speech
                    interrupt_thread = threading.Thread(target=self.check_wake_word_during_speech)
                    interrupt_thread.daemon = True
                    interrupt_thread.start()
                    
                    # Get user input
                    user_input = self.listen_for_speech()
                    
                    if user_input:
                        # Get AI response with emotion context
                        response = self.get_ai_response(user_input)
                        
                        # Save to conversation history if enabled
                        if self.save_history:
                            emotion = self.get_current_emotion() or "unknown"
                            self.conversation_history.append({
                                "timestamp": datetime.now().isoformat(),
                                "user_input": user_input,
                                "user_emotion": emotion,
                                "ai_response": response
                            })
                        
                        # Speak the response
                        self.speak(response)
                    
                    # Stop the interrupt thread
                    self.stop_speaking = True
                    interrupt_thread.join()
                    self.stop_speaking = False
        
        except KeyboardInterrupt:
            print("\nStopping conversation...")
        finally:
            self.cleanup()


def check_env_file():
    """Check if .env file exists and create it if not"""
    env_path = ".env"
    if not os.path.exists(env_path):
        print("Creating .env file with template values...")
        with open(env_path, "w") as f:
            f.write("""# Required
PORCUPINE_ACCESS_KEY=your_porcupine_access_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# Optional
VOICE_ID=
VOICE_RATE=175
VOICE_VOLUME=1.0
SAVE_HISTORY=false
""")
        print(f".env file created at {os.path.abspath(env_path)}")
        print("Please edit the file and add your API keys.")
        return False
    return True


if __name__ == "__main__":
    # Check .env file
    if not check_env_file():
        exit(1)
    
    # Create and run the robot
    try:
        robot = ConversationRobot()
        robot.run()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}") 