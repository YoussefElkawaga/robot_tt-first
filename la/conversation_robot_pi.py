#!/usr/bin/env python3
"""
Raspberry Pi Optimized Conversation Robot
Optimized for Pi camera, reduced CPU usage, and better compatibility
"""

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
import sys
import platform
import glob
from collections import deque
import requests
import tempfile
import subprocess

# Raspberry Pi specific optimizations
IS_RASPBERRY_PI = False
try:
    with open('/proc/cpuinfo', 'r') as f:
        cpuinfo = f.read()
        if 'BCM' in cpuinfo or 'ARM' in cpuinfo:
            IS_RASPBERRY_PI = True
            print("🍓 Raspberry Pi detected - applying optimizations")
except:
    pass

# Import ElevenLabs client with error handling
try:
    from elevenlabs.client import ElevenLabs
    ELEVENLABS_CLIENT_AVAILABLE = True
except ImportError:
    print("ElevenLabs client library not found. Will install if needed...")
    ELEVENLABS_CLIENT_AVAILABLE = False

# Set environment variables directly in code
os.environ["PORCUPINE_ACCESS_KEY"] = "qqlP6xCMkzy3yWVx9Wg3RDsATOG1d06E1KAgbFilHWeoAl3zcIjkag=="
os.environ["GEMINI_API_KEY"] = "AIzaSyBuFAaIvXFRRX_LfAaTFnVTFFva-eV2Zw8"
os.environ["ELEVENLABS_API_KEY"] = "sk_a815878bc3184834c55fe90e89c9588bcb96759e64d9cb61"
os.environ["ELEVENLABS_VOICE_ID"] = "21m00Tcm4TlvDq8ikWAM"
os.environ["WAKE_WORD"] = "alexa"
os.environ["CUSTOM_WAKE_WORDS"] = "jarvis,computer,hey google"
os.environ["SAVE_HISTORY"] = "true"
os.environ["ENABLE_BEEP"] = "true"
os.environ["USE_MEMORY"] = "true"
os.environ["MEMORY_SIZE"] = "10"
os.environ["MEMORY_FILE"] = "robot_memory.json"
os.environ["MEMORY_EXCHANGES_LIMIT"] = "5"
os.environ["VOICE_RATE"] = "150"
os.environ["VOICE_VOLUME"] = "0.8"
os.environ["SPEECH_RECOGNITION"] = "google"
os.environ["SILENCE_THRESHOLD"] = "0.8"
os.environ["SPEECH_TIMEOUT"] = "1.5"
os.environ["PHRASE_TIMEOUT"] = "5.0"
os.environ["USE_EMOTION_DETECTION"] = "true"
os.environ["SHOW_WEBCAM"] = "false"

# Raspberry Pi specific optimizations
if IS_RASPBERRY_PI:
    os.environ["PROCESS_EVERY_N_FRAMES"] = "30"  # Process fewer frames on Pi
    os.environ["CAMERA_WIDTH"] = "320"           # Lower resolution for Pi
    os.environ["CAMERA_HEIGHT"] = "240"
    os.environ["CAMERA_FPS"] = "15"              # Lower FPS for Pi
else:
    os.environ["PROCESS_EVERY_N_FRAMES"] = "15"
    os.environ["CAMERA_WIDTH"] = "640"
    os.environ["CAMERA_HEIGHT"] = "480"
    os.environ["CAMERA_FPS"] = "30"

# Try to load from .env file if it exists
try:
    load_dotenv()
except Exception as e:
    print(f"Note: Could not load .env file: {e}")
    print("Using built-in environment variables instead.")

# Cross-platform beep function optimized for Pi
def beep(frequency, duration):
    """Cross-platform beep function optimized for Raspberry Pi"""
    try:
        if IS_RASPBERRY_PI:
            # Use simpler beep method for Pi
            try:
                # Try using aplay with generated tone (most reliable on Pi)
                import subprocess
                subprocess.run([
                    'speaker-test', '-t', 'sine', '-f', str(frequency), 
                    '-l', '1', '-s', '1'
                ], capture_output=True, timeout=2)
                return True
            except:
                # Fallback to system beep
                try:
                    os.system(f'echo -e "\\a" > /dev/console')
                    return True
                except:
                    pass
        
        # Standard method for other platforms
        p = pyaudio.PyAudio()
        sample_rate = 22050  # Lower sample rate for Pi
        samples = (np.sin(2 * np.pi * np.arange(sample_rate * duration / 1000) * frequency / sample_rate)).astype(np.float32)
        samples = samples * 0.7  # Reduce volume for Pi
        
        stream = p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=sample_rate,
                        output=True)
        
        stream.write(samples.tobytes())
        stream.stop_stream()
        stream.close()
        p.terminate()
        return True
    except Exception as e:
        print(f"Error playing beep: {e}")
        return False

# Check if FER is installed, install with Pi optimizations
try:
    from fer import FER
    FER_AVAILABLE = True
except ImportError:
    print("Installing FER library for emotion detection...")
    try:
        if IS_RASPBERRY_PI:
            # Install with Pi-specific optimizations
            subprocess.check_call([sys.executable, "-m", "pip", "install", "fer", "--no-cache-dir"])
        else:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "fer"])
        from fer import FER
        FER_AVAILABLE = True
    except Exception as e:
        print(f"Could not install FER: {e}")
        print("Emotion detection will be disabled")
        FER_AVAILABLE = False

# Available default wake words for Porcupine
DEFAULT_WAKE_WORDS = [
    "bumblebee", "hey barista", "terminator", "pico clock", "alexa", 
    "hey google", "computer", "grapefruit", "grasshopper", "picovoice", 
    "porcupine", "jarvis", "hey siri", "ok google", "americano", "blueberry"
]

class ConversationRobotPi:
    def __init__(self, wake_word="computer", porcupine_access_key=None, save_history=False, 
                 voice_id=None, rate=None, volume=None, use_emotion_detection=True, show_webcam=False):
        print("🤖 Initializing Raspberry Pi Conversation Robot...")
        
        # Load environment variables
        load_dotenv()
        
        # Initialize wake word detection
        self.porcupine_access_key = porcupine_access_key or os.getenv("PORCUPINE_ACCESS_KEY")
        if not self.porcupine_access_key:
            raise ValueError("Porcupine access key is required. Set it in .env file or pass it to the constructor.")
        
        # Enable/disable beep sounds
        self.enable_beep = self._parse_env_bool("ENABLE_BEEP", True)
        
        # Setup wake words
        self.wake_word = wake_word or self._parse_env_str("WAKE_WORD", "computer")
        if self.wake_word not in DEFAULT_WAKE_WORDS:
            print(f"Warning: '{self.wake_word}' is not a default wake word. Using 'computer' instead.")
            self.wake_word = "computer"
        
        # Get custom wake words
        self.custom_wake_words = []
        custom_wake_words_str = self._parse_env_str("CUSTOM_WAKE_WORDS", "")
        if custom_wake_words_str:
            self.custom_wake_words = [w.strip() for w in custom_wake_words_str.split(",") if w.strip()]
        
        self.all_wake_words = [self.wake_word]
        self.wake_word_indices = {0: self.wake_word}
        
        self.porcupine = None
        self.pa = None
        self.audio_stream = None
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        
        # Initialize TTS with Pi optimizations
        try:
            self.tts_engine = pyttsx3.init()
            self.setup_voice(voice_id, rate, volume)
            self.has_fallback_tts = True
        except Exception as e:
            print(f"Warning: Could not initialize TTS system: {e}")
            self.tts_engine = None
            self.has_fallback_tts = False
            
        # Initialize ElevenLabs TTS
        self.elevenlabs_api_key = self._parse_env_str("ELEVENLABS_API_KEY", "")
        self.elevenlabs_voice_id = self._parse_env_str("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
        self.elevenlabs_model_id = self._parse_env_str("ELEVENLABS_MODEL_ID", "eleven_monolingual_v1")
        self.use_elevenlabs = True
        self.elevenlabs_failed = False
        
        # Speech interruption flags
        self.is_speaking = False
        self.stop_speaking = False
        self.interrupt_thread = None
        
        # Initialize Gemini AI
        self.gemini_api_key = self._parse_env_str("GEMINI_API_KEY", "")
        if not self.gemini_api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY in .env file.")
        
        genai.configure(api_key=self.gemini_api_key)
        
        # Initialize Gemini model
        try:
            print("Initializing Gemini AI...")
            self.model = genai.GenerativeModel('gemini-2.0-flash')
            self.chat_session = self.model.start_chat(history=[])
        except Exception as e:
            print(f"Error initializing Gemini model: {e}")
            try:
                self.model = genai.GenerativeModel('gemini-pro')
                self.chat_session = self.model.start_chat(history=[])
            except Exception as e2:
                print(f"Error initializing fallback model: {e2}")
        
        # Memory and conversation history
        self.save_history = save_history or self._parse_env_bool("SAVE_HISTORY", False)
        self.conversation_history = []
        self.use_memory = self._parse_env_bool("USE_MEMORY", True)
        self.memory_size = self._parse_env_int("MEMORY_SIZE", 10)
        self.memory_file = self._parse_env_str("MEMORY_FILE", "robot_memory.json")
        self.memory = deque(maxlen=self.memory_size)
        self.memory_exchanges_limit = self._parse_env_int("MEMORY_EXCHANGES_LIMIT", 5)
        
        if self.use_memory:
            self.load_memory()
        
        # Initialize emotion detection with Pi optimizations
        self.use_emotion_detection = use_emotion_detection and FER_AVAILABLE and self._parse_env_bool("USE_EMOTION_DETECTION", True)
        self.show_webcam = show_webcam or self._parse_env_bool("SHOW_WEBCAM", False)
        self.emotion_detector = None
        self.emotion_cap = None
        self.current_emotion = None
        self.emotion_thread = None
        self.emotion_running = False
        
        # Camera settings optimized for Pi
        self.camera_width = self._parse_env_int("CAMERA_WIDTH", 320 if IS_RASPBERRY_PI else 640)
        self.camera_height = self._parse_env_int("CAMERA_HEIGHT", 240 if IS_RASPBERRY_PI else 480)
        self.camera_fps = self._parse_env_int("CAMERA_FPS", 15 if IS_RASPBERRY_PI else 30)
        self.process_every_n_frames = self._parse_env_int("PROCESS_EVERY_N_FRAMES", 30 if IS_RASPBERRY_PI else 15)
        
        if self.use_emotion_detection:
            self.setup_emotion_detection()
        
        print("✅ Conversation Robot initialized successfully!")
    
    def _parse_env_str(self, key, default=""):
        """Parse environment variable as string"""
        value = os.getenv(key, default)
        if value:
            parts = value.split('#', 1)
            return parts[0].strip()
        return default
    
    def _parse_env_int(self, key, default=0):
        """Parse environment variable as integer"""
        value = self._parse_env_str(key, str(default))
        try:
            return int(value)
        except ValueError:
            return default
    
    def _parse_env_float(self, key, default=0.0):
        """Parse environment variable as float"""
        value = self._parse_env_str(key, str(default))
        try:
            return float(value)
        except ValueError:
            return default
    
    def _parse_env_bool(self, key, default=False):
        """Parse environment variable as boolean"""
        value = self._parse_env_str(key, str(default)).lower()
        return value in ('true', 'yes', '1', 't', 'y')
    
    def setup_camera_for_pi(self):
        """Setup camera with Raspberry Pi specific optimizations"""
        try:
            print("🎥 Setting up camera for Raspberry Pi...")
            
            # Try different camera indices for Pi
            camera_indices = [0, 1, 2]  # Try multiple camera indices
            
            for idx in camera_indices:
                try:
                    print(f"Trying camera index {idx}...")
                    cap = cv2.VideoCapture(idx)
                    
                    if cap.isOpened():
                        # Test if we can read a frame
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            print(f"✅ Camera {idx} working!")
                            
                            # Set Pi-optimized camera properties
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
                            cap.set(cv2.CAP_PROP_FPS, self.camera_fps)
                            
                            # Pi-specific optimizations
                            if IS_RASPBERRY_PI:
                                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size
                                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                            
                            return cap
                        else:
                            cap.release()
                    else:
                        cap.release()
                        
                except Exception as e:
                    print(f"Camera {idx} failed: {e}")
                    continue
            
            print("❌ No working camera found")
            return None
            
        except Exception as e:
            print(f"Error setting up camera: {e}")
            return None
    
    def setup_emotion_detection(self):
        """Set up emotion detection optimized for Pi"""
        if not FER_AVAILABLE:
            print("FER library not available, skipping emotion detection")
            self.use_emotion_detection = False
            return False
            
        try:
            print("🎭 Setting up emotion detection...")
            
            # Initialize FER with Pi optimizations
            if IS_RASPBERRY_PI:
                # Use CPU-only mode and disable MTCNN for better Pi performance
                self.emotion_detector = FER(mtcnn=False)
                print("Using lightweight emotion detection for Pi")
            else:
                self.emotion_detector = FER(mtcnn=True)
            
            # Setup camera with Pi optimizations
            self.emotion_cap = self.setup_camera_for_pi()
            
            if not self.emotion_cap or not self.emotion_cap.isOpened():
                print("❌ Cannot open camera for emotion detection")
                self.use_emotion_detection = False
                return False
            
            print("✅ Emotion detection setup successfully!")
            return True
            
        except Exception as e:
            print(f"Error setting up emotion detection: {e}")
            self.use_emotion_detection = False
            return False
    
    def test_camera(self):
        """Test camera functionality"""
        print("🧪 Testing camera...")
        
        cap = self.setup_camera_for_pi()
        if not cap:
            print("❌ Camera test failed - no camera available")
            return False
        
        try:
            # Try to capture a few frames
            for i in range(5):
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✅ Frame {i+1}: {frame.shape}")
                else:
                    print(f"❌ Frame {i+1}: Failed to capture")
                    cap.release()
                    return False
                time.sleep(0.1)
            
            cap.release()
            print("✅ Camera test passed!")
            return True
            
        except Exception as e:
            print(f"❌ Camera test failed: {e}")
            cap.release()
            return False
    
    def load_memory(self):
        """Load memory from previous sessions"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r') as f:
                    memory_data = json.load(f)
                    if isinstance(memory_data, list):
                        for conversation in memory_data:
                            self.memory.append(conversation)
                print(f"Loaded {len(self.memory)} conversations from memory")
        except Exception as e:
            print(f"Error loading memory: {e}")
            self.memory = deque(maxlen=self.memory_size)
    
    def save_memory(self):
        """Save current memory"""
        if not self.use_memory:
            return
        try:
            memory_data = list(self.memory)
            if self.conversation_history:
                current_conversation = {
                    "timestamp": datetime.now().isoformat(),
                    "exchanges": self.conversation_history
                }
                memory_data.append(current_conversation)
            
            with open(self.memory_file, 'w') as f:
                json.dump(memory_data, f, indent=2)
        except Exception as e:
            print(f"Error saving memory: {e}")
    
    def get_memory_summary(self, max_items=3):
        """Generate memory summary (reduced for Pi performance)"""
        if not self.use_memory or not self.memory:
            return ""
        
        summary = "PREVIOUS CONVERSATIONS:\n"
        recent_memory = list(self.memory)[-max_items:]
        
        for i, mem in enumerate(recent_memory):
            summary += f"Conversation {i+1}:\n"
            exchanges = mem.get("exchanges", [])
            
            for exchange in exchanges[:2]:  # Limit exchanges for Pi
                if isinstance(exchange, dict):
                    user_msg = exchange.get("user", "")
                    robot_msg = exchange.get("robot", "")
                    
                    if user_msg:
                        summary += f"User: {user_msg}\n"
                    if robot_msg:
                        if len(robot_msg) > 50:  # Shorter truncation for Pi
                            robot_msg = robot_msg[:50] + "..."
                        summary += f"Robot: {robot_msg}\n"
            summary += "\n"
        
        return summary
    
    def setup_voice(self, voice_id=None, rate=None, volume=None):
        """Configure voice with Pi optimizations"""
        if not self.tts_engine:
            return
            
        rate = rate or self._parse_env_int("VOICE_RATE", 150 if IS_RASPBERRY_PI else 200)
        volume = volume or self._parse_env_float("VOICE_VOLUME", 0.8)
        
        try:
            voices = self.tts_engine.getProperty('voices')
            print(f"Available voices: {len(voices)}")
            
            # Set Pi-optimized speech rate
            self.tts_engine.setProperty('rate', rate)
            self.tts_engine.setProperty('volume', volume)
            
            print(f"Set speech rate to: {rate}, volume to: {volume}")
            
        except Exception as e:
            print(f"Error setting voice: {e}")
    
    def play_beep(self, frequency=1000, duration=200):
        """Play beep optimized for Pi"""
        if not self.enable_beep:
            return False
        
        try:
            return beep(frequency, duration)
        except Exception as e:
            print(f"Could not play beep: {e}")
            return False
    
    def start_emotion_detection_thread(self):
        """Start emotion detection thread with Pi optimizations"""
        if not self.use_emotion_detection or not self.emotion_detector:
            return
        
        self.emotion_running = True
        self.emotion_thread = threading.Thread(target=self.run_emotion_detection)
        self.emotion_thread.daemon = True
        self.emotion_thread.start()
        print("🎭 Emotion detection thread started")
    
    def run_emotion_detection(self):
        """Run emotion detection optimized for Pi"""
        if not self.emotion_cap or not self.emotion_detector:
            return
        
        print("Running emotion detection in background...")
        frame_count = 0
        
        while self.emotion_running:
            try:
                ret, frame = self.emotion_cap.read()
                if not ret:
                    continue
                
                frame_count += 1
                if frame_count % self.process_every_n_frames != 0:
                    continue
                
                # Resize frame for faster processing on Pi
                if IS_RASPBERRY_PI:
                    frame = cv2.resize(frame, (160, 120))  # Very small for Pi
                
                # Detect emotions
                result = self.emotion_detector.detect_emotions(frame)
                
                if result:
                    face = result[0]
                    emotions = face["emotions"]
                    dominant_emotion = max(emotions, key=emotions.get)
                    dominant_score = emotions[dominant_emotion]
                    
                    self.current_emotion = {
                        "emotion": dominant_emotion,
                        "score": dominant_score,
                        "all_emotions": emotions
                    }
                    
                    # Optional: Show webcam window
                    if self.show_webcam:
                        box = face["box"]
                        x, y, w, h = box
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        text = f"{dominant_emotion}: {dominant_score:.2f}"
                        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.imshow("Emotion Detection", frame)
                        cv2.waitKey(1)
                
            except Exception as e:
                print(f"Error in emotion detection: {e}")
                time.sleep(1)
        
        if self.show_webcam:
            cv2.destroyAllWindows()
    
    def stop_emotion_detection(self):
        """Stop emotion detection thread"""
        self.emotion_running = False
        if self.emotion_thread and self.emotion_thread.is_alive():
            try:
                self.emotion_thread.join(timeout=1.0)
            except Exception:
                pass
    
    def get_current_emotion(self):
        """Get current detected emotion"""
        if not self.use_emotion_detection or not self.current_emotion:
            return None
        return self.current_emotion
    
    def setup_wake_word_detection(self):
        """Setup wake word detection"""
        try:
            keywords = [self.wake_word]
            sensitivities = [0.7]
            
            for word in self.custom_wake_words:
                if word in DEFAULT_WAKE_WORDS:
                    keywords.append(word)
                    sensitivities.append(0.7)
                    self.wake_word_indices[len(keywords) - 1] = word
            
            self.all_wake_words = keywords
            
            self.porcupine = pvporcupine.create(
                access_key=self.porcupine_access_key,
                keywords=keywords,
                sensitivities=sensitivities
            )
            
            self.pa = pyaudio.PyAudio()
            self.audio_stream = self.pa.open(
                rate=self.porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self.porcupine.frame_length
            )
            
            print(f"✅ Wake word detection setup: {', '.join(keywords)}")
            return True
            
        except Exception as e:
            print(f"❌ Error setting up wake word detection: {e}")
            return False
    
    def listen_for_wake_word(self):
        """Listen for wake word"""
        wake_words_str = ", ".join([f"'{word}'" for word in self.all_wake_words])
        print(f"👂 Listening for wake words: {wake_words_str}...")
        
        try:
            while True:
                pcm = self.audio_stream.read(self.porcupine.frame_length)
                pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)
                
                keyword_index = self.porcupine.process(pcm)
                if keyword_index >= 0:
                    detected_word = self.wake_word_indices.get(keyword_index, self.wake_word)
                    print(f"🎯 Wake word detected: '{detected_word}'!")
                    
                    if self.enable_beep:
                        self.play_beep(1200, 200)
                    
                    return True
                    
        except KeyboardInterrupt:
            return False
        except Exception as e:
            print(f"Error in wake word detection: {e}")
            return False
    
    def listen_for_speech(self):
        """Listen for speech with Pi optimizations"""
        print("👂 Listening for your question...")
        
        self.play_beep(800, 100)
        
        silence_threshold = self._parse_env_float("SILENCE_THRESHOLD", 1.0)
        speech_timeout = self._parse_env_float("SPEECH_TIMEOUT", 2.0)
        phrase_timeout = self._parse_env_float("PHRASE_TIMEOUT", 5.0)
        
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.1)
            self.recognizer.energy_threshold = 300 if IS_RASPBERRY_PI else 400
            self.recognizer.dynamic_energy_threshold = False
            self.recognizer.pause_threshold = silence_threshold
            
            try:
                audio = self.recognizer.listen(
                    source, 
                    timeout=speech_timeout,
                    phrase_time_limit=phrase_timeout
                )
                
                print("🔄 Processing speech...")
                
                # Check audio duration
                audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
                audio_duration = len(audio_data) / audio.sample_rate / audio.sample_width
                
                if audio_duration < 0.3:
                    print("Audio too short")
                    return None
                
                # Use Google Speech Recognition
                text = self.recognizer.recognize_google(audio)
                print(f"You said: {text}")
                return text
                
            except sr.WaitTimeoutError:
                print("No speech detected")
                return None
            except sr.UnknownValueError:
                print("Could not understand audio")
                return None
            except Exception as e:
                print(f"Error in speech recognition: {e}")
                return None
    
    def get_ai_response(self, user_input, emotion_data=None):
        """Get AI response with Pi optimizations"""
        if not user_input:
            return "I didn't catch that. Could you please repeat?"
        
        try:
            print("🧠 Getting AI response...")
            
            # Simplified system prompt for Pi
            system_prompt = """You are KindCompanion, a gentle AI assistant for autistic children. 
            Be patient, kind, and use simple, clear language. Keep responses short and encouraging."""
            
            if not self.chat_session.history:
                self.chat_session.send_message(system_prompt)
            
            # Prepare message with limited context for Pi performance
            message_parts = []
            
            # Add limited memory context
            if self.use_memory:
                memory_context = self.get_memory_summary(max_items=2)  # Reduced for Pi
                if memory_context:
                    message_parts.append(memory_context)
            
            # Add emotion context if available
            if emotion_data and self.use_emotion_detection:
                emotion_context = f"[EMOTION: {emotion_data['emotion']} ({emotion_data['score']:.2f})]"
                message_parts.append(emotion_context)
            
            message_parts.append(f"Child: {user_input}")
            enhanced_input = "\n\n".join(message_parts)
            
            response = self.chat_session.send_message(enhanced_input)
            return self.clean_ai_response(response.text)
            
        except Exception as e:
            print(f"Error getting AI response: {e}")
            return "I'm having trouble thinking right now. Please try again."
    
    def clean_ai_response(self, text):
        """Clean AI response for TTS"""
        if not text:
            return text
        
        # Remove markdown
        text = text.replace('* ', '')
        text = text.replace('- ', '')
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        text = re.sub(r'[#`]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def speak(self, text):
        """Speak text with Pi optimizations"""
        if not text:
            return
        
        try:
            print(f"🗣️ Robot: {text}")
            
            self.is_speaking = True
            self.stop_speaking = False
            
            # Use simple TTS for Pi
            if self.has_fallback_tts:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            else:
                print("No TTS available - text displayed only")
            
        except Exception as e:
            print(f"Error in speech: {e}")
        finally:
            self.is_speaking = False
            self.stop_speaking = False
    
    def cleanup(self):
        """Clean up resources"""
        print("🧹 Cleaning up...")
        
        self.is_speaking = False
        self.stop_speaking = True
        
        if self.use_emotion_detection:
            self.stop_emotion_detection()
            if self.emotion_cap:
                self.emotion_cap.release()
        
        if self.audio_stream:
            self.audio_stream.close()
        if self.pa:
            self.pa.terminate()
        if self.porcupine:
            self.porcupine.delete()
        
        if self.save_history and self.conversation_history:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(self.conversation_history, f, indent=2)
            print(f"Conversation saved to {filename}")
        
        if self.use_memory:
            self.save_memory()
    
    def run(self):
        """Main execution loop optimized for Pi"""
        print("🚀 Starting Conversation Robot for Raspberry Pi...")
        
        # Test camera first
        if self.use_emotion_detection:
            if not self.test_camera():
                print("⚠️ Camera test failed - disabling emotion detection")
                self.use_emotion_detection = False
        
        if not self.setup_wake_word_detection():
            print("❌ Failed to setup wake word detection")
            return
        
        # Start emotion detection if enabled
        if self.use_emotion_detection:
            self.start_emotion_detection_thread()
        
        # Test beep sounds
        if self.enable_beep:
            print("🔊 Testing beep sounds...")
            self.play_beep(1000, 200)
        
        print("\n" + "="*50)
        print("🤖 RASPBERRY PI CONVERSATION ROBOT READY")
        print("="*50)
        wake_words_str = ", ".join([f"'{word}'" for word in self.all_wake_words])
        print(f"Say: {wake_words_str}")
        print("="*50 + "\n")
        
        try:
            while True:
                if self.listen_for_wake_word():
                    user_input = self.listen_for_speech()
                    
                    if user_input:
                        if user_input.lower() in ["stop", "exit", "quit", "goodbye"]:
                            self.speak("Goodbye! Have a great day.")
                            break
                        
                        emotion_data = self.get_current_emotion()
                        ai_response = self.get_ai_response(user_input, emotion_data)
                        
                        if self.save_history:
                            history_entry = {
                                "timestamp": datetime.now().isoformat(),
                                "user": user_input,
                                "robot": ai_response
                            }
                            if emotion_data:
                                history_entry["emotion"] = {
                                    "dominant": emotion_data["emotion"],
                                    "score": emotion_data["score"]
                                }
                            self.conversation_history.append(history_entry)
                        
                        self.speak(ai_response)
                    
                    print("Returning to wake word detection...")
                    
        except KeyboardInterrupt:
            print("\n🛑 Robot stopped by user")
        finally:
            self.cleanup()


def main():
    """Main function with Pi-specific setup"""
    print("🍓 Raspberry Pi Conversation Robot")
    print("=" * 40)
    
    if IS_RASPBERRY_PI:
        print("✅ Raspberry Pi detected")
        print("🔧 Applying Pi-specific optimizations...")
    else:
        print("💻 Running on standard computer")
    
    # Check environment
    porcupine_key = os.environ.get("PORCUPINE_ACCESS_KEY")
    if not porcupine_key or "your_porcupine_access_key_here" in porcupine_key:
        print("❌ Porcupine access key not set correctly")
        print("Get a free key from https://console.picovoice.ai/")
        return
    
    try:
        # Create robot with Pi optimizations
        robot = ConversationRobotPi(
            wake_word="alexa",
            save_history=True,
            use_emotion_detection=True,
            show_webcam=False  # Keep false for Pi performance
        )
        
        robot.run()
        
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()