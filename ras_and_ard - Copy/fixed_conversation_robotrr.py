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
import serial  # Add import for serial communication
import io  # For handling byte streams with Lemonfox API

# Import ElevenLabs client
try:
    from elevenlabs.client import ElevenLabs
    ELEVENLABS_CLIENT_AVAILABLE = True
except ImportError:
    print("ElevenLabs client library not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "elevenlabs"])
    from elevenlabs.client import ElevenLabs
    ELEVENLABS_CLIENT_AVAILABLE = True
    print("ElevenLabs client library installed successfully.")

# Set default Arduino port based on platform
DEFAULT_ARDUINO_PORT = "COM3" if platform.system() == "Windows" else "/dev/ttyACM0"

# Set environment variables directly in code
# This ensures the script works even if the .env file has issues
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
os.environ["SPEECH_RECOGNITION"] = "lemonfox"  # Using Lemonfox Whisper API
os.environ["LEMONFOX_API_KEY"] = "APFEQBqCJrLvMqrMUT3TIKpB3veCuSkp"  # Lemonfox API key
os.environ["SILENCE_THRESHOLD"] = "0.8"
os.environ["SPEECH_TIMEOUT"] = "1.5"
os.environ["PHRASE_TIMEOUT"] = "5.0"
os.environ["USE_EMOTION_DETECTION"] = "true"
os.environ["SHOW_WEBCAM"] = "false"
os.environ["PROCESS_EVERY_N_FRAMES"] = "15"
os.environ["ARDUINO_PORT"] = DEFAULT_ARDUINO_PORT  # Default Arduino port based on platform
os.environ["ARDUINO_BAUD_RATE"] = "9600"  # Match baud rate in Arduino code

# Try to load from .env file if it exists (but we already have fallbacks set above)
try:
    load_dotenv()
except Exception as e:
    print(f"Note: Could not load .env file: {e}")
    print("Using built-in environment variables instead.")

# Import winsound for Windows or use cross-platform alternative
if platform.system() == 'Windows':
    import winsound
else:
    # For non-Windows platforms, we'll use numpy and PyAudio for beeps
    def beep(frequency, duration):
        """Cross-platform beep function using PyAudio and NumPy"""
        try:
            # Create a PyAudio object
            p = pyaudio.PyAudio()
            
            # Generate a sine wave for the beep
            sample_rate = 44100  # CD quality audio
            samples = (np.sin(2 * np.pi * np.arange(sample_rate * duration / 1000) * frequency / sample_rate)).astype(np.float32)
            
            # Increase volume for better audibility
            samples = samples * 0.9  # Amplify to 90% of maximum volume
            
            # Open a stream
            stream = p.open(format=pyaudio.paFloat32,
                            channels=1,
                            rate=sample_rate,
                            output=True)
            
            # Play the sound
            stream.write(samples.tobytes())
            
            # Close the stream
            stream.stop_stream()
            stream.close()
            p.terminate()
            return True
        except Exception as e:
            print(f"Error playing beep: {e}")
            return False

# Check if FER is installed, if not install it
try:
    from fer import FER
except ImportError:
    import subprocess
    print("Installing FER library...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fer"])
    try:
        from fer import FER
        print("FER library installed successfully.")
    except ImportError:
        print("Failed to import FER after installation. Trying alternative approach...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade pip"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "fer", "--no-cache-dir"])
        from fer import FER
        print("FER library installed successfully with alternative approach.")

# Available default wake words for Porcupine
DEFAULT_WAKE_WORDS = [
    "bumblebee", "hey barista", "terminator", "pico clock", "alexa", 
    "hey google", "computer", "grapefruit", "grasshopper", "picovoice", 
    "porcupine", "jarvis", "hey siri", "ok google", "americano", "blueberry"
]

class ConversationRobot:
    def __init__(self, wake_word="computer", porcupine_access_key=None, save_history=False, voice_id=None, rate=None, volume=None, use_emotion_detection=True, show_webcam=False):
        # Load environment variables
        load_dotenv()
        
        # Initialize wake word detection
        self.porcupine_access_key = porcupine_access_key or os.getenv("PORCUPINE_ACCESS_KEY")
        if not self.porcupine_access_key:
            print("\nERROR: Porcupine access key is not set.")
            print("You need to get a free access key from https://console.picovoice.ai/")
            print("Then set it in your .env file as PORCUPINE_ACCESS_KEY=your_key_here\n")
            raise ValueError("Porcupine access key is required. Set it in .env file or pass it to the constructor.")
        
        # Enable/disable beep sounds
        self.enable_beep = self._parse_env_bool("ENABLE_BEEP", True)
        
        # Check if the access key is still the placeholder
        if "your_porcupine_access_key_here" in self.porcupine_access_key:
            print("\nERROR: You're using the placeholder text as your Porcupine access key.")
            print("You need to get a real access key from https://console.picovoice.ai/")
            print("Sign up for a free account and get your access key.")
            print("Then set it in your .env file as PORCUPINE_ACCESS_KEY=your_real_key_here\n")
            raise ValueError("Invalid Porcupine access key. Please use a real access key.")
        
        # Setup wake words (primary and custom)
        self.wake_word = wake_word or self._parse_env_str("WAKE_WORD", "computer")
        if self.wake_word not in DEFAULT_WAKE_WORDS:
            print(f"Warning: '{self.wake_word}' is not a default wake word. Using 'computer' instead.")
            print(f"Available default wake words: {', '.join(DEFAULT_WAKE_WORDS)}")
            self.wake_word = "computer"
        
        # Get custom wake words from environment
        self.custom_wake_words = []
        custom_wake_words_str = self._parse_env_str("CUSTOM_WAKE_WORDS", "")
        if custom_wake_words_str:
            self.custom_wake_words = [w.strip() for w in custom_wake_words_str.split(",") if w.strip()]
            print(f"Custom wake words configured: {', '.join(self.custom_wake_words)}")
        
        # All wake words (primary + custom)
        self.all_wake_words = [self.wake_word]
        self.wake_word_indices = {0: self.wake_word}  # Map indices to wake words
        
        self.porcupine = None
        self.pa = None
        self.audio_stream = None
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        
        # Initialize text-to-speech with pyttsx3 as fallback
        try:
            self.tts_engine = pyttsx3.init()
            self.setup_voice(voice_id, rate, volume)
            self.has_fallback_tts = True
        except Exception as e:
            print(f"Warning: Could not initialize fallback TTS system: {e}")
            self.tts_engine = None
            self.has_fallback_tts = False
            
        # Initialize ElevenLabs TTS
        self.elevenlabs_api_key = self._parse_env_str("ELEVENLABS_API_KEY", "")
        if not self.elevenlabs_api_key and "ELEVENLABS_API_KEY" in os.environ:
            self.elevenlabs_api_key = os.environ["ELEVENLABS_API_KEY"]
        if not self.elevenlabs_api_key:
            self.elevenlabs_api_key = "sk_a815878bc3184834c55fe90e89c9588bcb96759e64d9cb61"
            print("Using hardcoded ElevenLabs API key")
            
        self.elevenlabs_voice_id = self._parse_env_str("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
        self.elevenlabs_model_id = self._parse_env_str("ELEVENLABS_MODEL_ID", "eleven_monolingual_v1")
        self.elevenlabs_voices = []
        self.use_elevenlabs = True
        self.elevenlabs_failed = False
        self.audio_player = None
        
        # Add flag to control speech interruption
        self.is_speaking = False
        self.stop_speaking = False
        self.interrupt_thread = None
        
        # Initialize Gemini AI
        self.gemini_api_key = self._parse_env_str("GEMINI_API_KEY", "")
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
        
        # Memory and conversation history settings
        self.save_history = save_history or self._parse_env_bool("SAVE_HISTORY", False)
        self.conversation_history = []
        
        # Brain/Memory system settings
        self.use_memory = self._parse_env_bool("USE_MEMORY", True)
        self.memory_size = self._parse_env_int("MEMORY_SIZE", 10)  # Number of past conversations to remember
        self.memory_file = self._parse_env_str("MEMORY_FILE", "robot_memory.json")
        self.memory = deque(maxlen=self.memory_size)  # Limited size memory for recent conversations
        self.memory_exchanges_limit = self._parse_env_int("MEMORY_EXCHANGES_LIMIT", 5)  # Max exchanges per conversation to include
        
        # Load memory from previous sessions if enabled
        if self.use_memory:
            self.load_memory()
        
        # Initialize emotion detection
        self.use_emotion_detection = use_emotion_detection or self._parse_env_bool("USE_EMOTION_DETECTION", True)
        self.show_webcam = show_webcam or self._parse_env_bool("SHOW_WEBCAM", False)
        self.emotion_detector = None
        self.emotion_cap = None
        self.current_emotion = None
        self.emotion_thread = None
        self.emotion_running = False
        
        if self.use_emotion_detection:
            # First ensure all dependencies are installed
            if self.ensure_fer_dependencies():
                self.setup_emotion_detection()
            else:
                print("Warning: Could not ensure FER dependencies, emotion detection may not work properly")
                self.use_emotion_detection = False
        
        # Initialize Arduino serial connection
        self.arduino_port = self._parse_env_str("ARDUINO_PORT", DEFAULT_ARDUINO_PORT)
        self.arduino_baud_rate = self._parse_env_int("ARDUINO_BAUD_RATE", 9600)
        self.arduino_serial = None
        self.setup_arduino_connection()
        
        print("Conversation Robot initialized successfully!")

    # ... [other methods remain unchanged] ...

    def listen_for_speech(self):
        """Listen for speech input and transcribe using Lemonfox Whisper API"""
        print("Listening for your question... (speak now)")
        
        # Play a "ready to listen" beep (different tone)
        self.play_beep(800, 100)
        
        # Visual feedback that we're listening (only if show_webcam is True)
        if self.use_emotion_detection and self.show_webcam and self.emotion_cap and self.emotion_cap.isOpened():
            _, frame = self.emotion_cap.read()
            if frame is not None:
                # Clear any previous text by drawing a black rectangle
                cv2.rectangle(frame, 
                             (0, frame.shape[0] - 40), 
                             (frame.shape[1], frame.shape[0]), 
                             (0, 0, 0), 
                             -1)
                
                # Add new text with bright color for visibility
                cv2.putText(
                    frame,
                    "LISTENING NOW - Please speak...",
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),  # Yellow color for better visibility
                    2,
                )
                cv2.imshow("Emotion Detection", frame)
                cv2.waitKey(1)
        
        # Get speech recognition settings from environment or use defaults
        silence_threshold = float(os.getenv("SILENCE_THRESHOLD", "1.0"))  # Seconds of silence to stop listening
        speech_timeout = float(os.getenv("SPEECH_TIMEOUT", "2.0"))  # Max seconds to wait for speech to start
        phrase_timeout = float(os.getenv("PHRASE_TIMEOUT", "5.0"))  # Max seconds for a phrase
        
        # Get Lemonfox API key
        lemonfox_api_key = self._parse_env_str("LEMONFOX_API_KEY", "APFEQBqCJrLvMqrMUT3TIKpB3veCuSkp")
        
        with sr.Microphone() as source:
            # Minimal ambient noise adjustment (just 0.1 seconds)
            self.recognizer.adjust_for_ambient_noise(source, duration=0.1)
            
            # More aggressive energy threshold for better speech/silence detection
            self.recognizer.energy_threshold = 400
            
            # Set non-dynamic threshold for more predictable behavior
            self.recognizer.dynamic_energy_threshold = False
            
            # Set a lower pause threshold to detect end of speech faster
            self.recognizer.pause_threshold = silence_threshold
            
            try:
                # Use shorter timeouts for faster response
                audio = self.recognizer.listen(
                    source, 
                    timeout=speech_timeout,  # Wait max 2 seconds for speech to start
                    phrase_time_limit=phrase_timeout  # Max 5 seconds per phrase
                )
                
                # Visual feedback that we're processing (only if show_webcam is True)
                if self.use_emotion_detection and self.show_webcam and self.emotion_cap and self.emotion_cap.isOpened():
                    _, frame = self.emotion_cap.read()
                    if frame is not None:
                        cv2.putText(
                            frame,
                            "Processing speech...",
                            (10, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 140, 0),  # Blue-orange
                            2,
                        )
                        cv2.imshow("Emotion Detection", frame)
                        cv2.waitKey(1)
                
                print("Processing speech with Lemonfox Whisper API...")
                
                # Check audio duration - if it's too short, it might be noise
                audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
                audio_duration = len(audio_data) / audio.sample_rate / audio.sample_width
                
                if audio_duration < 0.3:  # Less than 0.3 seconds is probably not speech
                    print("Audio too short, likely not speech")
                    return None
                
                # Get speech recognition method from environment
                speech_recognition_method = os.getenv("SPEECH_RECOGNITION", "lemonfox").lower()
                
                # Use Lemonfox Whisper API
                if speech_recognition_method == "lemonfox":
                    try:
                        # Save audio to temporary file for API upload
                        temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                        temp_audio_path = temp_audio_file.name
                        temp_audio_file.close()
                        
                        with open(temp_audio_path, "wb") as f:
                            f.write(audio.get_wav_data())
                        
                        # Prepare API request
                        url = "https://api.lemonfox.ai/v1/audio/transcriptions"
                        headers = {
                            "Authorization": f"Bearer {lemonfox_api_key}"
                        }
                        data = {
                            "language": "english",
                            "response_format": "json"
                        }
                        
                        # Open the file for upload
                        files = {
                            "file": open(temp_audio_path, "rb")
                        }
                        
                        # Send request to Lemonfox API
                        print("Sending audio to Lemonfox API...")
                        response = requests.post(url, headers=headers, files=files, data=data)
                        
                        # Clean up temp file
                        files["file"].close()
                        os.unlink(temp_audio_path)
                        
                        # Check response
                        if response.status_code == 200:
                            result = response.json()
                            if "text" in result:
                                text = result["text"].strip()
                                print(f"You said (Lemonfox): {text}")
                                return text
                            else:
                                print(f"Unexpected response format from Lemonfox API: {result}")
                                # Fall back to Google if available
                                try:
                                    text = self.recognizer.recognize_google(audio)
                                    print(f"Falling back to Google: {text}")
                                    return text
                                except:
                                    return None
                        else:
                            print(f"Error from Lemonfox API: Status {response.status_code}")
                            print(f"Response: {response.text}")
                            
                            # Fall back to Google if available
                            try:
                                text = self.recognizer.recognize_google(audio)
                                print(f"Falling back to Google: {text}")
                                return text
                            except:
                                return None
                    except Exception as e:
                        print(f"Error using Lemonfox API: {e}")
                        # Fall back to Google if available
                        try:
                            text = self.recognizer.recognize_google(audio)
                            print(f"Falling back to Google: {text}")
                            return text
                        except:
                            return None
                
                # Fallback to Google if Lemonfox is not selected
                else:
                    try:
                        text = self.recognizer.recognize_google(audio)
                        print(f"You said (Google): {text}")
                        return text
                    except Exception as e:
                        print(f"Google speech recognition error: {e}")
                        return None
                    
            except sr.WaitTimeoutError:
                print("No speech detected within timeout period.")
                return None
            except sr.UnknownValueError:
                print("Could not understand audio.")
                return None
            except Exception as e:
                print(f"Error in speech recognition: {e}")
                return None

    # ... [rest of the class methods remain unchanged] ... 