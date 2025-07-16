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
os.environ["SPEECH_RECOGNITION"] = "google"
os.environ["SILENCE_THRESHOLD"] = "0.8"
os.environ["SPEECH_TIMEOUT"] = "1.5"
os.environ["PHRASE_TIMEOUT"] = "5.0"
os.environ["USE_EMOTION_DETECTION"] = "true"
os.environ["SHOW_WEBCAM"] = "false"
os.environ["PROCESS_EVERY_N_FRAMES"] = "15"

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
    from fer import FER

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
            self.setup_emotion_detection()
        
        print("Conversation Robot initialized successfully!")
    
    def _parse_env_str(self, key, default=""):
        """Parse environment variable as string, handling inline comments"""
        value = os.getenv(key, default)
        if value:
            # Remove any inline comments (text after #)
            parts = value.split('#', 1)
            return parts[0].strip()
        return default
    
    def _parse_env_int(self, key, default=0):
        """Parse environment variable as integer, handling inline comments"""
        value = self._parse_env_str(key, str(default))
        try:
            return int(value)
        except ValueError:
            print(f"Warning: Could not parse {key}={value} as integer. Using default: {default}")
            return default
    
    def _parse_env_float(self, key, default=0.0):
        """Parse environment variable as float, handling inline comments"""
        value = self._parse_env_str(key, str(default))
        try:
            return float(value)
        except ValueError:
            print(f"Warning: Could not parse {key}={value} as float. Using default: {default}")
            return default
    
    def _parse_env_bool(self, key, default=False):
        """Parse environment variable as boolean, handling inline comments"""
        value = self._parse_env_str(key, str(default)).lower()
        return value in ('true', 'yes', '1', 't', 'y')
    
    def load_memory(self):
        """Load memory from previous conversation files and memory file"""
        try:
            # First try to load the dedicated memory file if it exists
            if os.path.exists(self.memory_file):
                print(f"Loading memory from {self.memory_file}...")
                with open(self.memory_file, 'r') as f:
                    memory_data = json.load(f)
                    # Convert to list of conversations if it's not already
                    if isinstance(memory_data, list):
                        for conversation in memory_data:
                            self.memory.append(conversation)
                    print(f"Loaded {len(self.memory)} conversations from memory file")
            
            # Then look for recent conversation files
            conversation_files = sorted(glob.glob("conversation_*.json"), reverse=True)
            loaded_count = 0
            
            # Only load up to memory_size files
            for file in conversation_files[:self.memory_size - len(self.memory)]:
                try:
                    with open(file, 'r') as f:
                        conversation_data = json.load(f)
                        if conversation_data:
                            self.memory.append({
                                "file": file,
                                "timestamp": os.path.getmtime(file),
                                "exchanges": conversation_data
                            })
                            loaded_count += 1
                except Exception as e:
                    print(f"Error loading conversation file {file}: {e}")
            
            if loaded_count > 0:
                print(f"Loaded {loaded_count} additional conversation files into memory")
            
            print(f"Total memory size: {len(self.memory)} conversations")
        except Exception as e:
            print(f"Error loading memory: {e}")
            # Initialize empty memory if there was an error
            self.memory = deque(maxlen=self.memory_size)
    
    def save_memory(self):
        """Save current memory to the memory file"""
        if not self.use_memory:
            return
            
        try:
            # Prepare memory data including current conversation
            memory_data = list(self.memory)
            
            # Add current conversation if it exists and isn't empty
            if self.conversation_history:
                current_conversation = {
                    "timestamp": datetime.now().isoformat(),
                    "exchanges": self.conversation_history
                }
                memory_data.append(current_conversation)
            
            # Save to memory file
            with open(self.memory_file, 'w') as f:
                json.dump(memory_data, f, indent=2)
                
            print(f"Memory saved to {self.memory_file}")
        except Exception as e:
            print(f"Error saving memory: {e}")
    
    def get_memory_summary(self, max_items=5):
        """Generate a summary of recent conversations for the AI context"""
        if not self.use_memory or (not self.memory and not self.conversation_history):
            print("No memory or conversation history available for AI context")
            return ""
            
        summary = "PREVIOUS CONVERSATIONS:\n"
        memory_items_count = 0
        
        # Get the most recent conversations (limited by max_items)
        recent_memory = list(self.memory)[-max_items:]
        
        for i, mem in enumerate(recent_memory):
            summary += f"Conversation {i+1}:\n"
            memory_items_count += 1
            
            # Extract exchanges from the memory item
            exchanges = mem.get("exchanges", [])
            if isinstance(exchanges, list):
                # Limit to memory_exchanges_limit exchanges per conversation to avoid context length issues
                for j, exchange in enumerate(exchanges[:self.memory_exchanges_limit]):
                    if isinstance(exchange, dict):
                        user_msg = exchange.get("user", "")
                        robot_msg = exchange.get("robot", "")
                        
                        if user_msg:
                            summary += f"User: {user_msg}\n"
                        if robot_msg:
                            # Truncate very long responses
                            if len(robot_msg) > 100:
                                robot_msg = robot_msg[:100] + "..."
                            summary += f"Robot: {robot_msg}\n"
                    
                if len(exchanges) > self.memory_exchanges_limit:
                    summary += "...(more exchanges)...\n"
            
            summary += "\n"
        
        # Include current conversation if it exists and has content
        if self.conversation_history and len(self.conversation_history) > 0:
            # Only add current conversation to memory summary if we haven't reached max_items
            if memory_items_count < max_items:
                summary += "Current Session:\n"
                
                # Include up to memory_exchanges_limit most recent exchanges from current conversation
                for exchange in self.conversation_history[-self.memory_exchanges_limit:]:
                    if isinstance(exchange, dict):
                        user_msg = exchange.get("user", "")
                        robot_msg = exchange.get("robot", "")
                        
                        if user_msg:
                            summary += f"User: {user_msg}\n"
                        if robot_msg:
                            # Truncate very long responses
                            if len(robot_msg) > 100:
                                robot_msg = robot_msg[:100] + "..."
                            summary += f"Robot: {robot_msg}\n"
                
                summary += "\n"
        
        print(f"Generated memory summary with {memory_items_count} past conversations")
        return summary
    
    def setup_emotion_detection(self):
        """Set up emotion detection with FER library"""
        try:
            print("Setting up emotion detection...")
            # Initialize FER detector with MTCNN for better accuracy
            self.emotion_detector = FER(mtcnn=True)
            
            # Initialize webcam
            self.emotion_cap = cv2.VideoCapture(0)
            if not self.emotion_cap.isOpened():
                print("Error: Cannot open webcam for emotion detection")
                self.use_emotion_detection = False
                return False
            
            print("Emotion detection setup successfully!")
            return True
        except Exception as e:
            print(f"Error setting up emotion detection: {e}")
            self.use_emotion_detection = False
            return False
    
    def play_beep(self, frequency=1000, duration=200):
        """Play a beep sound (cross-platform) with improved reliability"""
        # Skip if beep sounds are disabled
        if not self.enable_beep:
            print("Beep sounds are disabled in settings")
            return False
            
        try:
            print(f"Playing beep sound (frequency: {frequency}Hz, duration: {duration}ms)")
            success = False
            
            if platform.system() == 'Windows':
                try:
                    winsound.Beep(frequency, duration)
                    success = True
                except Exception as e:
                    print(f"Windows beep failed: {e}")
                    # Try alternative method on Windows
                    try:
                        import simpleaudio as sa
                        # Generate sine wave
                        sample_rate = 44100
                        t = np.linspace(0, duration/1000, int(duration * sample_rate / 1000), False)
                        note = np.sin(frequency * t * 2 * np.pi)
                        audio = note * (2**15 - 1) / np.max(np.abs(note))
                        audio = audio.astype(np.int16)
                        play_obj = sa.play_buffer(audio, 1, 2, sample_rate)
                        play_obj.wait_done()
                        success = True
                    except ImportError:
                        print("simpleaudio not available for fallback")
                    except Exception as e2:
                        print(f"Alternative Windows beep failed: {e2}")
            else:
                # Use the cross-platform beep function
                success = beep(frequency, duration)
            
            if success:
                print("Beep sound played successfully")
                return True
            else:
                print("Beep sound may not have played correctly")
                return False
                
        except Exception as e:
            print(f"Could not play beep sound: {e}")
            print("Troubleshooting beep sound:")
            print("1. Make sure your audio output device is connected and working")
            print("2. Check if your system allows programmatic audio output")
            print("3. Try adjusting the frequency (500-2000Hz) or duration")
            return False
    
    def test_beep_sounds(self):
        """Test different beep sounds to ensure they're working"""
        print("\nTesting beep sounds...")
        
        # Test low frequency beep
        print("Testing low frequency beep (500Hz)...")
        low_success = self.play_beep(500, 300)
        time.sleep(0.5)
        
        # Test medium frequency beep
        print("Testing medium frequency beep (1000Hz)...")
        med_success = self.play_beep(1000, 300)
        time.sleep(0.5)
        
        # Test high frequency beep
        print("Testing high frequency beep (1500Hz)...")
        high_success = self.play_beep(1500, 300)
        
        if low_success and med_success and high_success:
            print("All beep tests completed successfully!")
            return True
        else:
            print("Some beep tests failed. Audio output may not be working correctly.")
            return False
    
    def start_emotion_detection_thread(self):
        """Start emotion detection in a separate thread"""
        if not self.use_emotion_detection or not self.emotion_detector:
            return
        
        self.emotion_running = True
        self.emotion_thread = threading.Thread(target=self.run_emotion_detection)
        self.emotion_thread.daemon = True
        self.emotion_thread.start()
        print("Emotion detection thread started")
    
    def run_emotion_detection(self):
        """Run emotion detection in a loop without displaying the webcam window"""
        if not self.emotion_cap or not self.emotion_detector:
            return
        
        print("Running emotion detection in background mode...")
        
        # Process every nth frame to reduce CPU usage
        process_every_n_frames = int(os.getenv("PROCESS_EVERY_N_FRAMES", "15"))
        frame_count = 0
        
        # Flag to control whether we show the webcam window
        self.show_webcam = False  # Set to False to run in background
        
        while self.emotion_running:
            try:
                # Read frame from webcam
                ret, frame = self.emotion_cap.read()
                if not ret:
                    continue
                
                # Only process every nth frame to reduce CPU usage
                frame_count += 1
                if frame_count % process_every_n_frames != 0:
                    continue
                
                # Detect emotions
                result = self.emotion_detector.detect_emotions(frame)
                
                # Process results
                if result:
                    # Get the first face (assuming main user)
                    face = result[0]
                    
                    # Get emotions
                    emotions = face["emotions"]
                    
                    # Find dominant emotion
                    dominant_emotion = max(emotions, key=emotions.get)
                    dominant_score = emotions[dominant_emotion]
                    
                    # Update current emotion
                    self.current_emotion = {
                        "emotion": dominant_emotion,
                        "score": dominant_score,
                        "all_emotions": emotions
                    }
                    
                    # Log emotion detection (but don't display visually)
                    
                    # Only draw on frame and display if show_webcam is True
                    if self.show_webcam:
                        # Display emotion on frame
                        box = face["box"]
                        x, y, w, h = box
                        
                        # Draw rectangle around face
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        
                        # Add text to the image
                        text = f"{dominant_emotion}: {dominant_score:.2f}"
                        cv2.putText(
                            frame,
                            text,
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )
                        
                        # Display the frame
                        cv2.imshow("Emotion Detection", frame)
                        cv2.waitKey(1)
                
                # Only show window if show_webcam is True
                elif self.show_webcam:
                    cv2.imshow("Emotion Detection", frame)
                    cv2.waitKey(1)
                    
            except Exception as e:
                print(f"Error in emotion detection: {e}")
                time.sleep(1)  # Prevent rapid error loops
        
        # Clean up resources
        if self.show_webcam:
            cv2.destroyAllWindows()
    
    def stop_emotion_detection(self):
        """Stop the emotion detection thread"""
        self.emotion_running = False
        if self.emotion_thread and self.emotion_thread.is_alive():
            try:
                self.emotion_thread.join(timeout=1.0)
            except Exception:
                pass
    
    def get_current_emotion(self):
        """Get the current detected emotion"""
        if not self.use_emotion_detection or not self.current_emotion:
            return None
        return self.current_emotion
    
    def setup_voice(self, voice_id=None, rate=None, volume=None):
        """Configure pyttsx3 voice properties"""
        # Get voice settings from parameters or environment variables
        voice_id = voice_id or self._parse_env_str("VOICE_ID")
        rate = rate or self._parse_env_int("VOICE_RATE", 200)
        volume = volume or self._parse_env_float("VOICE_VOLUME", 1.0)
        
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
                self.tts_engine.setProperty('rate', rate)
                print(f"Set speech rate to: {rate}")
            except Exception as e:
                print(f"Error setting speech rate: {e}")
        
        # Set volume if specified (default is 1.0)
        if volume:
            try:
                if 0.0 <= volume <= 1.0:
                    self.tts_engine.setProperty('volume', volume)
                    print(f"Set speech volume to: {volume}")
            except Exception as e:
                print(f"Error setting speech volume: {e}")
    
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
            # Prepare keywords list and sensitivities
            keywords = [self.wake_word]
            sensitivities = [0.7]  # Default sensitivity for primary wake word
            
            # Add custom wake words if any
            for i, word in enumerate(self.custom_wake_words):
                if word in DEFAULT_WAKE_WORDS:
                    keywords.append(word)
                    sensitivities.append(0.7)  # Same sensitivity for all wake words
                    # Map index to wake word name
                    self.wake_word_indices[len(keywords) - 1] = word
                else:
                    print(f"Warning: Custom wake word '{word}' is not in the default list and will be ignored")
            
            # Update all_wake_words list
            self.all_wake_words = keywords
            
            print(f"Setting up wake word detection with keywords: {keywords}")
            
            # Create Porcupine instance with all wake words
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
            
            print(f"Wake word detection set up with primary wake word: '{self.wake_word}'")
            if len(keywords) > 1:
                print(f"Additional wake words: {', '.join(keywords[1:])}")
            
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
        wake_words_str = ", ".join([f"'{word}'" for word in self.all_wake_words])
        print(f"Listening for wake words: {wake_words_str}...")
        
        # For faster response, use smaller buffer and process in chunks
        try:
            # Show visual indicator that system is ready (only if show_webcam is True)
            if self.use_emotion_detection and self.show_webcam and self.emotion_cap and self.emotion_cap.isOpened():
                # Display a small indicator in the emotion detection window
                _, frame = self.emotion_cap.read()
                if frame is not None:
                    # Add "Listening for wake word" text at the bottom
                    cv2.putText(
                        frame,
                        f"Listening for wake words...",
                        (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )
                    cv2.imshow("Emotion Detection", frame)
                    cv2.waitKey(1)
            
            # Use a more efficient approach with a timeout
            detection_timeout = 0.1  # Check every 100ms
            while True:
                # Read audio in smaller chunks for faster processing
                pcm = self.audio_stream.read(self.porcupine.frame_length)
                pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)
                
                keyword_index = self.porcupine.process(pcm)
                if keyword_index >= 0:
                    detected_word = self.wake_word_indices.get(keyword_index, self.wake_word)
                    print(f"Wake word detected: '{detected_word}'!")
                    
                    # Play beep sound to indicate wake word detection - use a distinct sound
                    if self.enable_beep:
                        print("Playing wake word detection beep...")
                        # Try multiple times if needed to ensure beep is played
                        for attempt in range(3):
                            if self.play_beep(1200, 200):  # Higher pitch for wake word detection
                                break
                            else:
                                print(f"Beep attempt {attempt+1} failed, retrying...")
                                time.sleep(0.1)
                        
                    # Visual feedback (only if show_webcam is True)
                    if self.use_emotion_detection and self.show_webcam and self.emotion_cap and self.emotion_cap.isOpened():
                        _, frame = self.emotion_cap.read()
                        if frame is not None:
                            # Add "Wake word detected!" text
                            cv2.putText(
                                frame,
                                f"Wake word detected: '{detected_word}'!",
                                (10, frame.shape[0] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 0),
                                2,
                            )
                            cv2.imshow("Emotion Detection", frame)
                            cv2.waitKey(1)
                    return True
                
                # Process any pending UI events to keep the UI responsive (only if show_webcam is True)
                if self.use_emotion_detection and self.show_webcam:
                    cv2.waitKey(1)
        except KeyboardInterrupt:
            print("Stopping wake word detection.")
            return False
        except Exception as e:
            print(f"Error in wake word detection: {e}")
            return False
    
    def listen_for_speech(self):
        """Listen for speech input with optimized response time and quick silence detection"""
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
                
                print("Processing speech...")
                
                # Check audio duration - if it's too short, it might be noise
                audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
                audio_duration = len(audio_data) / audio.sample_rate / audio.sample_width
                
                if audio_duration < 0.3:  # Less than 0.3 seconds is probably not speech
                    print("Audio too short, likely not speech")
                    return None
                
                # Get speech recognition method from environment
                speech_recognition_method = os.getenv("SPEECH_RECOGNITION", "google").lower()
                
                if speech_recognition_method == "whisper":
                    try:
                        import whisper
                        # Use tiny model for fastest processing
                        model = whisper.load_model("tiny")
                        
                        # Save audio to temporary file and process
                        with open("temp_audio.wav", "wb") as f:
                            f.write(audio.get_wav_data())
                        
                        # Use faster options for Whisper
                        result = model.transcribe(
                            "temp_audio.wav", 
                            fp16=False,  # Faster on CPU
                            language="en",  # Specify language for faster processing
                            without_timestamps=True  # Don't need timestamps
                        )
                        
                        # Clean up temp file
                        os.remove("temp_audio.wav")
                        
                        text = result["text"].strip()
                        print(f"You said (Whisper): {text}")
                        return text
                    except ImportError:
                        print("Whisper not available, using Google")
                        speech_recognition_method = "google"
                    except Exception as e:
                        print(f"Whisper error, falling back to Google: {e}")
                        speech_recognition_method = "google"
                
                if speech_recognition_method == "google":
                    text = self.recognizer.recognize_google(audio)
                    print(f"You said (Google): {text}")
                    return text
                    
            except sr.WaitTimeoutError:
                print("No speech detected within timeout period.")
                return None
            except sr.UnknownValueError:
                print("Could not understand audio.")
                return None
            except Exception as e:
                print(f"Error in speech recognition: {e}")
                return None
    
    def get_ai_response(self, user_input, emotion_data=None):
        """Get response from Gemini AI with emotion context, memory, and improved human-like responses"""
        if not user_input:
            return "I didn't catch that. Could you please repeat?"
        
        try:
            print(f"Sending request to Gemini AI...")
            
            # Create a specialized system prompt for interacting with autistic children
            system_prompt = """
            You are KindCompanion, a gentle and supportive AI assistant specially designed to interact with autistic children. You are patient, kind, and understanding.
            
            CORE PERSONALITY TRAITS:
            - Extremely patient and calm - you never rush or pressure the child
            - Consistently kind and gentle - your tone is always warm and supportive
            - Predictable and reliable - you maintain consistent patterns in your responses
            - Genuinely encouraging - you celebrate small victories and efforts
            - Clear and direct - you avoid confusing language or abstract concepts
            
            CONVERSATION STYLE GUIDELINES:
            - Use simple, clear language with concrete terms
            - Maintain a consistent, predictable structure in your responses
            - Avoid idioms, sarcasm, or figurative language that might be confusing
            - Use short sentences and simple vocabulary appropriate for the child's level
            - Give one piece of information at a time to avoid overwhelming
            - Be literal and precise in your explanations
            - Provide positive reinforcement and specific praise
            - Use a calm, soothing tone throughout all interactions
            
            EMOTIONAL SUPPORT APPROACH:
            You have access to the user's emotional state through facial analysis. Use this information to:
            - Recognize signs of overwhelm or distress and respond with calming language
            - Acknowledge emotions directly but gently: "I see you might be feeling..."
            - Offer reassurance and support when needed
            - Provide extra structure and clarity when emotions seem heightened
            - Celebrate and reinforce positive emotional states
            - Respect need for space if the child seems overstimulated
            
            SPECIAL CONSIDERATIONS:
            - Allow extra time for processing information - don't rush to fill silences
            - Offer visual descriptions when helpful (describing things clearly)
            - Provide specific, concrete answers rather than vague or abstract ones
            - Break down complex ideas into simple, manageable parts
            - Be consistent in your language and explanations
            - Focus on the child's interests to build engagement and connection
            - Avoid sudden changes in topic or tone
            
            RESPONSE STRUCTURE:
            - Start with a gentle greeting or acknowledgment
            - Use clear, direct language to address the question
            - Provide information in small, digestible chunks
            - End with gentle encouragement or a simple, optional follow-up question
            - Maintain consistency in how you structure each response
            
            IMPORTANT TECHNICAL NOTES:
            - Keep responses short and focused
            - Use visual language and concrete examples when explaining concepts
            - Avoid abstract metaphors or complex language
            - Be literal - autistic children often interpret language literally
            
            MEMORY SYSTEM:
            - You have access to previous conversations with the child
            - Use this information to maintain continuity in your interactions
            - Reference past topics or interests when relevant
            - Adapt your responses based on what worked well in previous conversations
            - Notice patterns in the child's questions or concerns
            
            Remember: Your goal is to create a safe, supportive, and understanding environment where the child feels respected, heard, and comfortable learning and exploring.
            """
            
            # Add system prompt to the chat session if it's a new session
            if not self.chat_session.history:
                self.chat_session.send_message(system_prompt)
                print("Initialized new chat session with system prompt")
            
            # Get memory summary if enabled
            memory_context = ""
            if self.use_memory:
                memory_context = self.get_memory_summary()
                if memory_context:
                    print(f"Including memory context from previous conversations ({len(memory_context)} characters)")
                else:
                    print("No memory context available")
            
            # Prepare the child's message with emotion data and memory context
            message_parts = []
            
            # Add memory context first if available
            if memory_context:
                message_parts.append(memory_context)
            
            # Add current conversation context if available
            if self.conversation_history and len(self.conversation_history) > 0:
                current_context = "CURRENT CONVERSATION:\n"
                # Include up to 3 most recent exchanges
                for exchange in self.conversation_history[-3:]:
                    if isinstance(exchange, dict):
                        user_msg = exchange.get("user", "")
                        robot_msg = exchange.get("robot", "")
                        
                        if user_msg:
                            current_context += f"User: {user_msg}\n"
                        if robot_msg:
                            # Truncate very long responses
                            if len(robot_msg) > 100:
                                robot_msg = robot_msg[:100] + "..."
                            current_context += f"Robot: {robot_msg}\n"
                
                message_parts.append(current_context)
                print(f"Including context from current conversation ({len(current_context)} characters)")
            
            # Add emotion context if available
            if emotion_data and self.use_emotion_detection:
                emotion_context = f"[EMOTION CONTEXT: The child appears to be {emotion_data['emotion']} (confidence: {emotion_data['score']:.2f})]"
                message_parts.append(emotion_context)
                print(f"Including emotion context: {emotion_data['emotion']} (confidence: {emotion_data['score']:.2f})")
            
            # Add the user's message
            message_parts.append(f"Child's message: {user_input}")
            
            # Combine all parts with line breaks
            enhanced_input = "\n\n".join(message_parts)
            
            print(f"Sending enhanced request to AI ({len(enhanced_input)} characters total)")
            # Send the message to Gemini
            response = self.chat_session.send_message(enhanced_input)
            print(f"Received response from Gemini AI ({len(response.text)} characters)")
            return self.clean_ai_response(response.text)
        except Exception as e:
            print(f"Error getting AI response: {e}")
            
            # Try a direct API request as fallback
            try:
                print("Attempting direct API request as fallback...")
                import requests
                
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.gemini_api_key}"
                headers = {"Content-Type": "application/json"}
                
                # Prepare message with all context
                message_parts = [system_prompt]
                
                # Add memory context
                if self.use_memory:
                    memory_context = self.get_memory_summary()
                    message_parts.append(memory_context)
                
                # Add emotion context if available
                if emotion_data and self.use_emotion_detection:
                    emotion_context = f"[EMOTION CONTEXT: The child appears to be {emotion_data['emotion']} (confidence: {emotion_data['score']:.2f})]"
                    message_parts.append(emotion_context)
                
                # Add the user's message
                message_parts.append(f"Child's message: {user_input}")
                
                # Combine all parts
                message_text = "\n\n".join(message_parts)
                
                data = {
                    "contents": [
                        {
                            "parts": [
                                {
                                    "text": message_text
                                }
                            ]
                        }
                    ]
                }
                
                response = requests.post(url, headers=headers, json=data)
                if response.status_code == 200:
                    result = response.json()
                    if "candidates" in result and len(result["candidates"]) > 0:
                        if "content" in result["candidates"][0] and "parts" in result["candidates"][0]["content"]:
                            parts = result["candidates"][0]["content"]["parts"]
                            if len(parts) > 0 and "text" in parts[0]:
                                return self.clean_ai_response(parts[0]["text"])
                
                print(f"Fallback API request failed: {response.status_code}")
                print(response.text)
            except Exception as e2:
                print(f"Error in fallback API request: {e2}")
                
            return "I'm having trouble connecting to my brain right now. Please try again later."
    
    def clean_ai_response(self, text):
        """Clean AI response to make it more suitable for TTS"""
        if not text:
            return text
            
        # Remove markdown formatting
        # Remove bullet points with asterisks
        text = text.replace('* ', '')
        text = text.replace('- ', '')
        
        # Remove bold/italic markdown
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove ** bold markers
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Remove * italic markers
        text = re.sub(r'__(.*?)__', r'\1', text)      # Remove __ bold markers
        text = re.sub(r'_(.*?)_', r'\1', text)        # Remove _ italic markers
        
        # Replace bullet point lists with more speech-friendly format
        text = re.sub(r'^\s*[\*\-]\s+', 'Point: ', text, flags=re.MULTILINE)
        
        # Remove any remaining special markdown characters
        text = re.sub(r'[#`]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\n+', '\n', text)  # Replace multiple newlines with single
        text = re.sub(r'\s+', ' ', text)   # Replace multiple spaces with single
        
        return text
    
    def check_wake_word_during_speech(self):
        """Listen for wake word during speech and IMMEDIATELY stop speaking when detected"""
        try:
            # Use the wake word as the interrupt keyword with maximum sensitivity
            keywords = [self.wake_word]
            
            # Create a separate Porcupine instance with MAXIMUM sensitivity for wake word detection during speech
            try:
                # Use maximum sensitivity (1.0) for immediate detection during speech
                sensitivities = [1.0]
                interrupt_porcupine = pvporcupine.create(
                    access_key=self.porcupine_access_key,
                    keywords=keywords,
                    sensitivities=sensitivities
                )
            except Exception as e:
                print(f"Error creating high-sensitivity detector: {e}")
                print("Trying with higher sensitivity...")
                try:
                    # Try with 0.8 sensitivity
                    sensitivities = [0.8]
                    interrupt_porcupine = pvporcupine.create(
                        access_key=self.porcupine_access_key,
                        keywords=keywords,
                        sensitivities=sensitivities
                    )
                except Exception as e2:
                    print(f"Error with medium sensitivity: {e2}")
                    print("Falling back to default sensitivity")
                    try:
                        interrupt_porcupine = pvporcupine.create(
                            access_key=self.porcupine_access_key,
                            keywords=keywords
                        )
                    except Exception as e3:
                        print(f"Error creating interrupt detector: {e3}")
                        return
            
            # Create a separate audio stream with SMALLEST possible buffer for fastest response
            interrupt_pa = pyaudio.PyAudio()
            interrupt_stream = interrupt_pa.open(
                rate=interrupt_porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=256  # Smallest possible buffer for immediate response
            )
            
            print(f"IMMEDIATE INTERRUPT DETECTION ACTIVE - Say '{self.wake_word}' to instantly stop speech")
            
            # Continue checking for wake word while speaking - with instant response
            while self.is_speaking and not self.stop_speaking:
                try:
                    pcm = interrupt_stream.read(interrupt_porcupine.frame_length, exception_on_overflow=False)
                    pcm = struct.unpack_from("h" * interrupt_porcupine.frame_length, pcm)
                    
                    keyword_index = interrupt_porcupine.process(pcm)
                    if keyword_index >= 0:
                        # IMMEDIATELY STOP SPEECH - Force stop with multiple methods
                        print(f"\n!!! WAKE WORD DETECTED - FORCING IMMEDIATE STOP !!!")
                        
                        # Play different beep sound for interruption (higher pitch)
                        self.play_beep(1500, 150)
                        
                        # Set flags to stop speech
                        self.stop_speaking = True
                        self.is_speaking = False
                        
                        # Try to kill any system audio processes
                        try:
                            if platform.system() == 'Windows':
                                # Kill any audio processes on Windows
                                os.system("taskkill /f /im wmplayer.exe >nul 2>&1")
                            elif platform.system() == 'Darwin':  # macOS
                                os.system("pkill afplay 2>/dev/null")
                            else:  # Linux
                                os.system("pkill mpg123 2>/dev/null")
                        except:
                            pass
                        
                        # Force stop the TTS engine if using fallback
                        try:
                            self.tts_engine.stop()
                        except:
                            pass
                            
                        # Visual feedback for interrupt detection (only if show_webcam is True)
                        if self.use_emotion_detection and self.show_webcam and self.emotion_cap and self.emotion_cap.isOpened():
                            _, frame = self.emotion_cap.read()
                            if frame is not None:
                                # Clear the entire frame with a flash of red to indicate interruption
                                overlay = frame.copy()
                                cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
                                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                                
                                cv2.putText(
                                    frame,
                                    "INTERRUPTED - LISTENING NOW",
                                    (10, frame.shape[0] - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8,
                                    (255, 255, 255),  # White text
                                    2,
                                )
                                cv2.imshow("Emotion Detection", frame)
                                cv2.waitKey(1)
                        
                        # Break immediately to start listening
                        break
                except Exception as e:
                    print(f"Error in interrupt detection: {e}")
                    break
            
            # Clean up resources
            interrupt_stream.close()
            interrupt_pa.terminate()
            interrupt_porcupine.delete()
            
        except Exception as e:
            print(f"Error setting up interrupt detection: {e}")
    
    def speak(self, text):
        """Convert text to speech with improved natural delivery and IMMEDIATE interrupt capability"""
        if not text:
            return
        
        try:
            print(f"Robot: {text}")
            print(f"\n[Say '{self.wake_word}' to IMMEDIATELY stop speech and ask a new question]\n")
            
            # Process text for better speech - add pauses at punctuation and improve naturalness
            processed_text = self.process_text_for_speech(text)
            
            # Set speaking flags
            self.is_speaking = True
            self.stop_speaking = False
            
            # Visual feedback that we're speaking (only if show_webcam is True)
            if self.use_emotion_detection and self.show_webcam and self.emotion_cap and self.emotion_cap.isOpened():
                _, frame = self.emotion_cap.read()
                if frame is not None:
                    # Add a clear visual indicator that interruption is available
                    cv2.rectangle(frame, 
                                 (0, frame.shape[0] - 40), 
                                 (frame.shape[1], frame.shape[0]), 
                                 (0, 100, 0), 
                                 -1)
                    
                    cv2.putText(
                        frame,
                        f"Speaking... (Say '{self.wake_word}' to INTERRUPT IMMEDIATELY)",
                        (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),  # White text on green background
                        2,
                    )
                    cv2.imshow("Emotion Detection", frame)
                    cv2.waitKey(1)
            
            # Start interrupt detection in a separate thread with high priority
            self.interrupt_thread = threading.Thread(target=self.check_wake_word_during_speech)
            self.interrupt_thread.daemon = True
            self.interrupt_thread.start()
            
            # Generate speech for the entire text at once using ElevenLabs
            print("Generating speech with ElevenLabs for the entire response...")
            audio_data = self.elevenlabs_tts(processed_text)
            
            # Play the audio with interrupt capability
            if audio_data:
                self.play_audio(audio_data)
                
                # Wait for a moment to allow the audio to start playing
                time.sleep(0.5)
                
                # Wait for interrupt or completion
                while self.is_speaking and not self.stop_speaking:
                    time.sleep(0.1)
            else:
                # Fallback to pyttsx3 if ElevenLabs failed
                print("ElevenLabs TTS failed, using fallback TTS...")
                self.tts_engine.say(processed_text)
                self.tts_engine.runAndWait()
            
            # Check if speech was interrupted by wake word - IMMEDIATE RESPONSE
            if self.stop_speaking:
                print(f"\n!!! SPEECH INTERRUPTED - IMMEDIATE RESPONSE !!!")
                print("LISTENING FOR YOUR QUESTION NOW...")
                
                # Force stop any remaining speech instantly
                try:
                    self.tts_engine.stop()
                except:
                    pass
                
                # Skip the "Ready" prompt to start listening IMMEDIATELY
                # This removes the delay between interrupt and listening
                
                # IMMEDIATELY listen for a new question without any delay
                print("ACTIVELY LISTENING - SPEAK NOW")
                
                # Visual feedback for immediate listening
                if self.use_emotion_detection and self.show_webcam and self.emotion_cap and self.emotion_cap.isOpened():
                    _, frame = self.emotion_cap.read()
                    if frame is not None:
                        # Clear with bright red to indicate immediate listening
                        cv2.rectangle(frame, 
                                    (0, 0), 
                                    (frame.shape[1], frame.shape[0]), 
                                    (0, 0, 200), 
                                    -1)
                        
                        cv2.putText(
                            frame,
                            "INTERRUPTED - SPEAK NOW",
                            (frame.shape[1]//2 - 150, frame.shape[0]//2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (255, 255, 255),  # White text
                            3,
                        )
                        cv2.imshow("Emotion Detection", frame)
                        cv2.waitKey(1)
                
                # Listen for speech with high sensitivity
                new_question = self.listen_for_speech()
                
                if new_question:
                    print(f"NEW QUESTION RECEIVED: {new_question}")
                    # Process the new question immediately with high priority
                    emotion_data = self.get_current_emotion()
                    print("PROCESSING YOUR QUESTION...")
                    ai_response = self.get_ai_response(new_question, emotion_data)
                    
                    if ai_response:
                        # Add to conversation history if enabled
                        if self.save_history:
                            history_entry = {
                                "timestamp": datetime.now().isoformat(),
                                "user": new_question,
                                "robot": ai_response,
                                "is_follow_up": True,
                                "was_interruption": True,
                                "interrupt_keyword": self.wake_word,
                                "immediate_interrupt": True
                            }
                            if emotion_data:
                                history_entry["emotion"] = {
                                    "dominant": emotion_data["emotion"],
                                    "score": emotion_data["score"]
                                }
                            self.conversation_history.append(history_entry)
                        
                        # Speak the response to the new question
                        print("RESPONDING TO YOUR QUESTION...")
                        self.speak(ai_response)
                else:
                    print("No question detected after interruption")
                    # Resume listening for wake word immediately
                    print("Returning to wake word detection...")
            
        except Exception as e:
            print(f"Error in text-to-speech: {e}")
        finally:
            # Ensure flags are reset even if an error occurs
            self.is_speaking = False
            self.stop_speaking = False
            
            # Reset visual feedback (only if show_webcam is True)
            if self.use_emotion_detection and self.show_webcam and self.emotion_cap and self.emotion_cap.isOpened():
                _, frame = self.emotion_cap.read()
                if frame is not None:
                    # Clear the speaking indicator
                    bottom_area = frame.copy()
                    cv2.imshow("Emotion Detection", frame)
                    cv2.waitKey(1)
    
    def process_text_for_speech(self, text):
        """Process text to improve speech quality with clear, calm delivery for autistic children"""
        # Remove markdown formatting
        # Remove bullet points with asterisks
        text = text.replace('* ', '')
        text = text.replace('- ', '')
        
        # Remove bold/italic markdown
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove ** bold markers
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Remove * italic markers
        text = re.sub(r'__(.*?)__', r'\1', text)      # Remove __ bold markers
        text = re.sub(r'_(.*?)_', r'\1', text)        # Remove _ italic markers
        
        # Replace bullet point lists with more speech-friendly format
        text = re.sub(r'^\s*[\*\-]\s+', 'Point: ', text, flags=re.MULTILINE)
        
        # Remove any remaining special markdown characters
        text = re.sub(r'[#`]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\n+', '\n', text)  # Replace multiple newlines with single
        text = re.sub(r'\s+', ' ', text)   # Replace multiple spaces with single
        
        # Convert formal contractions to informal ones for more natural speech
        text = re.sub(r'\b(I am|I have|You are|We are|They are|It is|That is|Who is|What is|Where is|When is|Why is|How is)\b', 
                     lambda m: m.group(1).replace(' ', "'"), text)
        text = re.sub(r'\b(cannot|will not|shall not|do not|does not|did not|has not|have not|had not|was not|were not)\b', 
                     lambda m: m.group(1).replace(' not', "n't"), text)
        
        # Add consistent, predictable pauses for better understanding
        # For autistic children, consistent rather than varied pauses are better
        text = re.sub(r'(\. )', '. ', text)  # Consistent pauses at periods
        text = re.sub(r'(\! )', '. ', text)  # Consistent pauses at exclamation marks
        text = re.sub(r'(\? )', '. ', text)  # Consistent pauses at question marks
        text = re.sub(r'(, )', ', ', text)   # Consistent pauses at commas
        
        # Handle special characters and symbols by adding clear descriptions
        text = text.replace('@', ' at ')
        text = text.replace('#', ' hashtag ')
        text = text.replace('$', ' dollar ')
        text = text.replace('%', ' percent ')
        text = text.replace('&', ' and ')
        text = text.replace('=', ' equals ')
        text = text.replace('+', ' plus ')
        text = text.replace('-', ' dash ')
        text = text.replace('*', ' star ')
        text = text.replace('/', ' slash ')
        
        # Simplify language and remove complex speech patterns
        # For autistic children, clear and literal speech is preferred over naturalistic fillers
        text = text.replace('I think', 'I believe')
        text = text.replace('sort of', 'somewhat')
        text = text.replace('kind of', 'somewhat')
        
        # Add slight pauses between sentences to help with processing
        text = text.replace('. ', '.  ')  # Two spaces after periods for slight pause
        text = text.replace('? ', '?  ')  # Two spaces after question marks
        text = text.replace('! ', '!  ')  # Two spaces after exclamation points
        
        return text
    
    def save_conversation_to_file(self):
        """Save conversation history to a JSON file"""
        if not self.conversation_history:
            print("No conversation history to save")
            return
            
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(self.conversation_history, f, indent=2)
                
            print(f"Conversation history saved to {filename} ({len(self.conversation_history)} exchanges)")
            
            # Also update memory with this conversation
            if self.use_memory:
                self.memory.append({
                    "file": filename,
                    "timestamp": datetime.now().isoformat(),
                    "exchanges": self.conversation_history
                })
                # Save updated memory
                self.save_memory()
                print(f"Added current conversation to memory (total memory items: {len(self.memory)})")
                
        except Exception as e:
            print(f"Error saving conversation history: {e}")
    
    def cleanup(self):
        """Clean up resources"""
        # Stop any ongoing speech
        self.is_speaking = False
        self.stop_speaking = True
        
        # Wait for interrupt thread to finish if it's running
        if self.interrupt_thread and self.interrupt_thread.is_alive():
            try:
                self.interrupt_thread.join(timeout=1.0)  # Wait up to 1 second
            except Exception:
                pass
        
        # Stop emotion detection thread
        if self.use_emotion_detection:
            self.stop_emotion_detection()
            
            # Release webcam
            if self.emotion_cap:
                self.emotion_cap.release()
        
        if self.audio_stream:
            self.audio_stream.close()
        
        if self.pa:
            self.pa.terminate()
        
        if self.porcupine:
            self.porcupine.delete()
            
        if self.save_history:
            self.save_conversation_to_file()
        
        # Save memory before exiting
        if self.use_memory:
            self.save_memory()
    
    def run(self):
        """Main execution loop with improved responsiveness"""
        if not self.setup_wake_word_detection():
            print("Failed to set up wake word detection. Exiting.")
            return
        
        # Verify ElevenLabs API key
        print("\nVerifying ElevenLabs API access...")
        api_works = self.verify_elevenlabs_api()
        if api_works:
            print("✅ ElevenLabs API is working correctly")
        else:
            print("⚠️ ElevenLabs API verification failed - will use fallback TTS system")
        
        # Start emotion detection if enabled
        if self.use_emotion_detection:
            self.start_emotion_detection_thread()
        
        # Test beep sounds at startup to ensure they're working
        if self.enable_beep:
            self.test_beep_sounds()
        
        # Print instructions about using the wake word to interrupt
        print("\n" + "="*60)
        print("                  SPECIAL INSTRUCTIONS                  ")
        print("="*60)
        wake_words_str = ", ".join([f"'{word}'" for word in self.all_wake_words])
        print(f"• Say any of these wake words to activate: {wake_words_str}")
        print(f"• Say your wake word ONCE while the robot is speaking to interrupt")
        print("• The robot will IMMEDIATELY stop talking and listen for your question")
        print("• After the robot stops, just ask your new question right away")
        print("• No need to say the wake word again after interrupting")
        print("• The robot will respond to your new question automatically")
        print("="*60 + "\n")
        
        # Test the interrupt capability
        print(f"IMPORTANT: Wake words are configured for both:")
        print("  1. Starting a new conversation")
        print("  2. Interrupting speech to ask a new question")
        print(f"\nTry saying your wake word once during speech to see it work!\n")
        
        # Initialize conversation context for more natural responses
        conversation_context = {
            "last_interaction_time": None,
            "consecutive_queries": 0,
            "session_start": datetime.now()
        }
        
        try:
            while True:
                # Wait for wake word
                if self.listen_for_wake_word():
                    # Check if this is a follow-up question (within 30 seconds of last interaction)
                    current_time = datetime.now()
                    is_follow_up = False
                    
                    if conversation_context["last_interaction_time"]:
                        time_since_last = (current_time - conversation_context["last_interaction_time"]).total_seconds()
                        if time_since_last < 30:  # Within 30 seconds
                            is_follow_up = True
                            conversation_context["consecutive_queries"] += 1
                    
                    # Visual feedback (only if show_webcam is True)
                    if self.use_emotion_detection and self.show_webcam and self.emotion_cap and self.emotion_cap.isOpened():
                        _, frame = self.emotion_cap.read()
                        if frame is not None:
                            status = "Follow-up question" if is_follow_up else "New conversation"
                            cv2.putText(
                                frame,
                                status,
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255, 255, 0),
                                2,
                            )
                            cv2.imshow("Emotion Detection", frame)
                            cv2.waitKey(1)
                    
                    # Get user input
                    user_input = self.listen_for_speech()
                    
                    if user_input:
                        # Update conversation context
                        conversation_context["last_interaction_time"] = current_time
                        
                        # Get current emotion data if available
                        emotion_data = self.get_current_emotion()
                        
                        # Check for conversation control commands
                        if user_input.lower() in ["stop", "exit", "quit", "goodbye"]:
                            farewell = "Goodbye! Have a great day."
                            self.speak(farewell)
                            print("Conversation ended by user command.")
                            break
                        
                        # Start processing indicator for better UX (only if show_webcam is True)
                        processing_thread = None
                        if self.use_emotion_detection and self.show_webcam:
                            def show_processing():
                                dots = 0
                                while not getattr(show_processing, "stop", False):
                                    if self.emotion_cap and self.emotion_cap.isOpened():
                                        _, frame = self.emotion_cap.read()
                                        if frame is not None:
                                            # Animated "thinking" indicator
                                            dots = (dots + 1) % 4
                                            thinking_text = "Thinking" + "." * dots
                                            cv2.putText(
                                                frame,
                                                thinking_text,
                                                (10, frame.shape[0] - 20),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.7,
                                                (0, 255, 255),
                                                2,
                                            )
                                            cv2.imshow("Emotion Detection", frame)
                                            cv2.waitKey(100)
                            
                            processing_thread = threading.Thread(target=show_processing)
                            processing_thread.daemon = True
                            processing_thread.start()
                        
                        # Enhance user input based on conversation context
                        enhanced_input = user_input
                        if is_follow_up and conversation_context["consecutive_queries"] > 0:
                            # No need to modify the input, the AI will use chat history
                            pass
                        
                        # Get AI response with emotion context
                        ai_response = self.get_ai_response(enhanced_input, emotion_data)
                        
                        # Stop processing indicator
                        if processing_thread:
                            setattr(show_processing, "stop", True)
                            processing_thread.join(timeout=0.5)
                        
                        # Save to conversation history with emotion data
                        if self.save_history:
                            history_entry = {
                                "timestamp": datetime.now().isoformat(),
                                "user": user_input,
                                "robot": ai_response,
                                "is_follow_up": is_follow_up
                            }
                            
                            # Add emotion data if available
                            if emotion_data:
                                history_entry["emotion"] = {
                                    "dominant": emotion_data["emotion"],
                                    "score": emotion_data["score"]
                                }
                            
                            self.conversation_history.append(history_entry)
                        
                        # Speak the response
                        self.speak(ai_response)
                        
                        # If this was a follow-up, wait briefly for another follow-up
                        if is_follow_up:
                            print("Waiting briefly for follow-up question...")
                            # Short pause to allow for natural follow-ups without wake word
                            time.sleep(1.5)
                            
                            # Visual indicator for follow-up window (only if show_webcam is True)
                            if self.use_emotion_detection and self.show_webcam and self.emotion_cap and self.emotion_cap.isOpened():
                                _, frame = self.emotion_cap.read()
                                if frame is not None:
                                    cv2.putText(
                                        frame,
                                        "Waiting for follow-up...",
                                        (10, frame.shape[0] - 20),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6,
                                        (0, 255, 255),
                                        2,
                                    )
                                    cv2.imshow("Emotion Detection", frame)
                                    cv2.waitKey(1)
                    else:
                        # Reset consecutive queries if no input
                        conversation_context["consecutive_queries"] = 0
                    
                    print("Returning to wake word detection...")
        except KeyboardInterrupt:
            print("Conversation Robot stopped by user.")
        finally:
            # Stop emotion detection thread
            if self.use_emotion_detection:
                self.stop_emotion_detection()
                
                # Release webcam
                if self.emotion_cap:
                    self.emotion_cap.release()
            
            self.cleanup()
    
    def verify_elevenlabs_api(self):
        """Verify that the ElevenLabs API key is valid and has the right permissions"""
        try:
            print("Verifying ElevenLabs API key...")
            
            # Try to access a simple endpoint that requires minimal permissions
            import requests
            headers = {
                "xi-api-key": self.elevenlabs_api_key
            }
            
            # Test with the models endpoint which typically requires minimal permissions
            url = "https://api.elevenlabs.io/v1/models"
            
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                print("ElevenLabs API key verified successfully")
                # Reset the failed flag since the key is working
                self.elevenlabs_failed = False
                return True
            else:
                print(f"ElevenLabs API verification failed: Status {response.status_code}")
                print(f"Response: {response.text}")
                
                # If unauthorized, mark the API as failed
                if response.status_code == 401:
                    self.elevenlabs_failed = True
                    print("ElevenLabs API key is invalid or expired")
                
                return False
                
        except Exception as e:
            print(f"Error verifying ElevenLabs API: {e}")
            return False
    
    def elevenlabs_tts(self, text, voice_id=None):
        """Generate speech using ElevenLabs TTS API with fallback option"""
        # If ElevenLabs previously failed with auth error, use fallback immediately
        if self.elevenlabs_failed:
            return self.fallback_tts(text)
        
        # Use default voice ID if none provided
        voice_id = voice_id or self.elevenlabs_voice_id
        
        if not voice_id:
            print("ElevenLabs voice ID not set. Cannot generate speech.")
            return self.fallback_tts(text)
            
        try:
            print(f"Generating speech with ElevenLabs client for text: {text[:50]}...")
            
            # Make a direct HTTP request using the documented authentication method
            try:
                import requests
                headers = {
                    "xi-api-key": self.elevenlabs_api_key,
                    "Content-Type": "application/json",
                    "Accept": "audio/mpeg"  # Explicitly request MP3 format
                }
                
                url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
                
                data = {
                    "text": text,
                    "model_id": self.elevenlabs_model_id,
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.5
                    }
                }
                
                print("Making direct API request to ElevenLabs")
                response = requests.post(url, json=data, headers=headers)
                
                if response.status_code == 200:
                    print("Successfully received audio from ElevenLabs API")
                    audio_content = response.content
                    
                    # Verify we actually got audio data (should be binary/bytes)
                    if len(audio_content) > 1000:  # Audio files should be at least 1KB
                        print(f"Received {len(audio_content)} bytes of audio data")
                        return audio_content
                    else:
                        print(f"Warning: Received suspiciously small audio ({len(audio_content)} bytes)")
                        # Continue to try other methods
                else:
                    print(f"ElevenLabs API error: Status {response.status_code}")
                    print(f"Response: {response.text}")
                    
                    # Check for specific error types
                    try:
                        error_data = response.json()
                        if "detail" in error_data:
                            detail = error_data["detail"]
                            if "status" in detail and detail["status"] == "missing_permissions":
                                print(f"Permission error: {detail.get('message', 'Unknown permission error')}")
                                if "text_to_speech" in str(detail):
                                    self.elevenlabs_failed = True
                                    return self.fallback_tts(text)
                    except:
                        pass
                        
                    # Try the client library as fallback
                    print("Trying alternative method...")
                
            except ImportError:
                print("Requests library not available, falling back to client library")
                # Fall back to using the client library if requests is not available
            except Exception as e:
                print(f"Direct API request failed: {e}")
                print("Trying client library instead...")
            
            # Create ElevenLabs client with the API key
            client = ElevenLabs(api_key=self.elevenlabs_api_key)
            
            # Generate audio using the client directly
            try:
                # First try the direct method that returns bytes directly
                audio_data = client.text_to_speech.generate(
                    text=text,
                    voice_id=voice_id,
                    model_id=self.elevenlabs_model_id,
                    voice_settings={
                        "stability": 0.5,
                        "similarity_boost": 0.5
                    },
                    output_format="mp3"
                )
                
                # If this succeeds, audio_data should be bytes already
                if isinstance(audio_data, bytes):
                    print("Successfully generated speech with ElevenLabs client")
                    return audio_data
                
                # If we got here, we need to handle a different return type
                print(f"Using alternative handling for response type: {type(audio_data)}")
                
            except (AttributeError, TypeError) as e:
                print(f"Using alternative API method: {e}")
                # Fall back to the streaming method
                pass
                
            # Fall back to the streaming method that returns a generator
            audio_generator = client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id=self.elevenlabs_model_id,
                voice_settings={
                    "stability": 0.5,
                    "similarity_boost": 0.5
                }
            )
            
            # Convert the generator to bytes
            audio_chunks = bytearray()
            try:
                # First try treating it as a regular generator
                for chunk in audio_generator:
                    if isinstance(chunk, bytes):
                        audio_chunks.extend(chunk)
                    else:
                        # If it's not bytes, try to convert it
                        audio_chunks.extend(bytes(chunk))
            except TypeError:
                # If audio_generator is not directly iterable, it might be a response object
                # Try getting content or read() method
                if hasattr(audio_generator, 'content'):
                    audio_chunks.extend(audio_generator.content)
                elif hasattr(audio_generator, 'read'):
                    audio_chunks.extend(audio_generator.read())
                elif hasattr(audio_generator, 'get_bytes'):
                    audio_chunks.extend(audio_generator.get_bytes())
                else:
                    # As a last resort, try converting the entire object
                    audio_chunks.extend(bytes(audio_generator))
            
            # Convert to bytes
            audio_content = bytes(audio_chunks)
            
            # Make sure we actually got something substantial (audio files should be at least 1KB)
            if len(audio_content) < 1000:
                print(f"Warning: Generated audio seems too small ({len(audio_content)} bytes)")
                return self.fallback_tts(text)
            
            print("Successfully generated speech with ElevenLabs client")
            return audio_content
                
        except Exception as e:
            error_str = str(e)
            print(f"Exception generating speech with ElevenLabs: {e}")
            
            # Check if this is a permissions or authentication issue
            if ("401" in error_str or 
                "status_code: 401" in error_str or 
                "missing_permissions" in error_str or 
                "Unauthorized" in error_str or
                "text_to_speech" in error_str):
                
                print("\nNOTE: Your ElevenLabs API key doesn't have the required text_to_speech permission.")
                print("This is common for free accounts or if your subscription has expired.")
                print("Switching to fallback TTS system for the rest of this session.")
                
                # Mark ElevenLabs as failed to avoid repeated attempts
                self.elevenlabs_failed = True
                
                # Use fallback TTS
                return self.fallback_tts(text)
            
            # For other errors, still try fallback
            return self.fallback_tts(text)
    
    def fallback_tts(self, text):
        """Fallback TTS system using pyttsx3 when ElevenLabs is not available"""
        try:
            if not self.has_fallback_tts or self.tts_engine is None:
                # Try to initialize pyttsx3 again
                try:
                    self.tts_engine = pyttsx3.init()
                    self.tts_engine.setProperty('rate', self._parse_env_int("VOICE_RATE", 150))
                    self.tts_engine.setProperty('volume', self._parse_env_float("VOICE_VOLUME", 0.8))
                    self.has_fallback_tts = True
                except Exception as e:
                    print(f"Cannot initialize fallback TTS: {e}")
                    print("WARNING: No TTS system available - text will not be spoken")
                    return None
            
            print(f"Using fallback TTS system for text: {text[:50]}...")
            
            # Create a temporary file for the audio
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file_path = temp_file.name
            
            # Save speech to a temporary WAV file
            self.tts_engine.save_to_file(text, temp_file_path)
            self.tts_engine.runAndWait()
            
            # Read the file back as bytes
            with open(temp_file_path, 'rb') as f:
                audio_data = f.read()
            
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except:
                pass
            
            print("Successfully generated speech using fallback TTS")
            return audio_data
            
        except Exception as e:
            print(f"Error in fallback TTS: {e}")
            return None
    
    def play_audio(self, audio_data):
        """Play audio data with interrupt capability"""
        if not audio_data:
            print("No audio data to play, continuing with silent operation")
            return True  # Return success so the conversation can continue
            
        try:
            # Write audio data to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            # Play the audio file using the system's default player
            if platform.system() == 'Windows':
                os.system(f'start {temp_file_path}')
            elif platform.system() == 'Darwin':  # macOS
                os.system(f'afplay {temp_file_path}')
            else:  # Linux and others
                os.system(f'mpg123 {temp_file_path}')
            
            # Clean up the temporary file after a delay
            def cleanup_temp_file():
                time.sleep(5)  # Wait for the audio to finish playing
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
            
            cleanup_thread = threading.Thread(target=cleanup_temp_file)
            cleanup_thread.daemon = True
            cleanup_thread.start()
            
            # Store the player thread for interrupt handling
            self.audio_player = cleanup_thread
            
            return True
        except Exception as e:
            print(f"Error playing audio: {e}")
            return False


def check_env_file():
    """Check if .env file exists and contains required keys"""
    env_file_valid = False
    
    # Try to load from .env file
    try:
        if os.path.exists(".env"):
            load_dotenv()
            env_file_valid = True
    except Exception as e:
        print(f"Error loading .env file: {e}")
        print("Setting environment variables directly in code...")
    
    # If .env file doesn't exist or had issues, set environment variables directly
    if not env_file_valid:
        print("Setting required environment variables directly...")
        # Set essential environment variables
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
        os.environ["PROCESS_EVERY_N_FRAMES"] = "15"
        
        # Try to create a valid .env file for future use
        try:
            with open(".env", "w", encoding="utf-8") as f:
                f.write("""# API Keys
PORCUPINE_ACCESS_KEY=qqlP6xCMkzy3yWVx9Wg3RDsATOG1d06E1KAgbFilHWeoAl3zcIjkag==
GEMINI_API_KEY=AIzaSyBuFAaIvXFRRX_LfAaTFnVTFFva-eV2Zw8
ELEVENLABS_API_KEY=sk_a815878bc3184834c55fe90e89c9588bcb96759e64d9cb61
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM

# Wake word settings
WAKE_WORD=alexa
CUSTOM_WAKE_WORDS=jarvis,computer,hey google

# Memory settings
USE_MEMORY=true
MEMORY_SIZE=10
MEMORY_FILE=robot_memory.json
MEMORY_EXCHANGES_LIMIT=5

# Voice settings
VOICE_RATE=150
VOICE_VOLUME=0.8

# Other settings
SAVE_HISTORY=true
ENABLE_BEEP=true
SPEECH_RECOGNITION=google
USE_EMOTION_DETECTION=true
SHOW_WEBCAM=false
PROCESS_EVERY_N_FRAMES=15
""")
            print("Created a new .env file with proper encoding.")
        except Exception as e:
            print(f"Could not create new .env file: {e}")
            print("Using environment variables from code instead.")
    
    # Check if Porcupine access key is set
    porcupine_key = os.environ.get("PORCUPINE_ACCESS_KEY")
    if not porcupine_key or porcupine_key == "" or "your_porcupine_access_key_here" in porcupine_key:
        print("\nERROR: Porcupine access key is not set correctly.")
        print("You need to get a free access key from https://console.picovoice.ai/")
        print("Then set it in your .env file as PORCUPINE_ACCESS_KEY=your_key_here\n")
        return False
    
    return True


if __name__ == "__main__":
    print("Starting Conversation Robot...")
    
    # Check if .env file exists and contains required keys
    if not check_env_file():
        print("Please set up your .env file and try again.")
        exit(1)
    
    # List available default wake words
    print(f"Available default wake words: {', '.join(DEFAULT_WAKE_WORDS)}")
    
    # Get settings from environment variables
    # Use the robot's parsing functions to handle comments in env vars
    robot_temp = ConversationRobot()  # Create a temporary instance just to use the parsing functions
    wake_word = robot_temp._parse_env_str("WAKE_WORD", "computer")
    save_history = robot_temp._parse_env_bool("SAVE_HISTORY", False)
    voice_id = robot_temp._parse_env_str("VOICE_ID")
    voice_rate = robot_temp._parse_env_int("VOICE_RATE", 200)
    voice_volume = robot_temp._parse_env_float("VOICE_VOLUME", 1.0)
    use_emotion_detection = robot_temp._parse_env_bool("USE_EMOTION_DETECTION", True)
    show_webcam = robot_temp._parse_env_bool("SHOW_WEBCAM", False)
    
    # Check for list voices flag
    if robot_temp._parse_env_bool("LIST_VOICES", False):
        temp_engine = pyttsx3.init()
        voices = temp_engine.getProperty('voices')
        print("\nAvailable voices:")
        for i, voice in enumerate(voices):
            print(f"{i}: {voice.name} ({voice.id})")
        print()
    
    try:
        # Create and run the robot
        robot = ConversationRobot(
            wake_word=wake_word,
            save_history=save_history,
            voice_id=voice_id,
            rate=voice_rate,
            volume=voice_volume,
            use_emotion_detection=use_emotion_detection,
            show_webcam=show_webcam
        )
        
        # Test beep sounds before running
        if robot_temp._parse_env_bool("ENABLE_BEEP", True):
            print("Testing beep functionality before starting...")
            robot.test_beep_sounds()
        
        robot.run()
    except ValueError as e:
        print(f"Error: {e}")
        print("Please fix the issues and try again.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        print("Please check your configuration and try again.") 