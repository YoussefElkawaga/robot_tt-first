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
                rate_value = int(rate)
                self.tts_engine.setProperty('rate', rate_value)
                print(f"Set speech rate to: {rate_value}")
            except Exception as e:
                print(f"Error setting speech rate: {e}")
        
        # Set volume if specified (default is 1.0)
        if volume:
            try:
                volume_value = float(volume)
                if 0.0 <= volume_value <= 1.0:
                    self.tts_engine.setProperty('volume', volume_value)
                    print(f"Set speech volume to: {volume_value}")
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
            self.video_capture = cv2.VideoCapture(0)
            if not self.video_capture.isOpened():
                print("Could not open video capture device")
                return
            
            # Load face detection model
            try:
                self.face_cascade = cv2.CascadeClassifier(self.face_cascade_path)
                if self.face_cascade.empty():
                    print("Error: Face cascade classifier not loaded properly")
                    self.video_capture.release()
                    return
            except Exception as e:
                print(f"Error loading face detection model: {e}")
                self.video_capture.release()
                return
            
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
        """Main loop for emotion detection"""
        # Define possible emotions
        emotions = ["neutral", "happy", "surprise", "sad", "angry", "disgust", "fear"]
        
        frame_counter = 0
        emotion_change_interval = 30  # Change emotion every 30 processed frames
        
        while self.emotion_detection_active:
            try:
                ret, frame = self.video_capture.read()
                if not ret:
                    continue
                
                # Process every 5 frames to reduce CPU usage
                frame_counter += 1
                if frame_counter % 5 != 0:
                    time.sleep(0.05)
                    continue
                
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
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
                    
                    # Extract the face ROI
                    face_roi = gray[y:y+h, x:x+w]
                    
                    # Simple emotion "detection" based on timing and face properties
                    if frame_counter % emotion_change_interval == 0:
                        # Calculate simple metrics from the face
                        avg_brightness = np.mean(face_roi)
                        face_width_height_ratio = w / h if h > 0 else 1
                        
                        # Use these metrics to bias the emotion selection
                        # Higher brightness might indicate happiness, lower might indicate sadness
                        # Different width/height ratios might indicate different expressions
                        if avg_brightness > 150 and face_width_height_ratio > 0.95:
                            bias_emotions = ["happy", "neutral", "surprise"]
                        elif avg_brightness < 100:
                            bias_emotions = ["sad", "angry", "fear"]
                        elif face_width_height_ratio > 1.1:
                            bias_emotions = ["surprise", "fear"]
                        elif face_width_height_ratio < 0.85:
                            bias_emotions = ["sad", "disgust"]
                        else:
                            bias_emotions = ["neutral", "happy"]
                        
                        # Select an emotion
                        detected_emotion = random.choice(bias_emotions)
                        
                        if detected_emotion != self.last_emotion:
                            self.last_emotion = detected_emotion
                            print(f"Detected emotion: {detected_emotion}")
                
                # If no faces are detected, occasionally change to neutral
                elif frame_counter % 50 == 0 and self.last_emotion != "neutral":
                    self.last_emotion = "neutral"
                    print("No face detected, emotion set to neutral")
                
                # Sleep to reduce CPU usage
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error in emotion detection: {e}")
                time.sleep(1)
    
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