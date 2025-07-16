#!/usr/bin/env python3
import os
import sys
import time
import platform
import tempfile
import requests
import argparse

# Try to import pyttsx3 for fallback TTS
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    print("pyttsx3 not available. Installing...")
    import subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyttsx3"])
        import pyttsx3
        PYTTSX3_AVAILABLE = True
        print("pyttsx3 installed successfully.")
    except Exception as e:
        print(f"Failed to install pyttsx3: {e}")
        print("Fallback TTS will not be available.")

# Try to import ElevenLabs client
try:
    from elevenlabs.client import ElevenLabs
    ELEVENLABS_CLIENT_AVAILABLE = True
except ImportError:
    ELEVENLABS_CLIENT_AVAILABLE = False
    print("ElevenLabs client library not found. Installing...")
    import subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "elevenlabs"])
        from elevenlabs.client import ElevenLabs
        ELEVENLABS_CLIENT_AVAILABLE = True
        print("ElevenLabs client library installed successfully.")
    except Exception as e:
        print(f"Failed to install ElevenLabs client: {e}")
        print("ElevenLabs TTS will not be available.")

def test_elevenlabs_api(api_key, voice_id="21m00Tcm4TlvDq8ikWAM", model_id="eleven_monolingual_v1"):
    """Test the ElevenLabs API with the provided key"""
    print("\n=== Testing ElevenLabs API ===")
    
    if not api_key:
        print("❌ No API key provided")
        return False
    
    # First test the models endpoint
    print("Testing models endpoint...")
    headers = {
        "xi-api-key": api_key
    }
    
    try:
        response = requests.get("https://api.elevenlabs.io/v1/models", headers=headers)
        
        if response.status_code == 200:
            print("✅ Models endpoint test successful")
            models = response.json()
            if models:
                print(f"Available models: {', '.join([m.get('model_id', 'unknown') for m in models])}")
        else:
            print(f"❌ Models endpoint test failed: Status {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error testing models endpoint: {e}")
        return False
    
    # Then test the text-to-speech endpoint
    print("\nTesting text-to-speech endpoint...")
    tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    tts_headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }
    tts_data = {
        "text": "This is a test of the ElevenLabs text to speech API.",
        "model_id": model_id
    }
    
    try:
        tts_response = requests.post(tts_url, headers=tts_headers, json=tts_data)
        
        if tts_response.status_code == 200:
            print("✅ Text-to-speech endpoint test successful")
            audio_data = tts_response.content
            print(f"Received {len(audio_data)} bytes of audio data")
            
            # Save to temp file and play
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                temp_file_path = temp_file.name
                temp_file.write(audio_data)
            
            print(f"Audio saved to {temp_file_path}")
            play_audio(temp_file_path)
            
            return True
        else:
            print(f"❌ Text-to-speech endpoint test failed: Status {tts_response.status_code}")
            try:
                error_data = tts_response.json()
                if "detail" in error_data:
                    detail = error_data["detail"]
                    if isinstance(detail, dict) and "status" in detail:
                        print(f"Error status: {detail['status']}")
                        print(f"Error message: {detail.get('message', 'No message')}")
                        
                        if detail["status"] == "detected_unusual_activity":
                            print("\n⚠️ Unusual activity detected on your ElevenLabs account")
                            print("This is common for free accounts or if your subscription has restrictions.")
                            print("You may need to upgrade your subscription or contact ElevenLabs support.")
            except:
                print(f"Response: {tts_response.text}")
            
            return False
    except Exception as e:
        print(f"❌ Error testing text-to-speech endpoint: {e}")
        return False

def test_elevenlabs_client(api_key, voice_id="21m00Tcm4TlvDq8ikWAM", model_id="eleven_monolingual_v1"):
    """Test the ElevenLabs client library"""
    print("\n=== Testing ElevenLabs Client Library ===")
    
    if not ELEVENLABS_CLIENT_AVAILABLE:
        print("❌ ElevenLabs client library not available")
        return False
    
    try:
        print("Creating ElevenLabs client...")
        client = ElevenLabs(api_key=api_key)
        
        # Test text-to-speech
        print("Testing text-to-speech with client...")
        try:
            audio_data = client.text_to_speech.generate(
                text="This is a test of the ElevenLabs client library.",
                voice_id=voice_id,
                model_id=model_id,
                voice_settings={
                    "stability": 0.5,
                    "similarity_boost": 0.5
                },
                output_format="mp3"
            )
            
            if isinstance(audio_data, bytes) and len(audio_data) > 1000:
                print("✅ Successfully generated speech with ElevenLabs client")
                print(f"Received {len(audio_data)} bytes of audio data")
                
                # Save to temp file and play
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                    temp_file_path = temp_file.name
                    temp_file.write(audio_data)
                
                print(f"Audio saved to {temp_file_path}")
                play_audio(temp_file_path)
                
                return True
            else:
                print(f"❌ Generated audio seems too small or invalid: {type(audio_data)}")
                return False
        except Exception as e:
            print(f"❌ Error generating speech with client: {e}")
            
            # Try alternative method
            print("Trying alternative client method...")
            try:
                audio_generator = client.text_to_speech.convert(
                    text="This is a test using the alternative client method.",
                    voice_id=voice_id,
                    model_id=model_id
                )
                
                # Convert the generator to bytes
                audio_chunks = bytearray()
                for chunk in audio_generator:
                    if isinstance(chunk, bytes):
                        audio_chunks.extend(chunk)
                    else:
                        audio_chunks.extend(bytes(chunk))
                
                audio_content = bytes(audio_chunks)
                
                if len(audio_content) > 1000:
                    print("✅ Successfully generated speech with alternative client method")
                    print(f"Received {len(audio_content)} bytes of audio data")
                    
                    # Save to temp file and play
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                        temp_file_path = temp_file.name
                        temp_file.write(audio_content)
                    
                    print(f"Audio saved to {temp_file_path}")
                    play_audio(temp_file_path)
                    
                    return True
                else:
                    print(f"❌ Generated audio seems too small: {len(audio_content)} bytes")
                    return False
            except Exception as e2:
                print(f"❌ Alternative method also failed: {e2}")
                return False
    except Exception as e:
        print(f"❌ Error creating ElevenLabs client: {e}")
        return False

def test_pyttsx3():
    """Test the pyttsx3 fallback TTS"""
    print("\n=== Testing pyttsx3 Fallback TTS ===")
    
    if not PYTTSX3_AVAILABLE:
        print("❌ pyttsx3 not available")
        return False
    
    try:
        print("Initializing pyttsx3...")
        engine = pyttsx3.init()
        
        # Get available voices
        voices = engine.getProperty('voices')
        print(f"Available voices: {len(voices)}")
        for i, voice in enumerate(voices):
            print(f"  {i}: {voice.name} ({voice.id})")
        
        # Set properties
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.8)
        
        # Test direct speech
        print("\nTesting direct speech...")
        engine.say("This is a test of the pyttsx3 text to speech engine.")
        engine.runAndWait()
        
        # Test saving to file
        print("\nTesting saving speech to file...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file_path = temp_file.name
        
        engine.save_to_file("This is a test of saving speech to a file.", temp_file_path)
        engine.runAndWait()
        
        print(f"Audio saved to {temp_file_path}")
        play_audio(temp_file_path)
        
        return True
    except Exception as e:
        print(f"❌ Error testing pyttsx3: {e}")
        return False

def play_audio(file_path):
    """Play an audio file using the system's default player"""
    print(f"Playing audio file: {file_path}")
    
    try:
        if platform.system() == 'Windows':
            os.system(f'start {file_path}')
        elif platform.system() == 'Darwin':  # macOS
            os.system(f'afplay {file_path}')
        else:  # Linux and others
            # Try multiple players in order of preference
            players = [
                f'mpg123 "{file_path}"',
                f'mplayer "{file_path}"',
                f'aplay "{file_path}"',
                f'ffplay -nodisp -autoexit "{file_path}"',
                f'python3 -m playsound "{file_path}"'
            ]
            
            success = False
            for player_cmd in players:
                print(f"Trying to play audio with: {player_cmd}")
                exit_code = os.system(player_cmd)
                if exit_code == 0:
                    print(f"Successfully played audio with: {player_cmd}")
                    success = True
                    break
                else:
                    print(f"Failed to play with {player_cmd.split()[0]}, trying next player...")
            
            if not success:
                print("All audio players failed. Check your audio setup.")
                return False
        
        return True
    except Exception as e:
        print(f"Error playing audio: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test TTS functionality")
    parser.add_argument("--api-key", help="ElevenLabs API key")
    parser.add_argument("--voice-id", default="21m00Tcm4TlvDq8ikWAM", help="ElevenLabs voice ID")
    parser.add_argument("--model-id", default="eleven_monolingual_v1", help="ElevenLabs model ID")
    parser.add_argument("--text", default="This is a test of the text to speech functionality.", help="Text to convert to speech")
    parser.add_argument("--skip-elevenlabs", action="store_true", help="Skip ElevenLabs tests")
    parser.add_argument("--skip-pyttsx3", action="store_true", help="Skip pyttsx3 tests")
    
    args = parser.parse_args()
    
    print("TTS Test Script")
    print("==============")
    
    # Get API key from arguments or environment
    api_key = args.api_key or os.environ.get("ELEVENLABS_API_KEY")
    
    if not api_key and not args.skip_elevenlabs:
        print("No ElevenLabs API key provided. Please provide one with --api-key or set the ELEVENLABS_API_KEY environment variable.")
        api_key = input("Enter your ElevenLabs API key (or press Enter to skip ElevenLabs tests): ")
        if not api_key:
            args.skip_elevenlabs = True
    
    # Test ElevenLabs API
    if not args.skip_elevenlabs:
        print("\nTesting ElevenLabs...")
        elevenlabs_api_success = test_elevenlabs_api(api_key, args.voice_id, args.model_id)
        
        if elevenlabs_api_success:
            print("\nTesting ElevenLabs client library...")
            test_elevenlabs_client(api_key, args.voice_id, args.model_id)
    
    # Test pyttsx3
    if not args.skip_pyttsx3:
        print("\nTesting pyttsx3 fallback TTS...")
        test_pyttsx3()
    
    print("\nTTS tests completed.")

if __name__ == "__main__":
    main() 