import requests
import json

# The API key you provided
api_key = "sk_a815878bc3184834c55fe90e89c9588bcb96759e64d9cb61"

# Test the models endpoint (requires minimal permissions)
def test_models():
    print("Testing models endpoint...")
    headers = {"xi-api-key": api_key}
    response = requests.get("https://api.elevenlabs.io/v1/models", headers=headers)
    
    print(f"Status code: {response.status_code}")
    if response.status_code == 200:
        print("SUCCESS! API key is valid")
        data = response.json()
        # Handle the case where response is a list directly
        if isinstance(data, list):
            print(f"Found {len(data)} models")
            # List available models
            if data:
                print("Available models:")
                for i, model in enumerate(data):
                    model_id = model.get('model_id', 'unknown')
                    name = model.get('name', 'Unnamed model')
                    print(f"  {i+1}. {name} (ID: {model_id})")
        else:
            print(f"Found {len(data.get('models', []))} models")
    else:
        print(f"ERROR: {response.text}")
    
    return response.status_code == 200

# Test the voices endpoint
def test_voices():
    print("\nTesting voices endpoint...")
    headers = {"xi-api-key": api_key}
    response = requests.get("https://api.elevenlabs.io/v1/voices", headers=headers)
    
    print(f"Status code: {response.status_code}")
    if response.status_code == 200:
        print("SUCCESS! API key has voice listing permissions")
        data = response.json()
        voices = data.get('voices', [])
        print(f"Found {len(voices)} voices")
        
        # Return the first voice ID if available
        if voices:
            print("Available voices:")
            for i, voice in enumerate(voices):
                voice_id = voice.get('voice_id', 'unknown')
                name = voice.get('name', f'Voice {i+1}')
                print(f"  {i+1}. {name} (ID: {voice_id})")
            return voices[0].get('voice_id')
    else:
        print(f"ERROR: {response.text}")
    
    # Return None if no voices were found or there was an error
    return None

# List available subscription tiers
def test_subscription():
    print("\nChecking subscription status...")
    headers = {"xi-api-key": api_key}
    response = requests.get("https://api.elevenlabs.io/v1/user/subscription", headers=headers)
    
    print(f"Status code: {response.status_code}")
    if response.status_code == 200:
        print("SUCCESS! API key has user subscription permissions")
        try:
            data = response.json()
            tier = data.get('tier', 'unknown')
            print(f"Subscription tier: {tier}")
            
            # Check character limits
            character_count = data.get('character_count', 0)
            character_limit = data.get('character_limit', 0)
            print(f"Character usage: {character_count}/{character_limit}")
            
            return True
        except Exception as e:
            print(f"Error parsing subscription data: {e}")
    else:
        print(f"ERROR: {response.text}")
    
    return False

# Test text-to-speech endpoint
def test_tts(voice_id=None):
    print("\nTesting text-to-speech endpoint...")
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg"  # Explicitly request MP3 format
    }
    
    # Default ElevenLabs voice ID if none provided
    if not voice_id:
        voice_id = "21m00Tcm4TlvDq8ikWAM"  # Default voice
    
    # Request data - testing different models
    models_to_try = [
        "eleven_monolingual_v1",
        "eleven_multilingual_v2",
        "eleven_multilingual_v1",
        "eleven_turbo_v2",
    ]
    
    success = False
    audio_content = None
    
    for model_id in models_to_try:
        if success:
            break
            
        print(f"\nTrying model: {model_id}")
        data = {
            "text": "This is a test of the ElevenLabs API.",
            "model_id": model_id,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }
        
        response = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            headers=headers,
            json=data
        )
        
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            print(f"SUCCESS! API key has text-to-speech permissions with model {model_id}")
            audio_content = response.content
            print(f"Received {len(audio_content)} bytes of audio data")
            success = True
        else:
            try:
                error_text = response.text
                print(f"ERROR: {error_text}")
                
                # Try to parse the error for more details
                error_data = json.loads(error_text)
                if "detail" in error_data:
                    detail = error_data["detail"]
                    if isinstance(detail, dict) and "message" in detail:
                        print(f"Error message: {detail['message']}")
            except:
                print("Could not parse error response")
    
    # Save the audio if we got any
    if success and audio_content:
        # Save the audio to a file
        with open("test_output.mp3", "wb") as f:
            f.write(audio_content)
        print("Saved audio to test_output.mp3")
    
    return success

if __name__ == "__main__":
    print("ElevenLabs API Key Test")
    print("=======================")
    print(f"Using API key: {api_key[:5]}...{api_key[-5:]}")
    
    models_ok = test_models()
    voice_id = test_voices()  # This will return a voice ID if available
    subscription_ok = test_subscription()
    
    # Use the discovered voice ID if available
    tts_ok = test_tts(voice_id)
    
    print("\nSummary:")
    print(f"Models API:        {'✅ WORKING' if models_ok else '❌ FAILED'}")
    print(f"Voices API:        {'✅ WORKING' if voice_id else '❌ FAILED'}")
    print(f"Subscription API:  {'✅ WORKING' if subscription_ok else '❌ FAILED'}")
    print(f"Text-to-Speech:    {'✅ WORKING' if tts_ok else '❌ FAILED'}")
    
    if not tts_ok:
        print("\nYour API key does not have text-to-speech permissions.")
        print("This is common for free accounts or if your subscription has expired.")
        print("Please visit https://elevenlabs.io/ to upgrade your account or get a new API key.") 