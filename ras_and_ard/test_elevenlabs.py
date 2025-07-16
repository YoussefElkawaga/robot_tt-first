import requests
import sys
import json

# Your ElevenLabs API key
API_KEY = "sk_57580b4b142606a6e53249d0a3b105fe4ada6a1ae68f6b2b"

print(f"Testing ElevenLabs API with key: {API_KEY[:5]}...{API_KEY[-5:]}")

# Test 1: Check API status with voices endpoint
print("\nTest 1: Checking API status with voices endpoint...")
try:
    headers = {"xi-api-key": API_KEY}
    response = requests.get("https://api.elevenlabs.io/v1/voices", headers=headers)
    
    print(f"Status code: {response.status_code}")
    if response.status_code == 200:
        print("✅ Voices endpoint test passed")
        data = response.json()
        if "voices" in data:
            print(f"Available voices: {len(data['voices'])}")
            for voice in data["voices"]:
                print(f"  - {voice.get('name', 'Unknown')} (ID: {voice.get('voice_id', 'Unknown')})")
    else:
        print(f"❌ Voices endpoint test failed: {response.text}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 2: Check API status with models endpoint
print("\nTest 2: Checking API status with models endpoint...")
try:
    headers = {"xi-api-key": API_KEY}
    response = requests.get("https://api.elevenlabs.io/v1/models", headers=headers)
    
    print(f"Status code: {response.status_code}")
    if response.status_code == 200:
        print("✅ Models endpoint test passed")
        models = response.json()
        for model in models:
            print(f"  - {model.get('name', 'Unknown')} (ID: {model.get('model_id', 'Unknown')})")
    else:
        print(f"❌ Models endpoint test failed: {response.text}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 3: Check user subscription information
print("\nTest 3: Checking user subscription information...")
try:
    headers = {"xi-api-key": API_KEY}
    response = requests.get("https://api.elevenlabs.io/v1/user/subscription", headers=headers)
    
    print(f"Status code: {response.status_code}")
    if response.status_code == 200:
        print("✅ Subscription endpoint test passed")
        subscription = response.json()
        print(f"Subscription tier: {subscription.get('tier', 'Unknown')}")
        print(f"Character quota: {subscription.get('character_count', 'Unknown')} / {subscription.get('character_limit', 'Unknown')}")
        
        # Check if the account is rate limited
        if "status" in subscription and subscription["status"] == "rate_limited":
            print("❌ Your account is currently rate limited")
    else:
        print(f"❌ Subscription endpoint test failed: {response.text}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 4: Try a simple text-to-speech request
print("\nTest 4: Testing text-to-speech functionality...")
try:
    headers = {
        "xi-api-key": API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg"
    }
    
    # Use a default voice ID (Rachel)
    voice_id = "21m00Tcm4TlvDq8ikWAM"
    
    data = {
        "text": "This is a test of the ElevenLabs API.",
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }
    
    print(f"Sending TTS request with voice ID: {voice_id}")
    response = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
        headers=headers,
        json=data
    )
    
    print(f"Status code: {response.status_code}")
    if response.status_code == 200:
        print("✅ Text-to-speech test passed")
        print(f"Received {len(response.content)} bytes of audio data")
        
        # Save the audio to a file
        with open("test_tts.mp3", "wb") as f:
            f.write(response.content)
        print("Audio saved to test_tts.mp3")
    else:
        print(f"❌ Text-to-speech test failed: {response.text}")
        
        # Try to parse the error for more details
        try:
            error_data = response.json()
            if "detail" in error_data:
                detail = error_data["detail"]
                if isinstance(detail, dict) and "status" in detail:
                    print(f"Error status: {detail['status']}")
                    print(f"Error message: {detail.get('message', 'No message provided')}")
                    
                    # Check for specific error types
                    if detail["status"] == "detected_unusual_activity":
                        print("\n⚠️ Your account has been flagged for unusual activity.")
                        print("This often happens with free accounts that are used with multiple IPs or VPNs.")
                        print("You may need to contact ElevenLabs support or upgrade to a paid plan.")
        except:
            pass
except Exception as e:
    print(f"❌ Error: {e}")

print("\nTest complete. Check the results above to diagnose your ElevenLabs API issues.") 