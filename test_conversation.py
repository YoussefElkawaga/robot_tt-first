#!/usr/bin/env python3
"""
Test script for conversation robot without wake word detection.
This is useful for testing the AI and TTS functionality without needing a microphone.
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai
import pyttsx3
import requests

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize Gemini AI
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        print("Please create a .env file with your Gemini API key.")
        return
    
    # Configure Gemini
    genai.configure(api_key=gemini_api_key)
    
    # Try to initialize with gemini-2.0-flash model
    try:
        print("Initializing Gemini AI with model: gemini-2.0-flash")
        model = genai.GenerativeModel('gemini-2.0-flash')
        chat_session = model.start_chat(history=[])
    except Exception as e:
        print(f"Error initializing Gemini model 'gemini-2.0-flash': {e}")
        print("Trying fallback model 'gemini-pro'...")
        try:
            model = genai.GenerativeModel('gemini-pro')
            chat_session = model.start_chat(history=[])
        except Exception as e2:
            print(f"Error initializing fallback model: {e2}")
            print("Available models:")
            try:
                for model in genai.list_models():
                    print(f"- {model.name}")
            except:
                print("Could not list available models")
    
    # Initialize text-to-speech
    tts_engine = pyttsx3.init()
    
    # Configure voice settings
    voice_id = os.getenv("VOICE_ID")
    voice_rate = os.getenv("VOICE_RATE")
    voice_volume = os.getenv("VOICE_VOLUME")
    
    if voice_id:
        voices = tts_engine.getProperty('voices')
        try:
            voice_id = int(voice_id) if voice_id.isdigit() else voice_id
            
            if isinstance(voice_id, int) and 0 <= voice_id < len(voices):
                tts_engine.setProperty('voice', voices[voice_id].id)
                print(f"Set voice to index {voice_id}: {voices[voice_id].name}")
            else:
                # Try to find voice by ID or name
                for v in voices:
                    if voice_id in v.id or voice_id.lower() in v.name.lower():
                        tts_engine.setProperty('voice', v.id)
                        print(f"Set voice to: {v.name}")
                        break
        except Exception as e:
            print(f"Error setting voice: {e}")
    
    if voice_rate:
        try:
            rate_value = int(voice_rate)
            tts_engine.setProperty('rate', rate_value)
            print(f"Set speech rate to: {rate_value}")
        except Exception as e:
            print(f"Error setting speech rate: {e}")
    
    if voice_volume:
        try:
            volume_value = float(voice_volume)
            if 0.0 <= volume_value <= 1.0:
                tts_engine.setProperty('volume', volume_value)
                print(f"Set speech volume to: {volume_value}")
        except Exception as e:
            print(f"Error setting speech volume: {e}")
    
    # List available voices
    if os.getenv("LIST_VOICES", "false").lower() == "true":
        voices = tts_engine.getProperty('voices')
        print("\nAvailable voices:")
        for i, voice in enumerate(voices):
            print(f"{i}: {voice.name} ({voice.id})")
        print()
    
    print("\n=== Conversation Robot Test Mode ===")
    print("Type your messages and the AI will respond.")
    print("Type 'exit', 'quit', or press Ctrl+C to end the conversation.\n")
    
    try:
        while True:
            # Get user input
            user_input = input("You: ")
            
            # Check for exit command
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting conversation.")
                break
            
            # Get AI response
            try:
                print("Sending request to Gemini AI...")
                response = chat_session.send_message(user_input)
                print("Received response from Gemini AI")
                ai_response = response.text
                print(f"Robot: {ai_response}")
                
                # Speak the response
                tts_engine.say(ai_response)
                tts_engine.runAndWait()
            except Exception as e:
                print(f"Error getting AI response: {e}")
                
                # Try direct API request as fallback
                try:
                    print("Attempting direct API request as fallback...")
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_api_key}"
                    headers = {"Content-Type": "application/json"}
                    data = {
                        "contents": [
                            {
                                "parts": [
                                    {
                                        "text": user_input
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
                                    ai_response = parts[0]["text"]
                                    print(f"Robot: {ai_response}")
                                    tts_engine.say(ai_response)
                                    tts_engine.runAndWait()
                                    continue
                    
                    print(f"Fallback API request failed: {response.status_code}")
                    print(response.text)
                    ai_response = "I'm having trouble connecting to my brain right now. Please try again later."
                    print(f"Robot: {ai_response}")
                    tts_engine.say(ai_response)
                    tts_engine.runAndWait()
                except Exception as e2:
                    print(f"Error in fallback API request: {e2}")
                    ai_response = "I'm having trouble connecting to my brain right now. Please try again later."
                    print(f"Robot: {ai_response}")
                    tts_engine.say(ai_response)
                    tts_engine.runAndWait()
    
    except KeyboardInterrupt:
        print("\nConversation ended by user.")

if __name__ == "__main__":
    main() 