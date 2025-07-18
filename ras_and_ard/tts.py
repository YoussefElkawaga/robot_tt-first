from gtts import gTTS
import os

tts = gTTS(text="hello guys can you hear me very well ? without any problem ", lang='en')
tts.save("example.mp3")
os.system("start example.mp3")  # Windows
