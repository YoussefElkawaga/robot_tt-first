from elevenlabs import generate, play, save, set_api_key

# الخطوة 1: إعداد مفتاح API المجاني
set_api_key("sk_57580b4b142606a6e53249d0a3b105fe4ada6a1ae68f6b2b")

# الخطوة 2: توليد الصوت من نص معين
audio = generate(
    text="مرحبًا بك! هذا مثال باستخدام ElevenLabs API المجاني.",
    voice="Rachel",  # صوت مجاني متاح في الحساب المجاني
    model="eleven_monolingual_v1"  # هذا النموذج متاح في الحساب المجاني
)

# الخطوة 3: تشغيل الصوت فورًا
play(audio)

# الخطوة 4: حفظ الصوت إلى ملف MP3 (اختياري)
save(audio, "output.mp3")
