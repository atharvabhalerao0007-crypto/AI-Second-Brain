from gtts import gTTS
import tempfile

def speak_text(text):

    tts = gTTS(text)

    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")

    tts.save(temp_audio.name)

    return temp_audio.name