import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import whisper
def transcribe_audio(audio_path):
    model = whisper.load_model("turbo")
    result = model.transcribe(audio_path)
    print(result["text"])
if __name__ == "__main__":
    audio_path = './sources/bahubali/chunks/audios/audio_0_20.mp3'  # Change this to the path of your audio file
    transcribe_audio(audio_path)