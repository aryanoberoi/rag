from openai import OpenAI
from pathlib import Path
client=OpenAI()
speech_file_path = Path(__file__).parent / "speech.mp3"
response = client.audio.speech.create(
  model="tts-1",
  voice="alloy",
  input="Good morning have a beautiful day"
)

response.stream_to_file(speech_file_path)

def speechtotext(file):
    audio_file= open(file, "rb")
    transcription = client.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_file, 
    response_format="text"  
    )
    return str(transcription)

hello=speechtotext("hello.m4a")
print(hello)