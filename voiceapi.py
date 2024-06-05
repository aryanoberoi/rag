from flask import Flask, request, jsonify,json
from flask_restful import Resource, Api
from pydub import AudioSegment
from gtts import gTTS
import os
import base64
from openai import OpenAI
app = Flask(__name__)
client=OpenAI()
def speechtotext(file):
    audio_file= open(file, "rb")
    transcription = client.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_file, 
    response_format="text"  
    )
    return str(transcription)

@app.route('/speechtotext', methods=['POST'])
def process_audio():
    audio_data = request.get_json()['audio']
    audio_bytes = base64.b64decode(audio_data)
    
    # Convert the audio bytes to a file
    audio_file = 'audio.wav'
    with open(audio_file, 'wb') as f:
        f.write(audio_bytes)
    
    text = speechtotext(audio_file)
    
    # Return a response
    return jsonify({'message': 'Audio processed successfully', 'text': text})

@app.route('/texttospeech', methods=['POST'])
def text_to_speech():
    text = request.get_json()['text']
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )
    
    # Convert the audio to base64 string
    audio_base64 = base64.b64encode(response['audio']).decode('utf-8')
    
    # Return the base64 string as a response
    return jsonify({'message': 'Text converted to speech successfully', 'audio': audio_base64})

if __name__ == '__main__':
    app.run(debug=True, port=5001)