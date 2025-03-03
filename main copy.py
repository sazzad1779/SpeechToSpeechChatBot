import io
import os
import base64
import json
import time
import tempfile
import librosa
import soundfile as sf
import queue
from threading import Thread, Lock
from pydub import AudioSegment
from flask import Flask, render_template
from flask_socketio import SocketIO
from src.stt_llm_tts_inference.stt_infer import SpeechToText
from src.stt_llm_tts_inference.tts_infer import TextToSpeech
from src.stt_llm_tts_inference.llm_infer import LLM_Model
from src.stt_llm_tts_inference.vad import VAD
from transformers import TextIteratorStreamer

# Create Flask app and initialize SocketIO
app = Flask(__name__,
    static_folder="src/web/static",
    template_folder="src/web/templates")
socketio = SocketIO(app, cors_allowed_origins="*")
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching
system_message="আপনি একজন সহায়ক সহকারী। আপনার উত্তর আরো সঠিক এবং তথ্যপূর্ণ রাখুন. নোট করুন আপনি যদি উত্তরটি না জানেন তাহলে কোনো এলোমেলো উত্তর দেবেন না।"
# Set model directory path
model_dir = os.path.join(os.getcwd(), "models")

# Load models
asr_model = load_asr_model(model_dir)
vad_model, get_speech_timestamps = load_vad_model()
tts_bn_model = load_bangla_tts_model(model_dir, "female")
model, tokenizer = model_setup()

## stt load
stt = SpeechToText()
stt.load_model()

## vad 
vad = VAD()
vad.load_vad_model()

## LLM load
# model_name = "Qwen/Qwen2.5-0.5B-Instruct"
# llm_loader =LLM_Model(model_name)
# model, tokenizer =llm_loader.setup_model_and_tokenizer()

## tts load
tts_loader = TextToSpeech()
tts_loader.load_model()
print("Models loaded.")

# Dictionary to hold queues and locks for each session
session_queues = {}
session_locks = {}

# Model locks to avoid simultaneous access issues
model_locks = {
    'asr': Lock(),
    'vad': Lock(),
    'tts': Lock(),
}

# Initialize queue and lock for a new session
def initialize_session(session_id):
    if session_id not in session_queues:
        session_queues[session_id] = queue.Queue()  # Create a queue for each session
        session_locks[session_id] = Lock()  # Lock for sequential processing
        Thread(target=process_queue, args=(session_id,), daemon=True).start()  # Start queue processor thread

# Process queue items for each session one by one
def process_queue(session_id):
    while True:
        data = session_queues[session_id].get()  # Wait for next data in the queue
        with session_locks[session_id]:  # Ensure single processing at a time
            handle_audio(data, session_id)
            session_queues[session_id].task_done()

# Flask and SocketIO Routes and Handlers
@app.route('/')
def index():
    return render_template('client1_v2.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

# Audio Processing Function
def process_audio(data):
    try:
        audio_bytes = io.BytesIO(data)
        audio_array, sample_rate = convert_to_wav(audio_bytes)
        audio_array = resample_audio(audio_array, sample_rate)

        # Run VAD and ensure proper exception handling
        with model_locks['vad']:
            speech_array = apply_vad(vad_model, get_speech_timestamps, audio_array)
        
        if speech_array is None or len(speech_array) == 0:
            print("No speech detected in the audio segment.")
            return None

        return speech_array
    except Exception as e:
        raise RuntimeError(f"Error in processing audio: {e}")

# Convert audio to WAV format and resample if needed
def convert_to_wav(audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
        temp_audio.write(audio_bytes.read())
        temp_audio_path = temp_audio.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
        AudioSegment.from_file(temp_audio_path).export(temp_wav.name, format="wav")
        temp_wav_path = temp_wav.name

    os.remove(temp_audio_path)  # Remove the WebM temporary file
    audio_array, sample_rate = sf.read(temp_wav_path, dtype='float32')
    os.remove(temp_wav_path)  # Remove the WAV temporary file

    return audio_array, sample_rate

def resample_audio(audio_array, sample_rate, target_sr=16000):
    if sample_rate != target_sr:
        audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=target_sr)
    return audio_array

# Convert Speech to Bytes
def convert_audio_to_bytes(speech_audio):
    with io.BytesIO() as audio_file:
        sf.write(audio_file, speech_audio, 22050, format='wav')
        audio_file.seek(0)
        return audio_file.read()

# Streaming response generation
def generate_streaming_response(prompt, session_id,max_new_tokens:int=200,temperature:float=0.2,top_p:float=0.95):
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Start the generation in a separate thread to avoid blocking
    def generation_thread():
        model.generate(inputs['input_ids'], streamer=streamer, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, repetition_penalty=1.15)
    
    Thread(target=generation_thread, daemon=True).start()

    response_text = ""
    # Stream the tokens as they are generated
    for token in streamer:
        response_text += token  # Accumulate tokens for the complete response
        socketio.emit('llm_response', {'response': token, 'session_id': session_id})

    socketio.emit('llm_response_end', {'session_id': session_id})  # Notify client that response is complete
    return response_text

# Handle Audio Processing (modified to take session_id for queuing)
def handle_audio(data, session_id):
    try:
        print(f'Processing audio for session: {session_id}')
        file_path = process_audio(data)
        if file_path is None:
            print("No speech detected")
            return
        
        # Ensure ASR model usage is thread-safe
        with model_locks['asr']:
            transcription = stt.transcribe( file_path)
        
        print(f"Transcription: {transcription}")
        
        # Emit the transcription text (user input) to the client
        socketio.emit('transcription', {'transcription': transcription, 'session_id': session_id})
        time.sleep(2)
        # Generate the streaming response from the LLM model
        prompt = f"<s>[INST] <<SYS>>\n\n{system_message}\n\n<</SYS>>\n{transcription}\n[/INST]"
        response_text = generate_streaming_response(prompt, session_id,max_new_tokens=20)
        print("response_text: ",response_text)
        # Ensure TTS model usage is thread-safe and use the final response for TTS
        with model_locks['tts']:
            speech_audio = bangla_text_to_speech(tts_bn_model, response_text)
        
        audio_bytes = convert_audio_to_bytes(speech_audio)

        # Emit the generated TTS audio to the client
        response_data = {
            'text': response_text,
            'audio': base64.b64encode(audio_bytes).decode('utf-8'),
            'session_id': session_id
        }
        socketio.emit('audio', json.dumps(response_data))

    except Exception as e:
        print(f"Error in audio processing for session {session_id}: {e}")

# Handle Audio Input from Client and add to session queue
@socketio.on('audio')
def handle_audio_request(data):
    session_id = data.get('session_id')
    initialize_session(session_id)  # Ensure session is initialized

    # Decode the base64-encoded audio data
    base64_audio_data = data.get('data')
    if base64_audio_data:
        audio_bytes = base64.b64decode(base64_audio_data)
        session_queues[session_id].put(audio_bytes)  # Add audio bytes to the queue
        print(f"Audio data received for session {session_id}, added to queue...")
    else:
        print(f"No audio data received for session {session_id}")

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=4000)
