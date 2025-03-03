import librosa
import soundfile as sf

# Load stereo file
audio, sr = librosa.load("harvard.wav", sr=16000, mono=True)

# Save as mono file
sf.write("harvard_mono.wav", audio, sr)
