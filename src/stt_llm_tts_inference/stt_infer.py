import nemo.collections.asr as nemo_asr
import time
class SpeechToText:
    def __init__(self, model_name="nvidia/parakeet-tdt_ctc-110m"):
        """Initialize the STT model."""
        self.model_name = model_name
        self.asr_model = None

    def load_model(self):
        """Loads the ASR model from NeMo's pre-trained models."""
        print(f"Loading model: {self.model_name}...")
        self.asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.model_name)
        print("Model loaded successfully!")

    def transcribe(self, audio_files):
        """
        Transcribes audio files into text.

        Args:
            audio_files (list[str] or str): Path(s) to the audio file(s) to transcribe.

        Returns:
            list: Transcriptions for each input file.
        """
        if self.asr_model is None:
            raise ValueError("Model is not loaded. Call load_model() first.")

        if isinstance(audio_files, str):  # Convert single file to list
            audio_files = [audio_files]

        print("Transcribing...")
        start_time = time.time()
        transcripts = self.asr_model.transcribe(audio_files)
        
        end_time = time.time()-start_time
        print(50*"*")
        for file, text in zip(audio_files, transcripts):
            print(f"🎤 {file} → 📝 {text}")

        
        print(50*"*","\nSTT generation time:", end_time)
        
        return transcripts

# Example Usage
if __name__ == "__main__":
    stt = SpeechToText()
    stt.load_model()
    result = stt.transcribe(["output_0.wav","output_1.wav","output_2.wav","output_3.wav"])  # Replace with your audio file path
    
