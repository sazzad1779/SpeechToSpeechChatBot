import soundfile as sf
from kokoro import KPipeline
import time
from src.stt_llm_tts_inference.audio_utils import combine_audio_files
import os
import io
class TextToSpeech:
    def __init__(self, language_code='a', voice='af_bella', speed=1.0):
        """
        Initialize the TextToSpeech class with specified parameters.

        Args:
            language_code (str): Language code for the TTS model (default is 'a' for American English).
            voice (str): Voice selection for the TTS model (default is 'af_bella').
            speed (float): Speech speed; 1.0 is normal speed (default is 1.0).
        """
        self.language_code = language_code
        self.voice = voice
        self.speed = speed
        self.pipeline = None

    def load_model(self):
        """Initialize the TTS pipeline with the specified language code."""
        self.pipeline = KPipeline(lang_code=self.language_code)
        print(f"Model loaded for language: {self.language_code}")

    def transcribe(self, text):
        """
        Convert text to speech and save the output as a WAV file.

        Args:
            text (str): The text to be converted to speech.
        """
        if not self.pipeline:
            raise ValueError("Model is not loaded. Call load_model() first.")
        start_time = time.time()
        # Generate audio from text
        generator = self.pipeline(
            text,
            voice=self.voice,
            speed=self.speed,
            split_pattern=r'\n+'
        )
        end_time = time.time()-start_time
        print("generation time:", end_time)
        # Save the generated audio to a WAV file
        for i, (gs, ps, audio) in enumerate(generator):
            output_filename = f"output_{i}.wav"
            sf.write(output_filename, audio, 24000)
            print(f"Audio saved to {output_filename}")
            
    def predict(self,text):
        if not self.pipeline:
            raise ValueError("Model is not loaded. Call load_model() first.")
        start_time = time.time()
        # Generate audio from text
        generator = self.pipeline(
            text,
            voice=self.voice,
            speed=1,
            split_pattern=r'\n+'
        )
        end_time = time.time()-start_time
        print("generation time:", end_time)
        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save each audio file
        for i, (gs, ps, audio) in enumerate(generator):
            file_path = f"{output_dir}/{i}.wav"
            sf.write(file_path, audio, 24000)

        # Combine all audio files
        final_audio, samplerate = combine_audio_files(output_dir, i)

        # Save the final audio to a buffer
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, final_audio, samplerate, format="WAV")
        audio_buffer.seek(0)
        audio_data = audio_buffer.getvalue()
        audio_buffer.close()

        # Return the audio data
        return audio_data
        
if __name__ == "__main__":
    # Initialize the TTS system
    tts = TextToSpeech(language_code='a', voice='af_heart', speed=1.0)
    tts.load_model()

    # Define the text to be converted to speech
    # text = "Kokoro is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient. With Apache-licensed weights, Kokoro can be deployed anywhere from production environments to personal projects."
    text="""
    The Journey of Resilience: A Tale of Triumph Over Adversity. In a quaint village nestled between rolling hills and a serene river, lived a young girl named Maya. From an early age, Maya exhibited an insatiable curiosity and a zest for life. Her days were filled with laughter, play, and dreams of exploring the world beyond her village.However, life had other plans. A devastating illness swept through the village, claiming the lives of many, including Maya's parents. Left orphaned and alone, Maya was forced to confront the harsh realities of the world. Yet, she refused to succumb to despair. Drawing strength from the memories of her parents' love and teachings, she resolved to persevere.Determined to rebuild her life, Maya sought work in the village. She found employment as a seamstress, using her skills to create garments for the villagers. Despite the challenges, she poured her heart into her work, finding solace in the rhythmic motion of her needle and thread.
As the years passed, Maya's reputation as a skilled seamstress spread beyond the village. Merchants from distant towns sought her creations, and her small shop transformed into a thriving business. Yet, Maya remained humble, always remembering her roots and the lessons learned from her parents.
One day, a devastating flood ravaged the village, destroying homes, crops, and livelihoods. Maya's shop was among the many buildings washed away. Standing amidst the ruins, she felt a pang of loss but also a surge of determination. She had faced adversity before and emerged stronger. This time would be no different.
With unwavering resolve, Maya rallied the villagers. Together, they rebuilt their homes and their lives. Maya's leadership and resilience became a beacon of hope, inspiring others to find strength in the face of adversity.
Years later, Maya stood atop the hill overlooking the village she had rebuilt. The sun set behind her, casting a golden glow over the land. She smiled, knowing that her journey was a testament to the power of resilience, hope, and the unwavering human spirit.
Moral of the Story: Adversity is an inevitable part of life, but it is our response to challenges that defines us. Embrace resilience, seek strength in community, and never lose sight of hope.
    """
    # Convert text to speech and save the output
    tts.transcribe(text)