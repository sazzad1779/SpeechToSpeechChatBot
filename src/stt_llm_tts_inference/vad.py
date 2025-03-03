import torch
import numpy as np
class VAD:
    def __init__(self, model_name='snakers4/silero-vad',model='silero_vad'):
        self.model_name =model_name
        self.vad_model = None
        self.get_speech_timestamps =None
        self.model=model

    # Initialize VAD (Voice Activity Detection) Model
    def load_vad_model(self):
        self.vad_model, utils = torch.hub.load(repo_or_dir=self.model_name , model=self.model, force_reload=True)
        (self.get_speech_timestamps, _, _, _, _) = utils
        

    def apply_vad(self,audio_array, sampling_rate=16000):
        audio_tensor = torch.tensor(audio_array).unsqueeze(0)
        speech_timestamps = self.get_speech_timestamps(audio_tensor, self.vad_model, sampling_rate=sampling_rate)
        
        if not speech_timestamps:
            return None
        
        speech_chunks = [audio_array[timestamp['start']:timestamp['end']] for timestamp in speech_timestamps]
        speech_array = np.concatenate(speech_chunks)
        return speech_array