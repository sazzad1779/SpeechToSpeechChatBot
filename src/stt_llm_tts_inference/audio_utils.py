import os
import soundfile as sf
import numpy as np


def combine_audio_files(folder, index):
    """
    Combine all audio files from the folder and return the concatenated audio data.

    Args:
        - folder (str): Path to the folder containing the .wav files.
        - index (int): The index of the last file to be concatenated.

    Returns: None
    """
    # Get a sorted list of .wav files by their numeric names
    wav_files = sorted(
        [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".wav")],
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
    )

    # Raise an error if no .wav files are found
    if not wav_files:
        raise ValueError("No .wav files found in the input folder.")

    # Combine all .wav files
    audio_data = []
    samplerate = None

    for wav_file in wav_files[:index+1]:
        data, samplerate = sf.read(wav_file)
        audio_data.append(data)

    # Return the concatenated audio data and the sampling rate
    final_audio = np.concatenate(audio_data)
    return final_audio, samplerate
