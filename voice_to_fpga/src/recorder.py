# make the sound record

import sounddevice as sd
import time
import numpy as np
SAMPLE_RATE = 16000
DURATION_SECONDS = 2
JUMP_STEP = 1600


def play_voice(audio_section):
    print("in 1 second you will sound the voice")
    time.sleep(1)
    sd.play(audio_section, samplerate=SAMPLE_RATE)
    sd.wait()

def record_voice():
    print("Start recording in:")
    for i in range(2, 0, -1):
        print(i)
        time.sleep(1)

    print("Recording", end="", flush=True)
    audio = sd.rec(int(DURATION_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    for _ in range(DURATION_SECONDS):
        time.sleep(0.8)
        print(".", end="", flush=True)
    sd.wait()
    print("\nRecording complete.")
    return audio


def find_loudest_window(audio):
    index_energy = [0, 0]
    for i in range(0, len(audio) - SAMPLE_RATE, JUMP_STEP):
        window = audio[i:i + SAMPLE_RATE]
        energy = np.sum(window ** 2)
        if energy >= index_energy[1]:
            index_energy = [i, energy]
    #     print(f"Energy of segment {i}: {energy}")
    # print("___________________________")
    # print(f"Energy of segment {index_energy[0]}: {index_energy[1]}")
    return audio[index_energy[0]: index_energy[0] + SAMPLE_RATE].flatten()





def record_audio(seconds=1, sample_rate=16000):
    print("Recording...")
    audio = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    print("Done.")
    return audio.flatten()
