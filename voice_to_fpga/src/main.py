# 🎤 Captures audio from the microphone
# Returns a NumPy array containing the recorded waveform
import sounddevice as sd

# 📊 Core library for handling arrays and numerical processing
# Useful for slicing, FFT, normalization, math, etc.
import numpy as np

# 🤖 Loads your .tflite machine learning model
# Runs inference (prediction) using audio input
# Returns a result – e.g., "Detected word: Go"
from tensorflow.lite.python.interpreter import Interpreter
# from tensorflow.lite import Interpreter  # Alternative import path depending on version
# 📐 Provides advanced signal processing functions
# Like filtering, resampling, Fourier transforms
from scipy import signal

# ⚙️ Utility libraries
# Used for file operations, delays, and time measurement
import os
import time

import librosa
import parctice_modle
from tensorflow.keras.models import load_model



# שם הקובץ: speech_commands_float32.tflite
#
# סוג המודל: CNN שעבר אימון על מילים כמו "yes", "no", "go", "stop", ועוד.
#
# קלט נדרש: קובץ שמע בפורמט waveform (raw PCM) בגודל 1 שנייה, עם קצב דגימה של 16kHz.
#
# פלט: רשימה של הסתברויות לכל מילה.
SAMPLE_RATE = 16000
DURATION_SECONDS = 2
KEYWORD = "go"
TRESHOLD = 0.5
JUMP_STEP = 1600
MODEL_PATH = '/Users/wpyrmlkyly/Desktop/ProgramProjects/fpga/voice_to_fpga/command_model.h5'

def record_voice():
    print("Start recording in:")
    for i in range(3, 0, -1):
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


def record_and_preprocess():
    print("Start recording in:")
    for i in range(3, 0, -1):
        print(i)
        time.sleep(1)

    print("Recording", end="", flush=True)
    audio = sd.rec(int(SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    audio = audio.flatten()

    # חילוץ תכונות MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=parctice_modle.NUM_MFCC)
    mfcc_padded = parctice_modle.pad_or_truncate(mfcc)
    mfcc_padded = mfcc_padded[..., np.newaxis]  # הוספת ערוץ
    mfcc_padded = np.expand_dims(mfcc_padded, axis=0)  # הוספת מימד batch

    return mfcc_padded



def play_voice(audio_section):
    print("in two second you will sound the voice")
    time.sleep(2)
    sd.play(audio_section, samplerate=SAMPLE_RATE)
    sd.wait()


def trim_silence(audio):
    audio_section = audio.flatten()
    mask = np.abs(audio) > TRESHOLD
    if np.any(mask):
        return audio[np.where(mask)[0][0]: np.where(mask)[0][-1]]
    else:
        return audio_section


# the model require audio with 16k sample -> 1 sec, i will find the sec with the maximum energy

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
    return audio[index_energy[0]: index_energy[0] + SAMPLE_RATE]



def from_r2Spectrogram(window):
    s= signal.spectrogram(window,SAMPLE_RATE)
    s= Interpreter.signal(window,SAMPLE_RATE)
    Interpreter.sig


def predict_word(model, mfcc_input):
    prediction = model.predict(mfcc_input)
    predicted_index = np.argmax(prediction)
    predicted_word = parctice_modle.WORDS[predicted_index]
    confidence = prediction[0][predicted_index]

    print("\nAll probabilities:")
    for i, word in enumerate(parctice_modle.WORDS):
        print(f"{word}: {prediction[0][i]:.2f}")

    print(f"\nPredicted word: {predicted_word} ({confidence:.2f})")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    raw_audio = record_voice()
    loudest_window = find_loudest_window(raw_audio)  # 1 שנייה רועשת
    play_voice(loudest_window)
    mfcc = librosa.feature.mfcc(y=loudest_window.flatten(), sr=SAMPLE_RATE, n_mfcc=parctice_modle.NUM_MFCC)
    mfcc_padded = parctice_modle.pad_or_truncate(mfcc)
    mfcc_padded = mfcc_padded[..., np.newaxis]
    mfcc_input = np.expand_dims(mfcc_padded, axis=0)
    # זיהוי מילה מוקלטת
    print("\n--- Voice Command Recognition ---")
    m = load_model(MODEL_PATH)
    predict_word(m, mfcc_input)

    # a = np.zeros(4, dtype='float32')
    # b=[0.,0.,0.,0.]
    # print(b)
    #
    # print(a)
    # r = trim_silence(record_voice())
    # e_r = find_loudest_window(record_voice())
    # play_voice(e_r)
    # print("r is = ", r)
    #
    # for i in range(0, len(r), 2):
    #     segment = r[i:i + 4000]
    #     energy = np.sum(segment ** 2)
    #     print(f"Energy of segment {i // 4000}: {energy}")
    # print("after: is = ", r)
    # print("shape -> ", r.shape)
    # print("len -> ", len(r))
    # play_voice(trim_silence(record_voice()))
    # window = np.array(np.array([16000]
    # {0}), dtype = 'float32')
