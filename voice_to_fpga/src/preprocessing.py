# compute the MFCC

import librosa
import numpy as np
import parctice_modle


# def extract_mfcc(audio, sr=16000):
#     print("Audio length:", len(audio))
#     print("Expected: ~16000 samples (1 second at 16kHz)")
#     mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=parctice_modle.NUM_MFCC)
#     print("MFCC shape:", mfcc.shape)
#     mfcc = parctice_modle.pad_or_truncate(mfcc)
#     mfcc = mfcc[..., np.newaxis]
#     return np.expand_dims(mfcc, axis=0)


def extract_mfcc(audio, sr=16000):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=parctice_modle.NUM_MFCC)
    print("mfcc shape before padding:", mfcc.shape)
    mfcc = parctice_modle.pad_or_truncate(mfcc)
    mfcc = mfcc[..., np.newaxis]  # מוסיף ערוץ
    return np.expand_dims(mfcc, axis=0)  # מוסיף באטש
