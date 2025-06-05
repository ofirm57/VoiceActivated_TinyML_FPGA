#
# import os
# import numpy as np
# import tensorflow as tf
#
# #for signal proccing
# import librosa
# #data to train/test.
# from sklearn.model_selection import train_test_split
#
#
# #build the modle
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.callbacks import EarlyStopping
# import collections
# import random
#
# MODEL_PATH = 'command_model.h5'
# DATASET_PATH = '/Users/wpyrmlkyly/Desktop/ProgramProjects/fpga/voice_to_fpga/src/dataset'
# WORDS = ['up', 'down', 'left', 'right', 'stop']
#
#
#
# def extract_speaker_id(filename):
#     return filename.split('_')[0]  # e.g. 3bfd30e6 from 3bfd30e6_nohash_0.wav
#
# def load_dataset_by_speaker():
#     data = []
#     for label_idx, word in enumerate(WORDS):
#         word_path = os.path.join(DATASET_PATH, word)
#         for file_name in os.listdir(word_path):
#             if not file_name.endswith(".wav"):
#                 continue
#             speaker = extract_speaker_id(file_name)
#             file_path = os.path.join(word_path, file_name)
#             try:
#                 signal, sr = librosa.load(file_path, sr=16000)
#                 mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
#                 mfcc = np.mean(mfcc.T, axis=0)
#                 data.append((mfcc, label_idx, speaker))
#             except Exception as e:
#                 print(f"Error reading {file_path}: {e}")
#     return data
#
#
#
#
# def split_by_speaker(data, test_ratio=0.2):
#     speaker_data = collections.defaultdict(list)
#     for x, y, speaker in data:
#         speaker_data[speaker].append((x, y))
#
#     speakers = list(speaker_data.keys())
#     random.seed(42)
#     random.shuffle(speakers)
#
#     split_index = int(len(speakers) * (1 - test_ratio))
#     train_speakers = set(speakers[:split_index])
#     test_speakers = set(speakers[split_index:])
#
#     print("\n🔹 Speakers in Training Set:")
#     for s in sorted(train_speakers):
#         print(f" - {s}")
#
#     print("\n🔸 Speakers in Test Set:")
#     for s in sorted(test_speakers):
#         print(f" - {s}")
#
#     X_train, y_train, X_test, y_test = [], [], [], []
#
#     for speaker, samples in speaker_data.items():
#         for x, y in samples:
#             if speaker in train_speakers:
#                 X_train.append(x)
#                 y_train.append(y)
#             else:
#                 X_test.append(x)
#                 y_test.append(y)
#
#     return (
#         np.array(X_train), np.array(X_test),
#         np.array(y_train), np.array(y_test)
#     )
#
#
# """
#  Loads the speech command dataset from the specified DATASET_PATH.
#
#  For each of the selected words (in WORDS), it:
#  - Iterates over all .wav files in the corresponding folder.
#  - Loads each audio file using librosa with a sampling rate of 16kHz.
#  - Extracts 13 MFCC features from the audio.
#  - Averages the MFCCs over time to create a fixed-size feature vector.
#  - Appends the feature vector to X and the word label index to y.
#
#  Returns:
#      X (np.ndarray): Array of MFCC feature vectors.
#      y (np.ndarray): Array of integer labels (corresponding to WORDS).
#  """
# def load_dataset():
#     X, y = [], []
#     for label_idx, word in enumerate(WORDS):
#         word_path = os.path.join(DATASET_PATH, word)
#         for file_name in os.listdir(word_path):
#             file_path = os.path.join(word_path, file_name)
#             try:
#                 signal, sr = librosa.load(file_path, sr=16000)
#             except Exception as e:
#                 print(f"Error reading {file_path}: {e}")
#                 continue
#             # mfcc = librosa.feature.mfcc(signal, sr=sr, n_mfcc=13)
#             mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
#
#             mfcc_mean = np.mean(mfcc.T, axis=0)
#             mfcc_std = np.std(mfcc.T, axis=0)
#             mfcc = np.concatenate([mfcc_mean, mfcc_std])
#             X.append(mfcc)
#             y.append(label_idx)
#     return np.array(X), np.array(y)
#
#
# def build_model(x,num_of_words):
#     model = Sequential([
#         Dense(256, activation='relu', input_shape=(x,)),
#         Dropout(0.3),
#         Dense(128, activation='relu'),
#         Dropout(0.3),
#         Dense(num_of_words, activation='softmax')
#     ])
#     return model
#
#
# if __name__ == '__main__':
#
#     # X, y = load_dataset()
#     data = load_dataset_by_speaker()
#     X_train, X_test, y_train, y_test = split_by_speaker(data)
#
#     # early_stopping = EarlyStopping(
#     #     monitor='val_loss',
#     #     patience=5,
#     #     restore_best_weights=True
#     # )
#     #
#     # label_counts = collections.Counter(y)
#     # for label_idx, count in label_counts.items():
#     #     print(f"{WORDS[label_idx]}: {count} samples")
#
#     # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
#
#     y_train_cat = to_categorical(y_train, num_classes=len(WORDS))
#     y_test_cat = to_categorical(y_test, num_classes=len(WORDS))
#
#     if os.path.exists(MODEL_PATH):
#         print("loud existed model")
#         model = load_model(MODEL_PATH)
#     else:
#         print("build new modle")
#         model = build_model(X_train.shape[1],len(WORDS))
#
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
#
#     model.fit(X_train, y_train_cat, epochs=30, batch_size=32, validation_split=0.1,callbacks=[early_stopping])
#
#
#     test_loss, test_acc = model.evaluate(X_test, y_test_cat)
#     print(f'Test accuracy: {test_acc:.2f}')
#
#     model.save("command_model.h5")

#
# import os
# import numpy as np
# import tensorflow as tf
# import librosa
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.callbacks import EarlyStopping
# import collections
#
# MODEL_PATH = 'command_model.h5'
# DATASET_PATH = '/Users/wpyrmlkyly/Desktop/ProgramProjects/fpga/voice_to_fpga/src/dataset'
# WORDS = ['up', 'down', 'left', 'right', 'stop']
#
#
# def load_dataset_by_speaker():
#     speaker_to_samples = {}
#     for label_idx, word in enumerate(WORDS):
#         word_path = os.path.join(DATASET_PATH, word)
#         for file_name in os.listdir(word_path):
#             if not file_name.endswith('.wav'):
#                 continue
#             file_path = os.path.join(word_path, file_name)
#             speaker_id = file_name.split('_nohash_')[0]
#             try:
#                 signal, sr = librosa.load(file_path, sr=16000)
#             except Exception as e:
#                 print(f"Error reading {file_path}: {e}")
#                 continue
#             mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
#             mfcc = np.mean(mfcc.T, axis=0)
#             speaker_to_samples.setdefault(speaker_id, []).append((mfcc, label_idx))
#
#     # פיצול לפי דוברים
#     speakers = list(speaker_to_samples.keys())
#     train_speakers, test_speakers = train_test_split(speakers, test_size=0.2, random_state=42)
#
#     def extract_data(speaker_list):
#         X, y = [], []
#         for spk in speaker_list:
#             for mfcc, label in speaker_to_samples[spk]:
#                 X.append(mfcc)
#                 y.append(label)
#         return np.array(X), np.array(y)
#
#     return extract_data(train_speakers), extract_data(test_speakers)
#
#
# if __name__ == '__main__':
#     (X_train, y_train), (X_test, y_test) = load_dataset_by_speaker()
#
#     for label_idx in range(len(WORDS)):
#         count = np.sum(y_train == label_idx) + np.sum(y_test == label_idx)
#         print(f"{WORDS[label_idx]}: {count} samples")
#
#     y_train_cat = to_categorical(y_train, num_classes=len(WORDS))
#     y_test_cat = to_categorical(y_test, num_classes=len(WORDS))
#
#     early_stopping = EarlyStopping(
#         monitor='val_loss',
#         patience=5,
#         restore_best_weights=True
#     )
#
#     if os.path.exists(MODEL_PATH):
#         print("load existed model")
#         model = load_model(MODEL_PATH)
#     else:
#         print("build new model")
#         model = Sequential([
#             Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
#             Dropout(0.3),
#             Dense(128, activation='relu'),
#             Dropout(0.3),
#             Dense(len(WORDS), activation='softmax')
#         ])
#
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
#     model.fit(X_train, y_train_cat, epochs=30, batch_size=32, validation_split=0.1, callbacks=[early_stopping])
#
#     test_loss, test_acc = model.evaluate(X_test, y_test_cat)
#     print(f'Test accuracy: {test_acc:.2f}')
#
#     model.save(MODEL_PATH)
######################################################################################
import os
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib

import librosa
import collections
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

MODEL_PATH = 'command_model.h5'
DATASET_PATH = '/Users/wpyrmlkyly/Desktop/ProgramProjects/fpga/voice_to_fpga/src/dataset'
WORDS = ['up', 'down', 'left', 'right', 'stop']
NUM_MFCC = 13
MAX_LEN = 32  # מספר פריימים שנשמרים לכל דגימה

def pad_or_truncate(mfcc, max_len=MAX_LEN):
    if mfcc.shape[1] > max_len:
        return mfcc[:, :max_len]
    elif mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        return np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    return mfcc





def load_dataset_by_speaker():
    speaker_to_samples = {}
    for label_idx, word in enumerate(WORDS):
        word_path = os.path.join(DATASET_PATH, word)
        for file_name in os.listdir(word_path):
            if not file_name.endswith('.wav'):
                continue
            file_path = os.path.join(word_path, file_name)
            speaker_id = file_name.split('_nohash_')[0]
            try:
                signal, sr = librosa.load(file_path, sr=16000)
                mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=NUM_MFCC)
                mfcc = pad_or_truncate(mfcc)
                speaker_to_samples.setdefault(speaker_id, []).append((mfcc, label_idx))
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

    # פיצול לפי דוברים
    speakers = list(speaker_to_samples.keys())
    train_speakers, test_speakers = train_test_split(speakers, test_size=0.2, random_state=42)



    def extract_data(speaker_list):
        X, y = [], []
        for spk in speaker_list:
            for mfcc, label in speaker_to_samples[spk]:
                X.append(mfcc)
                y.append(label)
        return np.array(X), np.array(y)

    return extract_data(train_speakers), extract_data(test_speakers)

if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = load_dataset_by_speaker()

    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print("X shape before reshape:", X_train.shape)

    # מוסיפים ערוץ למודל CNN
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    print("X shape after reshape:", X_train.shape)

    y_train_cat = to_categorical(y_train, num_classes=len(WORDS))
    y_test_cat = to_categorical(y_test, num_classes=len(WORDS))

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    if os.path.exists(MODEL_PATH):
        print("load existed model")
        model = load_model(MODEL_PATH)
    else:
        print("build new model")
        model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(NUM_MFCC, MAX_LEN, 1)),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.3),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(len(WORDS), activation='softmax')
        ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train_cat, epochs=30, batch_size=32, validation_split=0.1, callbacks=[early_stopping])

    test_loss, test_acc = model.evaluate(X_test, y_test_cat)
    print(f'Test accuracy: {test_acc:.2f}')

    import matplotlib

    matplotlib.use('Agg')  # לא מנסה לפתוח חלון GUI

    # חיזוי על סט הבדיקה
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test_cat, axis=1)

    # הצגת מטריצת בלבול
    cm = confusion_matrix(y_true, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=WORDS)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")  # שומר את התמונה

    # plt.show()

    model.save(MODEL_PATH)
