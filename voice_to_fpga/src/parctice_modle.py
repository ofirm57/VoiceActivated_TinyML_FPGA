
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
from tensorflow.keras.callbacks import ModelCheckpoint



MODEL_PATH = 'command_model_last.h5'
DATASET_PATH = '/Users/wpyrmlkyly/Desktop/ProgramProjects/fpga/voice_to_fpga/src/dataset'
WORDS = ['up', 'down', 'left', 'right', 'stop','silence']
NUM_MFCC = 13
MAX_LEN = 32  # מספר פריימים שנשמרים לכל דגימה

checkpoint = ModelCheckpoint(
    'command_model_last.h5',       # איפה לשמור
    monitor='val_accuracy',# לפי איזה מדד
    save_best_only=True,   # רק כשהמודל משתפר
    mode='max',            # אנחנו רוצים מקסימום של val_accuracy
    verbose=1              # הדפסת מידע בזמן האימון
)


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

    # split by speakers
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

    for label_idx, word in enumerate(WORDS):
        count = np.sum(y_test == label_idx)
        print(f"{word}: {count} samples in test set")

    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print("X shape before reshape:", X_train.shape)

    # add channel to model cnn
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
            Dropout(0.35),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.35),
            Dense(len(WORDS), activation='softmax')
        ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train_cat, epochs=30, batch_size=32, validation_split=0.1, callbacks=[early_stopping, checkpoint])

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
