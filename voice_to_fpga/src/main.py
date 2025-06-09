#

from recorder import record_voice,find_loudest_window, play_voice
from preprocessing import extract_mfcc
from predictor import load_model, predict_word


audio = record_voice()
loudest_window = find_loudest_window(audio)
play_voice(loudest_window)
mfcc_input = extract_mfcc(loudest_window)
print("Final input shape:", mfcc_input.shape)

interpreter = load_model("command_model.tflite")
word, conf = predict_word(interpreter, mfcc_input)

    # print(f"\nDetected: {word} ({conf:.2f})")
