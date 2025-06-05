# running the model

import numpy as np
import parctice_modle
from tensorflow.lite.python.interpreter import Interpreter

TRESHOLD = 0.6
MODEL_PATH = '/Users/wpyrmlkyly/Desktop/ProgramProjects/fpga/voice_to_fpga/src/command_model.tflite'

def load_model(MODEL_PATH):
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

# def predict(interpreter, mfcc_input, threshold=0.6):
#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()
#     interpreter.set_tensor(input_details[0]['index'], mfcc_input.astype(np.float32))
#     interpreter.invoke()
#     prediction = interpreter.get_tensor(output_details[0]['index'])[0]
#     predicted_index = np.argmax(prediction)
#     confidence = prediction[predicted_index]
#     word = parctice_modle.WORDS[predicted_index]
#     return word if confidence >= threshold else "Unknown", confidence




def predict_word(interpreter, mfcc_input):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # ודא שהקלט בפורמט float32
    mfcc_input = mfcc_input.astype(np.float32)

    # הזן את הקלט למודל
    interpreter.set_tensor(input_details[0]['index'], mfcc_input)
    interpreter.invoke()

    # קבל את הפלט
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]
    predicted_index = np.argmax(prediction)
    predicted_word = parctice_modle.WORDS[predicted_index]
    confidence = prediction[predicted_index]

    print("\nAll probabilities:")
    for i, word in enumerate(parctice_modle.WORDS):
        print(f"{word}: {prediction[i]:.2f}")

    if confidence < TRESHOLD:
        print(f"\nDetected: Unknown or Silence (confidence={confidence:.2f})")
    else:
        print(f"\nDetected: {predicted_word} (confidence={confidence:.2f})")
    return predicted_word, confidence
