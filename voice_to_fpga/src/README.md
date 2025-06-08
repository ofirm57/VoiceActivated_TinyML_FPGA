Voice Activated - TinyML  🔊➡️🔌
This project is a voice-controlled embedded system using TinyML on the DE10-Standard board.

It recognizes spoken keywords (e.g., "up", "down", "left", "right", "stop") and sends a command to the FPGA fabric to activate an LED, motor, or any other logic-controlled component.


🔧 Current Stage: Python-based Voice Capture & Preprocessing
At this point, the focus is on:

Recording audio via microphone (sounddevice)
Extracting features (such as energy / spectrogram)
Preparing input format for TensorFlow Lite model inference
Planning transition to embedded inference on ARM + FPGA
🔧 Current Stage: Running on ARM + FPGA Response
The system currently includes:

📥 Audio Recording from microphone using Python (sounddevice)
🧠 Preprocessing & Feature Extraction (MFCC)
🤖 TinyML Inference using a TensorFlow Lite model on the ARM Cortex-A9 (Linux)
📤 Signal Transmission to FPGA, triggering a SystemVerilog RTL module
🧪 Real-time end-to-end testing for voice-to-hardware response
🛠 Tech Stack
Python 3.10
Libraries: numpy, scipy, sounddevice, tensorflow / tflite_runtime
Embedded Linux on ARM (DE10-Standard)
Model Format: .tflite
Audio Sampling Rate: 16kHz
Recording Duration: 1–3 seconds
RTL written in SystemVerilog (compiled for Cyclone V SoC FPGA)
📁 Project Structure
voice_to_fpga/
├── main.py               # Entry point: coordinates recording, preprocessing, prediction
├── recorder.py           # Handles microphone recording using sounddevice
├── preprocessing.py      # Extracts MFCC features from audio
├── predictor.py          # Loads and runs the TFLite model
├── parctice_modle.py     # (Likely practice/training or debugging script)
├── model/                # Contains the trained .tflite model
├── rtl/                  # (Planned) SystemVerilog logic for FPGA response
├── utils/                # Helper functions (e.g., file IO, logging)
├── README.md             # You are here
🚀 Next Steps
Integrate push-button fallback and override
Optimize latency and real-time performance
Add visual/debug tools (e.g., confidence plot or CLI display)
Package for deployment