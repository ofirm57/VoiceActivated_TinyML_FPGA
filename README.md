# Voice Activated - TinyML 🔊➡️🔌

This project is a voice-controlled embedded system using TinyML on the DE10-Standard board.  
It recognizes spoken keywords (e.g., "up", "down", "left", "right", "stop") and sends a command to the FPGA to activate an LED, motor, or other hardware component.

---

## 📖 About

Work-in-progress: full pipeline from keyword spotting to real-time FPGA control.  
The model was trained externally and deployed on the ARM processor for embedded inference.

---

## ✅ What’s Done

- 📥 Audio recording via microphone (`sounddevice`)  
- 🧠 MFCC feature extraction (`librosa`, `scipy`)  
- 🧪 Training and evaluation of a CNN-based classification model  
- 🧱 Converted model to `.tflite` and tested locally  
- 🚀 Transferred code to run on ARM Cortex-A9 (embedded Linux)  
- 🔗 Sent commands from ARM to FPGA RTL logic via GPIO

---

## 🔄 Current Stage

- Real-time end-to-end testing: microphone → inference on ARM → signal to FPGA  
- Inference runs with `tflite_runtime` on embedded Linux  
- SystemVerilog module receives and responds to prediction output

---

## 🛠 Tech Stack

- Python 3.10  
- Libraries: `numpy`, `scipy`, `librosa`, `sounddevice`, `tensorflow`, `tflite_runtime`  
- ARM Cortex-A9 on DE10-Standard (Linux)  
- Model Format: `.tflite`  
- Audio: 16kHz / 1–3 sec clips  
- RTL: SystemVerilog (Cyclone V SoC FPGA)

---

## 📁 Project Structure

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

---

## 🚀 Next Steps

- Add push-button override mechanism  
- Improve latency and timing precision  
- Add confidence/visual feedback (e.g., CLI plot)  
- Prepare for packaging and final deployment  
