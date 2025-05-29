# Voice to FPGA 🔊➡️🔌

This project is a voice-controlled embedded system using **TinyML** on the **DE10-Standard** board. It recognizes a spoken keyword (such as `"Go"`) and sends a command to the **FPGA fabric** to activate an LED, motor, or any logic-controlled component.

---

## 🔧 Current Stage: Python-based Voice Capture & Preprocessing

At this point, the focus is on:
- Recording audio via microphone (`sounddevice`)
- Extracting features (such as energy / spectrogram)
- Preparing input format for TensorFlow Lite model inference
- Planning transition to embedded inference on ARM + FPGA

---

## 🛠 Tech Stack

- **Python 3.10**
- `numpy`, `scipy`, `sounddevice`, `tensorflow`
- Audio sampling rate: `16kHz`
- Recording duration: `3 seconds`
- Model format: `.tflite`

---

## 📁 Project Structure

```bash
voice_to_fpga/
├── main.py               # Records and plays voice
├── preprocess.py         # Signal processing (WIP)
├── model/                # Contains .tflite model
├── utils/                # Helper functions
├── README.md             # You are here
