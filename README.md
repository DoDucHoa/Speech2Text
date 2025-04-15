# Whisper Speech-to-Text & Translation Application

This repository contains a Python application that leverages [OpenAI Whisper](https://github.com/openai/whisper) to perform speech-to-text transcription and optional translation. The application supports both file-based and real-time audio processing, utilizes GPU acceleration via PyTorch when available, and features a graphical user interface (GUI) built using Tkinter.

## Features

- **Speech Recognition and Translation**: Transcribes speech from an audio file or real-time recording, with an option to translate non-English speech into English.
- **Real-Time Audio Processing**: Records audio using your microphone (default duration is configurable).
- **Output Options**: Display the transcription on the GUI or save it to a text file.
- **GPU Support**: Uses a CUDA-enabled version of PyTorch if available, otherwise falls back to CPU processing.
- **Performance Metrics**: Measures and displays the processing time for model loading and audio transcription.
- **Modular Design**: The GUI code is separated into `whisper_gui.py` and imported into `main.py` for clarity and ease of maintenance.

## Prerequisites

- **Python Version**: It is recommended to use Python 3.10 or 3.11 (Python 3.12 may require building PyTorch from source for GPU support).
- **PyTorch**: For GPU acceleration, install a CUDA-enabled build (e.g., CUDA 12.1 or similar).
- **OpenAI Whisper**: Speech recognition library from OpenAI.
- **PyAudio**: For capturing audio in real-time.
- **Tkinter**: Standard GUI library for Python (usually comes pre-installed).

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/whisper-speech-to-text.git
   cd whisper-speech-to-text
   ```

2. **Create and Activate a Virtual Environment:**

   On macOS/Linux:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
   On Windows:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Upgrade pip:**

   ```bash
   pip install --upgrade pip
   ```

4. **Install PyTorch with CUDA Support (if applicable):**

   For example, using CUDA 12.1:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

5. **Install Other Dependencies:**

   ```bash
   pip install openai-whisper pyaudio
   ```

## Project Structure

```
.
├── main.py           # Main entry point for launching the GUI
├── gui.py            # Contains the WhisperGUI class and helper functions
└── README.md         # This documentation file
```

## Usage

Launch the application by running the following command:

```bash
python main.py
```

The GUI will open and allow you to:

- **Select Input Type**:
  - **Audio File**: Click the "Select Audio File" button to choose an existing audio file.
  - **Real-Time Recording**: Record audio directly from your microphone (adjust duration as needed).
- **Choose Output Type**:
  - **Display on CLI**: The transcription result will appear in the GUI text box.
  - **Save to TXT File**: You will be prompted to specify where to save the transcription.
- **Configure Options**:
  - Select the Whisper model (e.g., tiny, base, small, medium, large).
  - Enable translation if you want non-English speech to be translated into English.

During processing, the GUI will display device information, timing details for model loading and audio processing, and finally the transcription result.

## Development

- **gui.py**: Contains the `WhisperGUI` class and related helper functions such as audio recording and transcription. All GUI logic is implemented here.
- **main.py**: Initializes Tkinter and launches the GUI application by importing `WhisperGUI` from `whisper_gui.py`.

Feel free to modify these files or add new modules as you extend the application.

## Troubleshooting

- **FutureWarning from `torch.load`**:
  The application currently suppresses a FutureWarning regarding the use of `torch.load` with `weights_only=False`. This is a temporary measure:
  ```python
  warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")
  ```
  Refer to the [PyTorch SECURITY.md](https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models) for more details.

- **CUDA Not Detected**:
  Make sure that:
  - Your system has a compatible CUDA-enabled GPU.
  - The correct version of PyTorch (with CUDA support) is installed.
  - You are using a compatible Python version (preferably 3.10 or 3.11).

  You can verify CUDA support using:
  ```bash
  python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA not available')"
  ```

- **Python Version Compatibility**:
  While this application is developed and tested using Python 3.10/3.11, Python 3.12 may require additional steps (such as building PyTorch from source) to enable CUDA.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper) for the speech recognition engine.
- [PyTorch](https://pytorch.org/) for GPU-accelerated tensor computation.
- [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/) for audio recording functionality.
- The Python community for invaluable libraries and support.