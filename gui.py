# whisper_gui.py
import tkinter as tk
from tkinter import filedialog, ttk
import threading
import time
import warnings
import torch
import whisper
import pyaudio
import wave

# Suppress the FutureWarning from torch.load regarding weights_only.
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")


# Helper function to record audio from the microphone.
def record_audio(duration=5, filename="temp_recording.wav", channels=1, rate=16000, chunk=1024):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)
    frames = []
    for _ in range(0, int(rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()
    return filename


# Helper function to process the audio using Whisper.
def process_audio(model, audio_path, task):
    result = model.transcribe(audio_path, task=task, fp16=False)
    return result["text"]


# The GUI class
class WhisperGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Whisper Speech-to-Text and Translation")

        # Define Tkinter variables
        self.input_type = tk.StringVar(value="file")  # "file" or "realtime"
        self.output_type = tk.StringVar(value="cli")  # "cli" (display) or "txt" (save file)
        self.file_path = tk.StringVar()  # stores selected audio file path
        self.model_name = tk.StringVar(value="base")
        self.translate = tk.BooleanVar(value=False)
        self.record_duration = tk.IntVar(value=5)  # seconds for real-time recording

        # --- Input Frame ---
        input_frame = ttk.LabelFrame(root, text="Input")
        input_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        # Radio buttons for input type
        ttk.Label(input_frame, text="Select Input Type:").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(input_frame, text="Audio File", variable=self.input_type, value="file") \
            .grid(row=1, column=0, sticky="w")
        ttk.Radiobutton(input_frame, text="Real-Time Recording", variable=self.input_type, value="realtime") \
            .grid(row=1, column=1, sticky="w")

        # Button and label for audio file selection (for file input)
        self.file_select_button = ttk.Button(input_frame, text="Select Audio File", command=self.select_file)
        self.file_select_button.grid(row=2, column=0, pady=5)
        self.selected_file_label = ttk.Label(input_frame, textvariable=self.file_path)
        self.selected_file_label.grid(row=2, column=1, sticky="w")

        # Entry for real-time recording duration
        ttk.Label(input_frame, text="Recording Duration (sec):").grid(row=3, column=0, sticky="w")
        self.duration_entry = ttk.Entry(input_frame, textvariable=self.record_duration, width=5)
        self.duration_entry.grid(row=3, column=1, sticky="w")

        # --- Output Frame ---
        output_frame = ttk.LabelFrame(root, text="Output")
        output_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        ttk.Label(output_frame, text="Select Output Type:").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(output_frame, text="Display on CLI", variable=self.output_type, value="cli") \
            .grid(row=1, column=0, sticky="w")
        ttk.Radiobutton(output_frame, text="Save to TXT File", variable=self.output_type, value="txt") \
            .grid(row=1, column=1, sticky="w")

        # --- Options Frame ---
        options_frame = ttk.LabelFrame(root, text="Options")
        options_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")
        ttk.Label(options_frame, text="Model:").grid(row=0, column=0, sticky="w")
        model_menu = ttk.OptionMenu(options_frame, self.model_name, self.model_name.get(), "tiny", "base", "small",
                                    "medium", "large")
        model_menu.grid(row=0, column=1, sticky="w")
        self.translate_check = ttk.Checkbutton(options_frame, text="Translate to English", variable=self.translate)
        self.translate_check.grid(row=1, column=0, sticky="w")

        # --- Process Button ---
        self.process_button = ttk.Button(root, text="Process", command=self.process)
        self.process_button.grid(row=3, column=0, padx=10, pady=10)

        # --- Text Widget for Output Display ---
        self.output_text = tk.Text(root, wrap="word", height=10, width=60)
        self.output_text.grid(row=4, column=0, padx=10, pady=10)

    def select_file(self):
        file = filedialog.askopenfilename(title="Select Audio File", filetypes=[("Audio Files", "*.wav *.mp3 *.m4a")])
        if file:
            self.file_path.set(file)

    def process(self):
        # Disable button to prevent multiple clicks
        self.process_button.config(state="disabled")
        self.output_text.delete("1.0", tk.END)
        # Use a thread to avoid freezing the GUI
        threading.Thread(target=self.run_processing).start()

    def run_processing(self):
        try:
            # Start timing the process
            overall_start = time.time()

            # Set device using CUDA if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.append_output(f"Using hardware: {device}\n")

            # Timing the model load
            load_start = time.time()
            self.append_output(f"Loading Whisper model '{self.model_name.get()}'...\n")
            model = whisper.load_model(self.model_name.get(), device=device)
            load_time = time.time() - load_start
            self.append_output(f"Model loaded in {load_time:.2f} seconds.\n")

            task = "translate" if self.translate.get() else "transcribe"
            self.append_output(f"Task: {task}\n")

            # Determine input source: file or real-time
            if self.input_type.get() == "file":
                audio_path = self.file_path.get()
                if not audio_path:
                    self.append_output("No audio file selected!\n")
                    self.process_button.config(state="normal")
                    return
            else:
                duration = self.record_duration.get()
                self.append_output(f"Recording audio for {duration} seconds...\n")
                audio_path = record_audio(duration=duration)
                self.append_output("Recording complete.\n")

            # Timing the transcription process
            transcribe_start = time.time()
            self.append_output(f"Processing audio file: {audio_path}\n")
            transcription = process_audio(model, audio_path, task)
            transcribe_time = time.time() - transcribe_start

            self.append_output("\n--- Transcription ---\n")
            self.append_output(transcription + "\n")
            self.append_output(f"\nTranscription completed in {transcribe_time:.2f} seconds.\n")

            overall_time = time.time() - overall_start
            self.append_output(f"\nOverall process took {overall_time:.2f} seconds.\n")

            # If output type is set to TXT, ask for a save path and write the file
            if self.output_type.get() == "txt":
                save_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                                         filetypes=[("Text Files", "*.txt")],
                                                         title="Save Transcription As")
                if save_path:
                    with open(save_path, "w", encoding="utf-8") as f:
                        f.write(transcription)
                    self.append_output(f"\nTranscription saved to {save_path}\n")
        except Exception as e:
            self.append_output(f"Error: {e}\n")
        finally:
            self.process_button.config(state="normal")

    def append_output(self, text):
        self.output_text.insert(tk.END, text)
        self.output_text.see(tk.END)
