import tkinter as tk
from tkinter import ttk, filedialog
import threading
import wave
import pyaudio
import numpy as np
from pydub import AudioSegment
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class AudioPlayer:
    def __init__(self):
        self.audio = None
        self.audio_array = None
        self.stream = None
        self.play_thread = None
        self.paused = False
        self.playing = False
        self.position = 0
        self.chunk_size = 1024
        self.file_path = None

    def load_wav(self, file_path):
        if file_path:
            self.file_path = file_path
        self.audio = AudioSegment.from_wav(file_path)
        self.audio_array = np.array(self.audio.get_array_of_samples())
        self.loaded = True

    def play_wav(self, callback_onFinish=None):
        p = pyaudio.PyAudio()
        self.stream = p.open(format=p.get_format_from_width(self.audio.sample_width),
                             channels=self.audio.channels,
                             rate=self.audio.frame_rate,
                             output=True)
        self.playing = True
        self.paused = False
        self.position = 0

        def callback():
            while self.position < len(self.audio_array) and self.playing:
                if not self.paused:
                    end = self.position + self.chunk_size
                    self.stream.write(self.audio_array[self.position:end].tobytes())
                    self.position = end
                else:
                    time.sleep(0.1)
            self.stop_stream()
            if callback_onFinish:
                callback_onFinish()

        self.play_thread = threading.Thread(target=callback)
        self.play_thread.start()

    def pause_wav(self):
        self.paused = not self.paused

    def is_playing(self):
        return self.playing

    def is_ready(self):
        if self.file_path:
            return True
        else:
            return False

    def set_path(self, file_path):
        self.file_path = file_path

    def stop_stream(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        self.playing = False
