import tkinter as tk
import threading
import wave
import pyaudio
import numpy as np
import time
import math
import matplotlib.pyplot as plt

from tkinter import ttk, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.fft import fft, fftfreq
from pydub import AudioSegment

FREQUENCES = [32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000]
SPECTRUNS = [
    (    0,    48), # 32Hz 
    (   48,    96), # 64Hz
    (   96,   192), # 128Hz
    (  192,   384), # 256Hz
    (  384,   756), # 512Hz
    (  756,  1500), # 1KHz
    ( 1500,  3000), # 2KHz
    ( 3000,  6000), # 4KHz
    ( 6000, 12000), # 8KHz
    (12000, 20000)  # 16KHz
]

class AudioPlayer:
    def __init__(self):
        self.audio = None
        self.audio_array = None
        self.original_audio_array = None
        self.spectrum_amplitudes = []
        self.max_spectrum_amplitudes = []   # max_spectrum_amplitude * 4
        self.stream = None
        self.play_thread = None
        self.loaded = False
        self.paused = False
        self.playing = False
        self.position = 0
        self.chunk_size = 1024
        self.file_path = None

        self.sampwidth = None
        self.n_channels = None
        self.framerate = None
        self.n_frames = None

    def load_wav(self, file_path, verbose=False):
        self.loaded = False
        if file_path:
            self.file_path = file_path
            self.audio = AudioSegment.from_wav(file_path)
            self.sampwidth = self.audio.sample_width
            self.n_channels = self.audio.channels
            self.framerate = self.audio.frame_rate
            self.n_frames = int(self.audio.frame_count())
            self.audio_array = np.array(self.audio.get_array_of_samples())
            self.original_audio_array = np.copy(self.audio_array)
            self.set_audio_array(self.audio_array)
            self.loaded = True
            if verbose:
                print("sampwidth:", self.sampwidth)
                print("n_channels:", self.n_channels)
                print("framerate:", self.framerate)
                print("n_frames:", self.n_frames)


    def set_audio_array(self, array):
        self.audio_array = array
        self.spectrum_amplitudes.clear()

        amplitudes = []
        low = 0
        high = 0
        j = 0
        for i in range(len(self.audio_array)):
            if i > 0 and math.remainder(i, 4410) == 0: # for each 4410 samples (100 miliseconds) compute...
                low = high
                high = i
                array = self.audio_array[low:high]                
                if self.loaded:
                    if i == 4410:
                        print("Serão utilizadas as amplitudes já carregadas!")
                    max_ampl_val = self.max_spectrum_amplitudes[j]
                    amplitudes, _ = self.normalized_spectrum_amplitudes(array, self.framerate, max_ampl_val)
                else:
                    if i == 4410:
                        print("As amplitudes de referência serão carregadas...")
                    amplitudes, max_values = self.normalized_spectrum_amplitudes(array, self.framerate)
                    self.max_spectrum_amplitudes.append(max_values)

                self.spectrum_amplitudes.append(amplitudes)
                j += 1
        

    def normalized_spectrum_amplitudes(self, signal, framerate, max_ampl_spect_values=None): 
        N = len(signal)
        yf_fft = fft(signal)
        xf_fft = fftfreq(N, 1 / framerate)
        yf = np.abs(yf_fft[:N // 2])
        xf = xf_fft[:N // 2]
        j = 0
        start = 0

        compute_max_values = False
        max_values = [None] * len(SPECTRUNS)
        if not self.loaded: # If the wave is not loaded, freq. spectrum was not given, prepare to compute it. 
            compute_max_values = True
        else:
            max_values = max_ampl_spect_values

        amplitudes = [None] * len(SPECTRUNS)
        for j in range(len(SPECTRUNS)):
            amplitude = 0
            count = 0
            max_value = 0
            for i in range(start, N):
                if xf[i] >= SPECTRUNS[j][0] and xf[i] < SPECTRUNS[j][1]:
                    if yf[i] > max_value:
                        max_value = yf[i]
                    amplitude += yf[i]
                    count += 1
                else:
                    start = i + 1
                    break

            if compute_max_values:
                # We multiply the max_value by 4 because the biggest gain/loss is +/- 12dB, 
                # which amplify/attenuate the signal by 4.
                max_values[j] = (max_value * 2) 

            if count == 0 or max_values[j] == 0:
                print("Não foi encontrada amplitude!")
                amplitudes[j] = 0
            else:
                # nomalize the amplitude between 0 and 1
                amplitudes[j]= float(amplitude/(max_values[j] * count))
        return amplitudes, max_values


    def play_wav(self, progress_vars, callback_onFinish=None):
        p = pyaudio.PyAudio()
        self.stream = p.open(format=p.get_format_from_width(self.sampwidth),
                             channels=self.n_channels,
                             rate=self.framerate,
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
            p.terminate()

            if callback_onFinish:
                callback_onFinish()

        self.play_thread = threading.Thread(target=callback)
        self.play_thread.start()

        self.spectrum_pos = 0
        def callback_spectrum():
            while self.spectrum_pos < len(self.spectrum_amplitudes) and self.playing:
                if not self.paused:
                    self.spectrum_pos += 1
                    values = self.spectrum_amplitudes[self.spectrum_pos]
                    for i in range(len(values)):
                        progress_vars[i].set(math.ceil(values[i]*100))
                    time.sleep(0.1)
                else:
                    time.sleep(0.1)
            self.stop_stream()
            p.terminate()

            if callback_onFinish:
                callback_onFinish()

        self.spectrum_thread = threading.Thread(target=callback_spectrum)
        self.spectrum_thread.start()

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
