"""
Tutorial de uso do Tkinter:    https://www.pythontutorial.net/tkinter/

Required libraries:
- matplotlib
- numpy
- pydub
- pyaudio
"""

import tkinter as tk

from tkinter import ttk # newer themed widgets
from tkinter import font
from tkinter import filedialog

import matplotlib.pyplot as plt
import matplotlib
import wave
import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from audio import AudioPlayer

from scipy.fft import fft, fftfreq


"""
Global variables
"""
sliders = [None] * 10
slider_labels = [None] * 10
sliders_freqs = [
        (   16,    48), # 32Hz 
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
audio_player = AudioPlayer()


"""
Callback functions
"""
def onClick_Open(wave_frame, dft_frame, frm_command, label_file):
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if file_path:
        audio_player.load_wav(file_path)
        label_file.configure(text=file_path)
    
        update_graph_frames(wave_frame, dft_frame, audio_player)    

        for widget in frm_command.winfo_children():
            widget.state(['!disabled'])


def onClick_Apply(wave_frame, dft_frame, btn_save, audio_player: AudioPlayer):
    slider_values = [float(slider.get()) for slider in sliders]

    if audio_player.loaded:
        fs = audio_player.framerate
        M = 1001
        filter_sum = np.zeros(M)
        print("Equalizer values:", slider_values)
        for i in range(len(sliders)):
            coef = np.zeros(M)
            lowcut = sliders_freqs[i][0]
            highcut = sliders_freqs[i][1]
            coef += bandpass_filter(lowcut, highcut, fs, M)

            slider_value_dB = float(sliders[i].get())
            filter_sum += amplify_attenuate_db(coef, slider_value_dB)
    
        audio_array = audio_player.audio_array
        audio_array = apply_filter(audio_array, filter_sum)
        audio_player.audio_array = audio_array.astype(np.int16)
        update_graph_frames(wave_frame, dft_frame, audio_player)
        btn_save.state(['!disabled'])


def onChange_Repeat():
    print("Checkbox: ", var_check.get())


def onClick_Reset():
    for i in range(len(sliders)):
        sliders[i].set(0)
        slider_labels[i].configure(text=str(formated_value(0)) + 'dB')


def onChanged_Slider00(event):
    j = 0
    slider_labels[j].configure(text=str(formated_value(sliders[j].get())) + ' dB')

def onChanged_Slider01(event):
    j = 1
    slider_labels[j].configure(text=str(formated_value(sliders[j].get())) + ' dB')

def onChanged_Slider02(event):
    j = 2
    slider_labels[j].configure(text=str(formated_value(sliders[j].get())) + ' dB')

def onChanged_Slider03(event):
    j = 3
    slider_labels[j].configure(text=str(formated_value(sliders[j].get())) + ' dB')

def onChanged_Slider04(event):
    j = 4
    slider_labels[j].configure(text=str(formated_value(sliders[j].get())) + ' dB')

def onChanged_Slider05(event):
    j = 5
    slider_labels[j].configure(text=str(formated_value(sliders[j].get())) + ' dB')

def onChanged_Slider06(event):
    j = 6
    slider_labels[j].configure(text=str(formated_value(sliders[j].get())) + ' dB')

def onChanged_Slider07(event):
    j = 7
    slider_labels[j].configure(text=str(formated_value(sliders[j].get())) + ' dB')

def onChanged_Slider08(event):
    j = 8
    slider_labels[j].configure(text=str(formated_value(sliders[j].get())) + ' dB')

def onChanged_Slider09(event):
    j = 9
    slider_labels[j].configure(text=str(formated_value(sliders[j].get())) + ' dB')


def onClick_Save(audio_player: AudioPlayer):
    file_path = filedialog.asksaveasfilename(defaultextension=".wav",
                                             filetypes=[("Wave files", "*.wav"), ("All files", "*.*")])
    if file_path:
        print("Selected file path:", file_path)
        save_wav(file_path, audio_player.audio_array, audio_player.framerate)
    else:
        print("No file selected")    


def onClick_Exit():
    audio_player.stop_stream()
    root.destroy()

def audio_finish(btn_play, btn_pause):
    print('Terminou o WAV.')
    btn_pause.state(['disabled'])
    btn_play.state(['!disabled'])

def pause_audio(btn_pause, btn_play):
    print('Pause!')
    audio_player.pause_wav()
    # set the disabled flag
    btn_pause.state(['disabled'])
    btn_play.state(['!disabled'])

# Function to play the WAV file using a buffer
def play_audio(btn_play, btn_pause):
    print('Play...')
    if audio_player.is_ready():
        if audio_player.is_playing():
            audio_player.pause_wav()
        else:
            audio_player.play_wav(lambda:audio_finish(btn_play, btn_pause))
    
    # set the disabled flag
    btn_play.state(['disabled'])
    btn_pause.state(['!disabled'])


"""
Functions
"""
def bandpass_filter(lowcut, highcut, fs, numtaps, verbose=False):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    taps = np.zeros(numtaps)
    for i in range(numtaps):
        n = i - (numtaps - 1) / 2
        if n == 0:
            taps[i] = 2 * (high - low)
        else:
            taps[i] = (np.sin(2 * np.pi * high * n) - np.sin(2 * np.pi * low * n)) / (np.pi * n)
    taps *= np.hamming(numtaps)
    if verbose:
        print('bandpass_filter(...):')
        print('    lowcut:', low)
        print('    highcut:', high)
        print('    framerate:', fs)
        print('    n. taps:', numtaps)
        print('    return(taps):', taps.shape)
    return taps


def apply_filter(filter_coef, filter_taps, verbose=False):
    convolved = np.convolve(filter_coef, filter_taps, mode='same')
    if verbose:
        print('apply_filter(...):')
        print('    filter_coef:', filter_coef.shape)
        print('    filter_taps:', filter_taps.shape)
        print('    return(convolved):', convolved.shape)
    return convolved


def amplify_attenuate_db(signal, db, verbose=False):
    linear_gain = 10**(db / 20)
    final_signal = signal * linear_gain
    if verbose:
        print('amplify_attenuate(...):')
        print('    signal:', signal.shape)
        print('    db:', db)
        print('    return(final_signal):', final_signal.shape)
    return final_signal


def update_graph_frames(wave_frame, dft_frame, audio_player):
        for widget in wave_frame.winfo_children():
            widget.destroy()

        for widget in dft_frame.winfo_children():
            widget.destroy()

        plot_waveform(wave_frame, audio_player)
        plot_dft(dft_frame, audio_player)


def formated_value(value):
    return '{: .2f}'.format(value)


def plot_waveform(frame, audio_player: AudioPlayer):
    matplotlib.use('TkAgg')

    # create a figure
    figure = Figure(figsize=(6, 1.85), dpi=100)
    figure.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)  # Adjust margins

    # create axes
    axes = figure.add_subplot()
    axes.set_xlabel("Time [s]", fontsize=8)
    axes.set_ylabel("Amplitude", fontsize=8)

    # Set the font size of the tick labels
    axes.tick_params(axis='both', which='major', labelsize=8)

    if (audio_player):
        n_frames = audio_player.n_frames
        framerate = audio_player.framerate

        # Convert the audio data to numpy array
        audio_array = audio_player.audio_array
        
        # Create time array for x-axis
        time_array = np.linspace(0, n_frames/framerate, num=n_frames)
        
        # create the waveform plot
        axes.plot(time_array, audio_array)
    else:
        # create the placeholder plot
        axes.axhline(y=10, color='blue', linestyle='-')

    # Use tight layout to avoid any cutoff
    figure.tight_layout()

    # create FigureCanvasTkAgg object
    figure_canvas = FigureCanvasTkAgg(figure, frame)
    figure_canvas.draw()
    figure_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)


def plot_dft(frame, audio_player: AudioPlayer):
    matplotlib.use('TkAgg')

    # create a figure
    figure = Figure(figsize=(6, 1.85), dpi=100)
    figure.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)  # Adjust margins

    # create axes
    axes = figure.add_subplot()
    axes.set_xlabel("Frequency [Hz]", fontsize=8)
    axes.set_ylabel("Magnitude (dB)", fontsize=8)

    # Set the font size of the tick labels
    axes.tick_params(axis='both', which='major', labelsize=8)

    if (audio_player):
        framerate = audio_player.framerate

        # Convert the audio data to numpy array
        audio_array = audio_player.audio_array

        # Compute the DFT
        N = len(audio_array)
        yf = fft(audio_array)
        xf = fftfreq(N, 1 / framerate)

        # The magnitude spectrum of the DFT is plotted. Only the positive frequencies up 
        # to the Nyquist frequency are plotted (xf[:N // 2] and np.abs(yf[:N // 2])).
        axes.plot(xf[:N // 2], np.abs(yf[:N // 2]))
    else:
        # create the placeholder plot
        axes.axhline(y=10, color='blue', linestyle='-')

    # Use tight layout to avoid any cutoff
    figure.tight_layout()

    # create FigureCanvasTkAgg object
    figure_canvas = FigureCanvasTkAgg(figure, frame)
    figure_canvas.draw()
    figure_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)


def add_frequence_labels(container, row, font_style):
    ttk.Label(container, text=' 32Hz', font=font_style).grid(column=0, row=row, sticky=tk.N)
    ttk.Label(container, text=' 64Hz', font=font_style).grid(column=1, row=row, sticky=tk.N)
    ttk.Label(container, text='128Hz', font=font_style).grid(column=2, row=row, sticky=tk.N)
    ttk.Label(container, text='256Hz', font=font_style).grid(column=3, row=row, sticky=tk.N)
    ttk.Label(container, text='512Hz', font=font_style).grid(column=4, row=row, sticky=tk.N)
    ttk.Label(container, text=' 1KHz', font=font_style).grid(column=5, row=row, sticky=tk.N)
    ttk.Label(container, text=' 2KHz', font=font_style).grid(column=6, row=row, sticky=tk.N)
    ttk.Label(container, text=' 4KHz', font=font_style).grid(column=7, row=row, sticky=tk.N)
    ttk.Label(container, text=' 8KHz', font=font_style).grid(column=8, row=row, sticky=tk.N)
    ttk.Label(container, text='16KHz', font=font_style).grid(column=9, row=row, sticky=tk.N)


def create_equalizer_frame(container, row):
    frame = ttk.Frame(container)

    # grid layout for the input frame
    frame.columnconfigure(0, weight=8)
    frame.columnconfigure(1, weight=2)

    # Create inner frames
    frm_left = ttk.Frame(frame)
    frm_right = ttk.Frame(frame)

    # Create widgets 
    btn_open = ttk.Button(frm_left, text='Open')
    lbl_file = ttk.Label(frm_left, text='WAVE file...')
    btn_apply = ttk.Button(frm_right, text='Apply')
    btn_reset = ttk.Button(frm_right, text='Reset')

    # Place widgets
    frame.grid(column=0, row=row, columnspan=10, sticky=tk.EW, padx=10, pady=5)
    frm_left.grid(column=0, row=0, sticky=tk.W)
    btn_open.pack(side=tk.LEFT)
    lbl_file.pack(side=tk.LEFT, padx=5)
    frm_right.grid(column=1, row=0, sticky=tk.E)
    btn_apply.pack(side=tk.RIGHT)
    btn_reset.pack(side=tk.RIGHT)
    return lbl_file, btn_open, btn_reset, btn_apply


def create_wave_frame(container, row):
    frame = ttk.Frame(container)
    frame.columnconfigure(0, weight=1)

    lbl_gph_wave = ttk.Label(frame, text='WAVE graph')
    frm_graph = ttk.Frame(frame)
    frm_graph['borderwidth'] = 5
    frm_graph['relief'] = 'sunken'
    plot_waveform(frm_graph, None)
    lbl_gph_wave.grid(column=0, row=0, sticky=tk.W)
    frm_graph.grid(column=0, row=1, sticky=(tk.W, tk.E, tk.N, tk.S))

    lbl_gph_dft = ttk.Label(frame, text='Discrete Fourier Transform (DTF) graph')
    frm_dft_graph = ttk.Frame(frame)
    frm_dft_graph['borderwidth'] = 5
    frm_dft_graph['relief'] = 'sunken'
    plot_dft(frm_dft_graph, None)
    lbl_gph_dft.grid(column=0, row=2, sticky=tk.W)
    frm_dft_graph.grid(column=0, row=3, sticky=(tk.W, tk.E, tk.N, tk.S))

    frame.grid(column=0, row=row, columnspan=10, sticky=tk.EW, padx=10, pady=5)

    return frm_graph, frm_dft_graph


def create_command_frame(container, row, checked_var):
    frame = ttk.Frame(container)

    # grid layout for the input frame
    frame.columnconfigure(0, weight=1)
    frame.columnconfigure(1, weight=1)

    frm_left  = ttk.Frame(frame)
    btn_check = ttk.Checkbutton(frm_left, text='Repeat', variable=checked_var)
    btn_play  = ttk.Button(frm_left, text='Play')
    btn_pause = ttk.Button(frm_left, text='Pause')

    frm_right = ttk.Frame(frame)
    btn_save  = ttk.Button(frm_right, text='Save')
    btn_exit  = ttk.Button(frm_right, text='Exit')

    btn_play.state(['disabled'])
    btn_pause.state(['disabled'])
    btn_check.state(['disabled'])
    btn_save.state(['disabled'])

    btn_play.pack(side=tk.LEFT)
    btn_pause.pack(side=tk.LEFT)
    btn_check.pack(side=tk.LEFT)
    btn_exit.pack(side=tk.RIGHT)
    btn_save.pack(side=tk.RIGHT)
    frm_left.grid(column=0, row=0, sticky=tk.W)
    frm_right.grid(column=1, row=0, sticky=tk.E)
    
    frame.grid(column=0, row=row, columnspan=10, sticky=tk.EW, padx=10, pady=5)
    return frm_left, btn_play, btn_pause, btn_check, btn_exit, btn_save


def quit(player):
    player.stop_stream()
    onClick_Exit()


def save_wav(file_name, signal, framerate):
    signal = signal.astype(np.int16)
    with wave.open(file_name, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(framerate)
        wav_file.writeframes(signal.tobytes())
    return signal


"""
Equalizer code
"""
root = tk.Tk()

# window attributes
root.title("Equalizer")
root.geometry("900x690+0+0")
root.resizable(False, False)
root.iconbitmap("logo_unb.ico")

# configure the grid
root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=1)
root.columnconfigure(2, weight=1)
root.columnconfigure(3, weight=1)
root.columnconfigure(4, weight=1)
root.columnconfigure(5, weight=1)
root.columnconfigure(6, weight=1)
root.columnconfigure(7, weight=1)
root.columnconfigure(8, weight=1)
root.columnconfigure(9, weight=1)

# First line frame (file path, open button and apply button)
font_style = font.Font(size=8, weight="bold")
add_frequence_labels(root, row=0, font_style=font_style)

for i in range(0, 10):
    ttk.Label(root, text='+6').grid(column=i, row=1, sticky=tk.N)

# creating sliders widget
for i in range(len(sliders)):
    sliders[i] = ttk.Scale(root, from_=6, to=-6, orient='vertical', variable=tk.DoubleVar())
    sliders[i].grid(column=i, row=2, sticky=tk.N)    

sliders[0].configure(command=onChanged_Slider00)
sliders[1].configure(command=onChanged_Slider01)
sliders[2].configure(command=onChanged_Slider02)
sliders[3].configure(command=onChanged_Slider03)
sliders[4].configure(command=onChanged_Slider04)
sliders[5].configure(command=onChanged_Slider05)
sliders[6].configure(command=onChanged_Slider06)
sliders[7].configure(command=onChanged_Slider07)
sliders[8].configure(command=onChanged_Slider08)
sliders[9].configure(command=onChanged_Slider09)

for i in range(0, 10):
    ttk.Label(root, text='-6').grid(column=i, row=3, sticky=tk.N)

# Create labels to receive the sliders values
for i in range(len(slider_labels)):
    slider_labels[i] = ttk.Label(root, text='0.00 dB', font=font_style, width=7)
    slider_labels[i].grid(column=i, row=4, sticky=tk.N)

lbl_file, btn_open, btn_reset, btn_apply = create_equalizer_frame(root, row=5)

frm_graph, frm_dft_graph = create_wave_frame(root, row=6)

var_check = tk.IntVar()
frm_command, btn_play, btn_pause, btn_check, btn_exit, btn_save = create_command_frame(root, 7, var_check)

# Event configuration (command)
btn_open.configure(command=lambda: onClick_Open(frm_graph, frm_dft_graph, frm_command, lbl_file))
btn_reset.configure(command=onClick_Reset)
btn_apply.configure(command=lambda: onClick_Apply(frm_graph, frm_dft_graph, btn_save, audio_player))
btn_play.configure(command=lambda:play_audio(btn_play, btn_pause))
btn_pause.configure(command=lambda:play_audio(btn_pause, btn_play))
btn_check.configure(command=onChange_Repeat)
btn_exit.configure(command=lambda:quit(audio_player))
btn_save.configure(command=lambda:onClick_Save(audio_player))

# code to run across multi platforms
try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1) # it runs on Windows, but not on macOS or Linux
finally:
    # keep the window displaying
    root.mainloop()