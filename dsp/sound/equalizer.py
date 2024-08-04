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
Callback functions
"""
def open_file(graph_frame, dft_frame, frm_command, label_file):
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if file_path:
        audio_player.load_wav(file_path)
        label_file.configure(text=file_path)
        
        for widget in graph_frame.winfo_children():
            widget.destroy()

        for widget in dft_frame.winfo_children():
            widget.destroy()

        plot_waveform(graph_frame, file_path)
        plot_dft(dft_frame, file_path)

        for widget in frm_command.winfo_children():
            widget.state(['!disabled'])


def agreement_changed():
    print("Checkbox: ", checked.get())

def slider01_changed(event):
    slider01_label.configure(text=formated_value(value_s01.get()))

def slider02_changed(event):
    slider02_label.configure(text=formated_value(value_s02.get()))

def slider03_changed(event):
    slider03_label.configure(text=formated_value(value_s03.get()))

def slider04_changed(event):
    slider04_label.configure(text=formated_value(value_s04.get()))

def slider05_changed(event):
    slider05_label.configure(text=formated_value(value_s05.get()))

def slider06_changed(event):
    slider06_label.configure(text=formated_value(value_s06.get()))

def slider07_changed(event):
    slider07_label.configure(text=formated_value(value_s07.get()))

def slider08_changed(event):
    slider08_label.configure(text=formated_value(value_s08.get()))

def slider09_changed(event):
    slider09_label.configure(text=formated_value(value_s09.get()))

def slider10_changed(event):
    slider10_label.configure(text=formated_value(value_s10.get()))

def exit():
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
def formated_value(value):
    return '{: .2f}'.format(value)


def plot_waveform(frame, file_path):
    matplotlib.use('TkAgg')

    # create a figure
    figure = Figure(figsize=(6, 1.75), dpi=100)
    figure.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)  # Adjust margins

    # create axes
    axes = figure.add_subplot()
    axes.set_xlabel("Time [s]", fontsize=8)
    axes.set_ylabel("Amplitude", fontsize=8)

    # Set the font size of the tick labels
    axes.tick_params(axis='both', which='major', labelsize=8)

    print('Path:', file_path)
    if file_path:
        print('Opening WAV file...')
        with wave.open(file_path, 'rb') as wav_file:
            # Extract audio parameters
            n_channels = wav_file.getnchannels()
            sampwidth = wav_file.getsampwidth()
            framerate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            
            # Read the audio data
            audio_data = wav_file.readframes(n_frames)
        
        print('Closing WAV file...')
        # Convert the audio data to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # If stereo, take only one channel for simplicity
        if n_channels == 2:
            audio_array = audio_array[::2]

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


def plot_dft(frame, file_path):
    matplotlib.use('TkAgg')

    # create a figure
    figure = Figure(figsize=(6, 1.75), dpi=100)
    figure.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)  # Adjust margins

    # create axes
    axes = figure.add_subplot()
    axes.set_xlabel("Frequency [Hz]", fontsize=8)
    axes.set_ylabel("Magnitude", fontsize=8)

    # Set the font size of the tick labels
    axes.tick_params(axis='both', which='major', labelsize=8)

    print('Path:', file_path)
    if file_path:
        print('Opening WAV file...')
        with wave.open(file_path, 'rb') as wav_file:
            # Extract audio parameters
            n_channels = wav_file.getnchannels()
            sampwidth = wav_file.getsampwidth()
            framerate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            
            # Read the audio data
            audio_data = wav_file.readframes(n_frames)
        
        print('Closing WAV file...')
        # Convert the audio data to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # If stereo, take only one channel for simplicity
        if n_channels == 2:
            audio_array = audio_array[::2]

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


def create_file_frame(container, row):
    frame = ttk.Frame(container)

    # grid layout for the input frame
    frame.columnconfigure(0, weight=8)
    frame.columnconfigure(1, weight=2)
    
    # Create inner frames
    frm_label = ttk.Frame(frame)
    frm_button = ttk.Frame(frame)

    # Create widgets 
    lbl_file = ttk.Label(frm_label, text='WAVE file...')
    btn_open = ttk.Button(frm_button, text='Open')

    # Place widgets into the inner frames
    lbl_file.pack(side=tk.LEFT)
    btn_open.pack(side=tk.RIGHT)
    frm_label.grid(column=0, row=0, sticky=tk.W)
    frm_button.grid(column=1, row=0, sticky=tk.E)
    frame.grid(column=0, row=row, columnspan=10, sticky=tk.EW, padx=10, pady=5)

    return lbl_file, btn_open


def create_equalizer_frame(container, row):
    frame = ttk.Frame(container)
    btn_apply = ttk.Button(frame, text='Apply')
    btn_apply.pack(side=tk.RIGHT)
    frame.grid(column=0, row=row, columnspan=10, sticky=tk.E, padx=10, pady=5)
    return btn_apply


def create_graph_frame(container, row):
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
    btn_exit  = ttk.Button(frame, text='Exit')

    btn_play.state(['disabled'])
    btn_pause.state(['disabled'])
    btn_check.state(['disabled'])

    btn_play.pack(side=tk.LEFT)
    btn_pause.pack(side=tk.LEFT)
    btn_check.pack(side=tk.LEFT)
    frm_left.grid(column=0, row=0, sticky=tk.W)
    btn_exit.grid(column=1, row=0, sticky=tk.E)
    
    frame.grid(column=0, row=row, columnspan=10, sticky=tk.EW, padx=10, pady=5)
    return frm_left, btn_play, btn_pause, btn_check, btn_exit


"""
Equalizer code
"""
root = tk.Tk()

# window attributes
root.title("Equalizer")
root.geometry("950x690+0+0")
root.resizable(False, False)
root.iconbitmap("logo_unb.ico")

# Creating a audio player
audio_player = AudioPlayer()

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
lbl_file, btn_open = create_file_frame(root, row=0)

# creating slider bars
value_s01 = tk.DoubleVar()
value_s02 = tk.DoubleVar()
value_s03 = tk.DoubleVar()
value_s04 = tk.DoubleVar()
value_s05 = tk.DoubleVar()
value_s06 = tk.DoubleVar()
value_s07 = tk.DoubleVar()
value_s08 = tk.DoubleVar()
value_s09 = tk.DoubleVar()
value_s10 = tk.DoubleVar()

slider01 = ttk.Scale(root, from_=6, to=-6, orient='vertical', command=slider01_changed, variable=value_s01)
slider02 = ttk.Scale(root, from_=6, to=-6, orient='vertical', command=slider02_changed, variable=value_s02)
slider03 = ttk.Scale(root, from_=6, to=-6, orient='vertical', command=slider03_changed, variable=value_s03)
slider04 = ttk.Scale(root, from_=6, to=-6, orient='vertical', command=slider04_changed, variable=value_s04)
slider05 = ttk.Scale(root, from_=6, to=-6, orient='vertical', command=slider05_changed, variable=value_s05)
slider06 = ttk.Scale(root, from_=6, to=-6, orient='vertical', command=slider06_changed, variable=value_s06)
slider07 = ttk.Scale(root, from_=6, to=-6, orient='vertical', command=slider07_changed, variable=value_s07)
slider08 = ttk.Scale(root, from_=6, to=-6, orient='vertical', command=slider08_changed, variable=value_s08)
slider09 = ttk.Scale(root, from_=6, to=-6, orient='vertical', command=slider09_changed, variable=value_s09)
slider10 = ttk.Scale(root, from_=6, to=-6, orient='vertical', command=slider10_changed, variable=value_s10)

for i in range(0, 10):
    ttk.Label(root, text='+6').grid(column=i, row=1, sticky=tk.N)
slider01.grid(column=0, row=2, sticky=tk.N, padx=4)
slider02.grid(column=1, row=2, sticky=tk.N, padx=4)
slider03.grid(column=2, row=2, sticky=tk.N, padx=4)
slider04.grid(column=3, row=2, sticky=tk.N, padx=4)
slider05.grid(column=4, row=2, sticky=tk.N, padx=4)
slider06.grid(column=5, row=2, sticky=tk.N, padx=4)
slider07.grid(column=6, row=2, sticky=tk.N, padx=4)
slider08.grid(column=7, row=2, sticky=tk.N, padx=4)
slider09.grid(column=8, row=2, sticky=tk.N, padx=4)
slider10.grid(column=9, row=2, sticky=tk.N, padx=4)
for i in range(0, 10):
    ttk.Label(root, text='-6').grid(column=i, row=3, sticky=tk.N)

font_style = font.Font(size=10, weight="bold")
slider01_label = ttk.Label(root, text='0', font=font_style)
slider01_label.grid(column=0, row=4, sticky=tk.N)
slider02_label = ttk.Label(root, text='0', font=font_style)
slider02_label.grid(column=1, row=4, sticky=tk.N)
slider03_label = ttk.Label(root, text='0', font=font_style)
slider03_label.grid(column=2, row=4, sticky=tk.N)
slider04_label = ttk.Label(root, text='0', font=font_style)
slider04_label.grid(column=3, row=4, sticky=tk.N)
slider05_label = ttk.Label(root, text='0', font=font_style)
slider05_label.grid(column=4, row=4, sticky=tk.N)
slider06_label = ttk.Label(root, text='0', font=font_style)
slider06_label.grid(column=5, row=4, sticky=tk.N)
slider07_label = ttk.Label(root, text='0', font=font_style)
slider07_label.grid(column=6, row=4, sticky=tk.N)
slider08_label = ttk.Label(root, text='0', font=font_style)
slider08_label.grid(column=7, row=4, sticky=tk.N)
slider09_label = ttk.Label(root, text='0', font=font_style)
slider09_label.grid(column=8, row=4, sticky=tk.N)
slider10_label = ttk.Label(root, text='0', font=font_style)
slider10_label.grid(column=9, row=4, sticky=tk.N)

btn_apply = create_equalizer_frame(root, row=5)

frm_graph, frm_dft_graph = create_graph_frame(root, row=6)

var_check = tk.IntVar()
frm_command, btn_play, btn_pause, btn_check, btn_exit = create_command_frame(root, 7, var_check)

# Event configuration (command)
btn_open.configure(command=lambda: open_file(frm_graph, frm_dft_graph, frm_command, lbl_file))
btn_play.configure(command=lambda:play_audio(btn_play, btn_pause))
btn_pause.configure(command=lambda:play_audio(btn_pause, btn_play))
btn_check.configure(command=agreement_changed)
btn_exit.configure(command=exit)

# code to run across multi platforms
try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1) # it runs on Windows, but not on macOS or Linux
finally:
    # keep the window displaying
    root.mainloop()