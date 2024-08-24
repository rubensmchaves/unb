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
import math
import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from audio import AudioPlayer
from audio import SPECTRUNS
from graphs import plot_filters
from graphs import plot_filter
from graphs import my_graphs

from scipy.fft import fft, fftfreq
from scipy.signal import firwin


"""
Global variables
"""
sliders = [None] * 10
slider_labels = [None] * 10
filters = [None] * 10
sliders_freqs = SPECTRUNS
audio_player = AudioPlayer()
frequences = ['32Hz', '64Hz', '128Hz', '256Hz', '512Hz', '1KHz', '2KHz', '4KHz', '8KHz', '16KHz']

"""
Callback functions
"""
def onClick_Open(wave_frame, dft_frame, frm_command, btn_apply, btn_filters, label_file):
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if file_path:
        audio_player.load_wav(file_path, verbose=True)
        label_file.configure(text=file_path)
    
        update_graph_frames(wave_frame, dft_frame, audio_player)    

        for widget in frm_command.winfo_children():
            widget.state(['!disabled'])
        btn_apply.state(['!disabled'])
        btn_filters.state(['!disabled'])


def onClick_Apply(wave_frame, dft_frame, btn_save, audio_player: AudioPlayer):

    if audio_player.loaded:
        fs = audio_player.framerate
        filters, M = create_filters(fs)
        filter_sum = np.zeros(M)
        for i in range(len(sliders)):
            slider_value_dB = float(sliders[i].get())
            filters[i] = amplify_attenuate_db(filters[i], slider_value_dB)
            filter_sum += filters[i]
        
        audio_array = audio_player.original_audio_array
        audio_array = apply_filter(audio_array, filter_sum)
        audio_player.set_audio_array(audio_array.astype(np.int16)) # numpy.ndarray
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
def play_audio(btn_play, btn_pause, progress_vars):
    print('Play...')
    if audio_player.is_ready():
        if audio_player.is_playing():
            audio_player.pause_wav()
        else:
            audio_player.play_wav(progress_vars, lambda:audio_finish(btn_play, btn_pause))
    
    # set the disabled flag
    btn_play.state(['disabled'])
    btn_pause.state(['!disabled'])


def onClick_Filters():
    filter_size = 10001
    flts, window_size = create_filters(audio_player.framerate, filter_size)
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    titles = frequences
    #plot_filter(flts, frequences, colors, filter_size)

    print("Filtros:\n", type(flts))
    array = np.array(flts)
    print("Salvando filtro...")
    np.savetxt("meus_filtros.csv", array, delimiter=",")
    np.savetxt(f"meus_filtros_{filter_size}.csv", array.T, delimiter=",")
    #my_graphs(sliders_freqs, flts)



"""
Functions
"""
def create_filters(framerate, filter_size=1001):
    for i in range(len(filters)):
        filters[i] = np.zeros(filter_size)
        lowcut = sliders_freqs[i][0]
        highcut = sliders_freqs[i][1]
        filters[i] += bandpass_filter(lowcut, highcut, framerate, filter_size)
    return filters, filter_size
        

def bandpass_filter_v0(lowcut, highcut, fs, numtaps, verbose=False):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    cutoff = [low, high]  # Normalized cutoff frequencies (e.g., 0.2 = 20% of Nyquist freq)
    # FIR filter design using a Hamming window
    b = firwin(numtaps, cutoff, window='hamming', pass_zero=False)
    return b


def bandpass_filter_v1(lowcut, highcut, fs, numtaps, verbose=False):
    nyquist = 1 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    taps = np.zeros(numtaps)
    for i in range(numtaps):
        n = i - (numtaps - 1) / 2
        if n == 0:
            taps[i] = 2 * (high - low)
        else:
            taps[i] = (np.sin(2 * np.pi * high * n) - np.sin(2 * np.pi * low * n)) / (np.pi * n)
    #taps *= hamming_window(numtaps)
    taps *= np.hamming(numtaps)
    if verbose:
        print('bandpass_filter(...):')
        print('    lowcut:', low)
        print('    highcut:', high)
        print('    framerate:', fs)
        print('    n. taps:', numtaps)
        print('    return(taps):', taps.shape)
    return taps


def bandpass_filter(f1, f2, fs, numtaps=101):
    left = f1 / (fs / 2)
    right = f2 / (fs / 2)

    # Build up the coefficients.
    alpha = 0.5 * (numtaps - 1)
    m = np.arange(0, numtaps) - alpha
    h = 0
    h += right * np.sinc(right * m)
    h -= left * np.sinc(left * m)

    # Get and apply the window function.
    win = np.hamming(numtaps)
    h *= win

    # Now handle scaling if desired.
    # Get the first passband.
    #left, right = bands[0]
    if left == 0:
        scale_frequency = 0.0
    elif right == 1:
        scale_frequency = 1.0
    else:
        scale_frequency = 0.5 * (left + right)
    c = np.cos(np.pi * m * scale_frequency)
    s = np.sum(h * c)
    h /= s

    return h


def hamming_window(M):
    if M <= 0:
        return np.array([])
    # Generate the window
    n = np.arange(M)
    return 0.54 - 0.46 * np.cos(2.0 * np.pi * n / (M - 1))


def apply_filter(filter_coef, filter_taps, verbose=False):
    convolved = np.convolve(filter_coef, filter_taps, mode='same')
    if verbose:
        print('apply_filter(...):')
        print('    filter_coef:', filter_coef.shape)
        print('    filter_taps:', filter_taps.shape)
        print('    return(convolved):', convolved.shape)
    return convolved


def amplify_attenuate_db(signal, db, verbose=False):
    linear_gain = round(10**(db / 20), 2)
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
    figure = Figure(figsize=(9, 1.9), dpi=100)
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
    figure = Figure(figsize=(4.5, 1.7), dpi=100)
    figure.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)  # Adjust margins

    # create axes
    axes = figure.add_subplot()
    axes.set_xlabel("Frequency [Hz]", fontsize=8)
    axes.set_ylabel("Magnitude (dB)", fontsize=8)

    # Set the font size of the tick labels
    axes.tick_params(axis='both', which='major', labelsize=8)

    if (audio_player):
        framerate = audio_player.framerate
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
    return frame


def spectrum_analyzer_frame(container, row):
    frame = ttk.Frame(container)

    # Create a style object
    style = ttk.Style()

    # Configure the style for the TFrame widget
    style.configure('White.TFrame', background='white')
    frm_bars = ttk.Frame(frame, style='White.TFrame')
    frm_bars['borderwidth'] = 5
    frm_bars['relief'] = 'sunken'

    progress_vars = []
    
    # grid layout for the input frame to place the progress bar widgets
    for i in range(len(frequences)):
        frame.columnconfigure(i, weight=1)
        frm_bars.columnconfigure(i, weight=1)
        progress_var = tk.DoubleVar()
        progress_vars.append(progress_var)
        spectrum_bar = ttk.Progressbar(frm_bars, orient='vertical', variable=progress_var, mode='determinate', length=150)
        spectrum_bar.grid(column=i, row=0, sticky=tk.N)
        ttk.Label(frm_bars, text=frequences[i], background='white', font=font.Font(size=8)).grid(column=i, row=1, sticky=tk.N, padx=2.5)

    ttk.Label(frame, text='Spectrum analyzer').grid(column=0, row=0, columnspan=3, sticky=tk.W)
    frm_bars.grid(column=0, row=1, columnspan=3, sticky=tk.EW, padx=2.5)

    # placing the DFT graph
    lbl_gph_dft = ttk.Label(frame, text='Discrete Fourier Transform (DTF) graph')
    frm_dft_graph = ttk.Frame(frame)
    frm_dft_graph['borderwidth'] = 5
    frm_dft_graph['relief'] = 'sunken'
    plot_dft(frm_dft_graph, None)
    lbl_gph_dft.grid(column=3, row=0, columnspan=7, sticky=tk.W, padx=2.5)
    frm_dft_graph.grid(column=3, row=1, columnspan=7, sticky=tk.EW, padx=2.5)

    frame.grid(column=0, row=row, columnspan=10, sticky=tk.EW, padx=10, pady=2.5)

    return progress_vars, frm_dft_graph


def add_frequence_labels(container, row, font_style):
    for i in range(len(frequences)):
        ttk.Label(container, text=frequences[i], font=font_style).grid(column=i, row=row, sticky=tk.N)


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
    btn_apply = ttk.Button(frm_right, text='Apply', state=['disabled'])
    btn_filters = ttk.Button(frm_right, text='Filters', state=['disabled'])
    btn_reset = ttk.Button(frm_right, text='Reset')

    # Place widgets
    frame.grid(column=0, row=row, columnspan=10, sticky=tk.EW, padx=10, pady=5)
    frm_left.grid(column=0, row=0, sticky=tk.W)
    btn_open.pack(side=tk.LEFT)
    lbl_file.pack(side=tk.LEFT, padx=5)
    frm_right.grid(column=1, row=0, sticky=tk.E)
    btn_apply.pack(side=tk.RIGHT)
    btn_filters.pack(side=tk.RIGHT)
    btn_reset.pack(side=tk.RIGHT)
    return lbl_file, btn_open, btn_filters, btn_reset, btn_apply


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

    frame.grid(column=0, row=row, columnspan=10, sticky=tk.EW, padx=10, pady=5)
    return frm_graph


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
    ttk.Label(root, text='+12').grid(column=i, row=1, sticky=tk.N)

# creating sliders widget
for i in range(len(sliders)):
    sliders[i] = ttk.Scale(root, from_=12, to=-12, orient='vertical', variable=tk.DoubleVar())
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
    ttk.Label(root, text='-12').grid(column=i, row=3, sticky=tk.N)

# Create labels to receive the sliders values
for i in range(len(slider_labels)):
    slider_labels[i] = ttk.Label(root, text='0.00 dB', font=font_style, width=8)
    slider_labels[i].grid(column=i, row=4, sticky=tk.N)

lbl_file, btn_open, btn_filters, btn_reset, btn_apply = create_equalizer_frame(root, row=5)

frm_graph = create_wave_frame(root, row=6)
progress_vars, frm_dft_graph = spectrum_analyzer_frame(root, row=7)

var_check = tk.IntVar()
frm_command, btn_play, btn_pause, btn_check, btn_exit, btn_save = create_command_frame(root, 8, var_check)

# Event configuration (command)
btn_open.configure(command=lambda: onClick_Open(frm_graph, frm_dft_graph, frm_command, btn_apply, btn_filters, lbl_file))
btn_filters.configure(command=onClick_Filters)
btn_reset.configure(command=onClick_Reset)
btn_apply.configure(command=lambda: onClick_Apply(frm_graph, frm_dft_graph, btn_save, audio_player))
btn_play.configure(command=lambda:play_audio(btn_play, btn_pause, progress_vars))
btn_pause.configure(command=lambda:pause_audio(btn_pause, btn_play))
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