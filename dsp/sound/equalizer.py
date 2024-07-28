"""
Tutorial de uso do Tkinter:    https://www.pythontutorial.net/tkinter/
"""

import tkinter as tk

from tkinter import ttk # newer themed widgets
from tkinter import font
from tkinter import filedialog

import matplotlib.pyplot as plt
#from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
import wave
import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


"""
Callback functions
"""
def open_file(frame):
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if file_path:
        l_file_name.configure(text=file_path)
        for widget in frame.winfo_children():
            widget.destroy()
        plot_waveform(frame, file_path)

def agreement_changed():
    print("Checkbox: ", checked.get())

def button_clicked():
    print('Button clicked')

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


"""
Functions
"""
def formated_value(value):
    return '{: .2f}'.format(value)


def plot_waveform(frame, file_path):
    matplotlib.use('TkAgg')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # create a figure
    figure = Figure(figsize=(6, 2), dpi=100)

    # create axes
    axes = figure.add_subplot()

    print('Path:', file_path)
    if file_path:
        print('Abrindo arquivo WAV...')
        with wave.open(file_path, 'rb') as wav_file:
            # Extract audio parameters
            n_channels = wav_file.getnchannels()
            sampwidth = wav_file.getsampwidth()
            framerate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            
            # Read the audio data
            audio_data = wav_file.readframes(n_frames)
        
        print('Deixando arquivo...')
        # Convert the audio data to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # If stereo, take only one channel for simplicity
        if n_channels == 2:
            audio_array = audio_array[::2]

        # Create time array for x-axis
        time_array = np.linspace(0, n_frames/framerate, num=n_frames)
        

        # create the barchart
        axes.plot(time_array, audio_array)
        
    else:
        # create the barchart
        axes.axhline(y=10, color='blue', linestyle='-')
        #axes.plot([0, 100], [-3, 3])

    # Remove axis labels
    axes.set_xticks([])
    axes.set_yticks([])

    figure.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # create FigureCanvasTkAgg object
    figure_canvas = FigureCanvasTkAgg(figure, frame)
    figure_canvas.draw()
    figure_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    

def create_command_frame(container, checked_variable):

    frame = ttk.Frame(container)

    # grid layout for the input frame
    frame.columnconfigure(0, weight=1)
    frame.columnconfigure(1, weight=1)
    frame.columnconfigure(2, weight=1)

    ttk.Button(frame, text='Play', command=button_clicked)
    ttk.Button(frame, text='Pause', command=button_clicked)
    ttk.Checkbutton(frame, text='Repeat', command=agreement_changed, variable=checked_variable)

    i = 0
    for widget in frame.winfo_children():
        widget.grid(column=i, row=0, padx=10, pady=5)
        i += 1

    return frame


"""
 Equalizer code
"""
root = tk.Tk()

# window attributes
root.title("Equalizer")
root.geometry("700x480+50+50")
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

# place a label on the root window
l_file_name = ttk.Label(root, text='WAVE file...')

# Adding to the grid
l_file_name.grid(column=0, row=0, columnspan=7, sticky=tk.W, padx=10, pady=5)
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

frame_graph = ttk.Frame(root)
frame_graph['borderwidth'] = 5
frame_graph['relief'] = 'sunken'
plot_waveform(frame_graph, None)
frame_graph.grid(column=0, row=5, columnspan=10, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10)

b_select_file = ttk.Button(root, text='Open file', command=lambda: open_file(frame_graph))
b_select_file.grid(column=7, row=0, columnspan=3, sticky=tk.E, padx=10, pady=5)

value_slider = tk.DoubleVar()
slider = ttk.Scale(root, from_=0, to=100, variable=value_slider) # add param: command=[function]
slider.grid(column=0, row=6, columnspan=10, sticky=tk.EW, padx=40)

checked = tk.IntVar()
button_frame = create_command_frame(root, checked)
button_frame.grid(column=0, row=7, columnspan=4, sticky=tk.W)
b_exit  = ttk.Button(root, text='Exit', command=root.destroy).grid(column=8, row=7, columnspan=2, sticky=tk.E, padx=10)



# code to run across multi platforms
try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1) # it runs on Windows, but not on macOS or Linux
finally:
    # keep the window displaying
    root.mainloop()