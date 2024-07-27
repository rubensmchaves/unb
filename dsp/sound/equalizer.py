"""
Tutorial de uso do Tkinter:    https://www.pythontutorial.net/tkinter/
"""

import tkinter as tk
from tkinter import ttk # newer themed widgets

root = tk.Tk()

# window attributes
root.title("Equalizer")
root.geometry("600x400+50+50")

# place a label on the root window
ttk.Label(root, text='Themed Label').pack()

# code to run across multi platforms
try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1) # it runs on Windows, but not on macOS or Linux
finally:
    # keep the window displaying
    root.mainloop()