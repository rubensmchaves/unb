import tkinter as tk
from tkinter import ttk

def start_progress():
    progress_var.set(0)
    root.after(200, update_progress)

def update_progress():
    current_value = progress_var.get()
    if current_value < 100:
        progress_var.set(current_value + 10)
        root.after(200, update_progress)
    else:
        progress_var.set(100)

# Create the main window
root = tk.Tk()
root.title("Progress Bar Example")
root.geometry("300x150")

# Create a variable to hold the progress bar's value
progress_var = tk.DoubleVar()

# Create the progress bar
progress_bar = ttk.Progressbar(root, variable=progress_var, maximum=100)
progress_bar.pack(pady=20)

# Create the "Start" button
start_button = tk.Button(root, text="Start", command=start_progress)
start_button.pack(pady=10)

# Run the main event loop
root.mainloop()
