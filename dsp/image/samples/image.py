import tkinter as tk
from tkinter import Frame, Label, Button
from PIL import Image, ImageTk

# Function to change the image
def change_image():
    new_image_path = "butterfly.jpg"  # Replace with the path to your new image
    new_image = Image.open(new_image_path)
    new_image = new_image.resize((500, 500), Image.LANCZOS)
    new_photo = ImageTk.PhotoImage(new_image)
    label.config(image=new_photo)
    label.image = new_photo  # Keep a reference to avoid garbage collection

# Create the main window
root = tk.Tk()
root.title("Image Display with Tkinter and Frame")

# Create a frame to hold the image
frame = Frame(root, width=500, height=500)
frame.grid(row=0, column=0, padx=10, pady=10)

# Load the initial image using PIL
initial_image_path = "girasol_abelha.jpg"  # Replace with the path to your initial image
initial_image = Image.open(initial_image_path)
initial_image = initial_image.resize((500, 500), Image.LANCZOS)
initial_photo = ImageTk.PhotoImage(initial_image)

# Create a label to display the image
label = Label(frame, image=initial_photo)
label.grid(row=0, column=0)
label.image = initial_photo  # Keep a reference to avoid garbage collection

# Create a button to change the image
change_image_button = Button(root, text="Change Image", command=change_image)
change_image_button.grid(row=1, column=0, pady=10)

# Start the Tkinter main loop
root.mainloop()
