"""
Application to apply filters on an image.

Libraries (> py -m pip install [library_name])
- pillow
"""
import tkinter as tk

from tkinter import ttk # newer themed widgets
from tkinter import filedialog
from PIL import Image, ImageTk

"""
Callback functions
"""
def open_file(path_label, image_label):
	image_path = filedialog.askopenfilename(filetypes=[("Image file", "*.jpg; *png")])
	if image_path:
		path_label.configure(text=image_path)

		# Load the image using PIL
		image = Image.open(image_path)

		# Resize the image to fit the frame
		image = image.resize((400, 300), Image.LANCZOS)

		# Convert the image to a PhotoImage object
		new_photo = ImageTk.PhotoImage(image)

		# Change image of the label
		image_label.config(image=new_photo)
		image_label.image = new_photo  # Keep a reference to avoid garbage collection


"""
Frames
"""
def create_frame_file_select(container):
	frame = ttk.Frame(container)

    # grid layout for the input frame
	frame.columnconfigure(0, weight=5)
	frame.columnconfigure(1, weight=1)

	path_label  = ttk.Label(frame, text='Image source file...')
	open_button = ttk.Button(frame, text='Open file')

	path_label.grid(column=0, row=0, padx=5, sticky=tk.W)
	open_button.grid(column=1, row=0, padx=5, sticky=tk.E)

	return frame, path_label, open_button


def create_frame_images(container):
	frame = ttk.Frame(container)
	# grid layout for the input frame
	frame.columnconfigure(0, weight=1)
	frame.columnconfigure(1, weight=1)

	ttk.Label(frame, text='Original image').grid(column=0, row=0, padx=5, pady=5, sticky=tk.N)
	ttk.Label(frame, text='Filtered image').grid(column=1, row=0, padx=5, pady=5, sticky=tk.N)

	# Load the image using PIL
	imageA_path = "blank_source.jpg"  # Replace with the path to your image
	imageA = Image.open(imageA_path)
	imageA = imageA.resize((400, 300), Image.LANCZOS)
	photoA = ImageTk.PhotoImage(imageA)

	# Create a frame to hold the image
	frameA = ttk.Frame(frame, width=400, height=300)
	frameA['borderwidth'] = 3
	frameA['relief'] = 'sunken'
	image_label = ttk.Label(frameA, image=photoA)
	image_label.image = photoA
	image_label.pack()
	frameA.grid(column=0, row=1, padx=5, sticky=tk.N)

	# Load the image using PIL
	imageB_path = "blank.jpg"  # Replace with the path to your image
	imageB = Image.open(imageB_path)
	imageB = imageB.resize((400, 300), Image.LANCZOS)
	photoB = ImageTk.PhotoImage(imageB)

	frameB = ttk.Frame(frame, width=400, height=300)
	frameB['borderwidth'] = 3
	frameB['relief'] = 'sunken'
	image_filtered_label = ttk.Label(frameB, image=photoB)
	image_filtered_label.image = photoB
	image_filtered_label.pack()
	frameB.grid(column=1, row=1, padx=5, sticky=tk.N)

	return frame, image_label, image_filtered_label 


def create_frame_filters(container):
	frame = ttk.Frame(container)
	# grid layout for the input frame
	frame.columnconfigure(0, weight=3)
	frame.columnconfigure(1, weight=1)
	frame.columnconfigure(2, weight=3)

	ttk.Label(frame, text='Available filters').grid(column=0, row=0, padx=5, pady=5, sticky=tk.W)
	ttk.Label(frame, text='Selected filters').grid(column=2, row=0, padx=5, pady=5, sticky=tk.W)

	# create a list box
	filters = ('Filter 1', 'Filter 2', 'Filter 3', 'Filter 4', 'Filter 5', 'Filter 6', 'Filter 7', 
		'Filter 8', 'Filter 9', 'Filter 10', 'Filter 11', 'Filter 12', 'Filter 13', 'Filter 14')

	varA = tk.Variable(value=filters)
	listboxA = tk.Listbox(frame, listvariable=varA, height=8, selectmode=tk.MULTIPLE)
	listboxA.grid(column=0, row=1, rowspan=4, sticky=tk.EW, padx=5)

	add_button = ttk.Button(frame, text='>')
	rem_buttom = ttk.Button(frame, text='<')
	add_all_button = ttk.Button(frame, text='>>')
	rem_all_button = ttk.Button(frame, text='<<')
	add_button.grid(column=1, row=1, sticky=tk.N)
	rem_buttom.grid(column=1, row=2, sticky=tk.N)
	add_all_button.grid(column=1, row=3, sticky=tk.N)
	rem_all_button.grid(column=1, row=4, sticky=tk.N)

	varB = tk.Variable(value=None)
	listboxB = tk.Listbox(frame, listvariable=varB, height=8, selectmode=tk.MULTIPLE)
	listboxB.grid(column=2, row=1, rowspan=4, sticky=tk.EW, padx=5)
	return frame


def create_frame_apply(container):
	frame_button = ttk.Frame(container)
	apply_button = ttk.Button(frame_button, text='Apply')
	apply_all_button = ttk.Button(frame_button, text='Apply all')
	apply_button.pack(side=tk.LEFT)
	apply_all_button.pack(side=tk.LEFT)
	return frame_button, apply_button, apply_all_button


"""
Main window
"""
root = tk.Tk()

# window attributes
root.title("Image filters")
root.geometry("860x600+50+50")
root.resizable(False, False)
root.iconbitmap("logo_unb.ico")

# configure the grid
root.columnconfigure(0, weight=4)
root.columnconfigure(1, weight=1)
root.columnconfigure(2, weight=4)

frame_file_select, path_label, open_button = create_frame_file_select(root)
frame_file_select.grid(column=0, row=0, columnspan=3, sticky=(tk.EW), padx=10, pady=5)

frame_images, source_image, filtered_image = create_frame_images(root)
frame_images.grid(column=0, row=1, columnspan=3, padx=5, pady=5, sticky=tk.N)

open_button.configure(command=lambda:open_file(path_label, source_image))

frame_filters = create_frame_filters(root)
frame_filters.grid(column=0, row=2, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=5)

frame_button, apply_button, apply_all_button = create_frame_apply(root)
frame_button.grid(column=2, row=3, padx=10, pady=5, sticky=tk.E)

# code to run across multi platforms
try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1) # it runs on Windows, but not on macOS or Linux
finally:
    # keep the window displaying
    root.mainloop()

