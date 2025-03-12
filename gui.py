import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import ImageTk, Image
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('my_model.h5')

# Traffic sign classes
classes = {
    1: 'Speed limit (20km/h)',
    2: 'Speed limit (30km/h)',
    3: 'Speed limit (50km/h)',
    4: 'Speed limit (60km/h)',
    5: 'Speed limit (70km/h)',
    6: 'Speed limit (80km/h)',
    7: 'End of speed limit (80km/h)',
    8: 'Speed limit (100km/h)',
    9: 'Speed limit (120km/h)',
    10: 'No passing',
    11: 'No passing veh over 3.5 tons',
    12: 'Right-of-way at intersection',
    13: 'Priority road',
    14: 'Yield',
    15: 'Stop',
    16: 'No vehicles',
    17: 'Veh > 3.5 tons prohibited',
    18: 'No entry',
    19: 'General caution',
    20: 'Dangerous curve left',
    21: 'Dangerous curve right',
    22: 'Double curve',
    23: 'Bumpy road',
    24: 'Slippery road',
    25: 'Road narrows on the right',
    26: 'Road work',
    27: 'Traffic signals',
    28: 'Pedestrians',
    29: 'Children crossing',
    30: 'Bicycles crossing',
    31: 'Beware of ice/snow',
    32: 'Wild animals crossing',
    33: 'End speed + passing limits',
    34: 'Turn right ahead',
    35: 'Turn left ahead',
    36: 'Ahead only',
    37: 'Go straight or right',
    38: 'Go straight or left',
    39: 'Keep right',
    40: 'Keep left',
    41: 'Roundabout mandatory',
    42: 'End of no passing',
    43: 'End no passing veh > 3.5 tons'
}

# Create main window
top = tk.Tk()
top.geometry('900x700')
top.title('Traffic Sign Recognition System')
top.configure(background='#2A2A2A')

# Modern color scheme
BG_COLOR = '#2A2A2A'
ACCENT_COLOR = '#4CAF50'
TEXT_COLOR = '#FFFFFF'
BTN_COLOR = '#37474F'

# Custom font styles
TITLE_FONT = ('Helvetica', 24, 'bold')
LABEL_FONT = ('Helvetica', 14)
BTN_FONT = ('Helvetica', 12, 'bold')

# Create gradient background
canvas = tk.Canvas(top, width=900, height=700, bg=BG_COLOR, highlightthickness=0)
canvas.create_rectangle(0, 0, 900, 200, fill='#1B5E20', outline='')
canvas.create_rectangle(0, 200, 900, 700, fill=BG_COLOR, outline='')
canvas.pack()

# Header section
header_frame = tk.Frame(top, bg='#1B5E20')
header_frame.place(relx=0.5, rely=0.1, anchor='center')

title_label = tk.Label(header_frame,
                       text="Traffic Sign Recognition",
                       font=TITLE_FONT,
                       fg=TEXT_COLOR,
                       bg='#1B5E20')
title_label.pack(pady=10)

subtitle_label = tk.Label(header_frame,
                          text="Upload an image of a traffic sign to classify",
                          font=('Helvetica', 14),
                          fg=TEXT_COLOR,
                          bg='#1B5E20')
subtitle_label.pack()

# Image preview frame
preview_frame = tk.Frame(top, bg=BG_COLOR, bd=2, relief='groove')
preview_frame.place(relx=0.5, rely=0.45, anchor='center', width=400, height=300)

sign_image = tk.Label(preview_frame, bg=BG_COLOR)
sign_image.pack(pady=20)

# Result display
result_frame = tk.Frame(top, bg=BG_COLOR)
result_frame.place(relx=0.5, rely=0.75, anchor='center')

result_label = tk.Label(result_frame,
                        text="",
                        font=LABEL_FONT,
                        bg=BG_COLOR,
                        fg=ACCENT_COLOR,
                        width=40)
result_label.pack()

# Styled buttons
button_frame = tk.Frame(top, bg=BG_COLOR)
button_frame.place(relx=0.5, rely=0.85, anchor='center')

style = ttk.Style()
style.configure('TButton',
                font=BTN_FONT,
                padding=10,
                background=BTN_COLOR,
                foreground='black')

# Global variable to store the classify button
classify_btn = None

def classify(file_path):
    try:
        image = Image.open(file_path)
        image = image.resize((30, 30)).convert('RGB')
        img_array = np.array(image)
        img_array = np.expand_dims(img_array, axis=0)

        pred = np.argmax(model.predict(img_array))
        sign = classes[pred + 1]

        result_label.config(text=f"Detected Sign: {sign}")

    except Exception as e:
        result_label.config(text=f"Error: {str(e)}")


def show_classify_button(file_path):
    global classify_btn

    # Remove existing button if it exists
    if classify_btn:
        classify_btn.destroy()

    # Create new classify button
    classify_btn = ttk.Button(button_frame,
                              text="Analyze Image",
                              command=lambda: classify(file_path),
                              style='TButton')
    classify_btn.pack(side='left', padx=10)


def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        image = Image.open(file_path)
        image.thumbnail((380, 280))
        photo = ImageTk.PhotoImage(image)

        sign_image.config(image=photo)
        sign_image.image = photo
        result_label.config(text="")
        show_classify_button(file_path)

    except Exception as e:
        result_label.config(text=f"Error: {str(e)}")


upload_btn = ttk.Button(button_frame,
                        text="Upload Image",
                        command=upload_image,
                        style='TButton')
upload_btn.pack(side='left', padx=10)

# Decorative elements
canvas.create_line(50, 180, 850, 180, fill=ACCENT_COLOR, width=2)
canvas.create_oval(820, 50, 880, 110, outline=ACCENT_COLOR, width=2)

# Responsive layout
top.grid_rowconfigure(0, weight=1)
top.grid_columnconfigure(0, weight=1)

top.mainloop()
