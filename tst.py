import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading

class Video360Viewer:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError("Errore nell'apertura del video.")
        self.theta = 0
        self.phi = 0
        self.fov = 70
        self.width_out = 1200
        self.height_out = 600
        self.setup_gui()

    def extract_frame_equirectangular(self, frame_number):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if not ret:
            print("Errore nella lettura del frame.")
            return None
        return frame

    def equirectangular_to_perspective(self, equirectangular):
        # Dummy transformation function. Replace with actual transformation logic.
        return equirectangular

    def update_image(self):
        frame = self.extract_frame_equirectangular(0)  # Example frame number
        if frame is not None:
            perspective_image = self.equirectangular_to_perspective(frame)
            img = cv2.cvtColor(perspective_image, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk

    def setup_gui(self):
        self.window = tk.Tk()
        self.window.title("Visualizzatore immagine prospettica")
        self.window.geometry('1250x800')
        self.image_label = ttk.Label(self.window)
        self.image_label.pack()
        # Add more GUI setup here if needed

    def run(self):
        self.update_image()  # Initial image update
        self.window.mainloop()

if __name__ == "__main__":
    video_path = 'video_1.MP4'
    viewer = Video360Viewer(video_path)
    viewer.run()
