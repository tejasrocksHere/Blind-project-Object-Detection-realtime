import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import pyttsx3

# Initialize the text-to-speech engine
engine = pyttsx3.init()

class Application:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.cap = cv2.VideoCapture(0)  # For Webcam
        self.cap.set(3, 1280)
        self.cap.set(4, 720)

        self.model = YOLO("./yolov8l.pt")

        self.class_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                            "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                            "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                            "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                            "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                            "teddy bear", "hair drier", "toothbrush"
                            ]

        self.prev_frame_time = 0
        self.new_frame_time = 0

        # Create GUI components
        self.video_canvas = tk.Canvas(window, width=800, height=600)  # Increased width and height
        self.video_canvas.grid(row=0, column=0, padx=10, pady=10)

        self.log_text = scrolledtext.ScrolledText(window, width=40, height=10)
        self.log_text.grid(row=0, column=1, padx=10, pady=10)

        self.start_button = ttk.Button(window, text="Start", command=self.start_detection)
        self.start_button.grid(row=1, column=0, padx=10, pady=10)

        self.stop_button = ttk.Button(window, text="Stop", command=self.stop_detection)
        self.stop_button.grid(row=1, column=1, padx=10, pady=10)

        self.is_detecting = False
        self.detect()

    def detect(self):
        if self.is_detecting:
            success, img = self.cap.read()
            results = self.model(img, stream=True)

            detected_elements = []  # List to store detected elements in the current frame

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Confidence
                    conf = math.ceil((box.conf[0] * 100)) / 100

                    # Draw bounding box only when confidence level is above 0.8
                    if conf > 0.7:
                        # Bounding Box
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        w, h = x2 - x1, y2 - y1
                        cvzone.cornerRect(img, (x1, y1, w, h))

                        # Class Name
                        cls = int(box.cls[0])
                        class_name = self.class_names[cls]

                        # Add detected element to the list
                        detected_elements.append(f'{class_name} ')

                        cvzone.putTextRect(img, f'{class_name} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

            fps = 1 / (time.time() - self.prev_frame_time)
            self.prev_frame_time = time.time()

            # Display the frame with bounding boxes
            self.display_frame(img, fps, detected_elements)

            # Speak the detected elements
            if detected_elements:
                text_to_speak = ", ".join(detected_elements)
                engine.say(f'Detected: {text_to_speak}')
                engine.runAndWait()

            # Call detect function after 10 milliseconds
            self.window.after(10, self.detect)

    def display_frame(self, frame, fps, detected_elements):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(image=frame)

        self.video_canvas.config(width=photo.width(), height=photo.height())
        self.video_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.video_canvas.update()

        self.log_text.delete(1.0, tk.END)
        # Display detected elements in log_text
        if detected_elements:
            text_to_display = ", ".join(detected_elements)
            self.log_text.insert(tk.END, f'Detected Objects: {text_to_display}\n')
        self.log_text.insert(tk.END, f'FPS: {fps:.2f}\n')

    def start_detection(self):
        self.is_detecting = True
        self.detect()

    def stop_detection(self):
        self.is_detecting = False

# Create a Tkinter window
root = tk.Tk()
root.title("Object Detection")  # Set the title of the window
app = Application(root, "Object Detection App")
root.mainloop()

# Release the webcam and close the application
app.cap.release()
cv2.destroyAllWindows()