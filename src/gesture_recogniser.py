import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageTk

class GestureRecognizer:
    def __init__(self, window, video_label, gesture_label, video_path=None):
        self.window = window
        self.video_label = video_label
        self.gesture_label = gesture_label
        model = 'src/models/ChordVision/chord_recognition_vision.task'
        base_options = python.BaseOptions(model_asset_path=model)
        options = vision.GestureRecognizerOptions(base_options=base_options)
        self.recognizer = vision.GestureRecognizer.create_from_options(options)
        
        # If a file is provided, read it
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
        else:
            self.cap = cv2.VideoCapture(0)
            
        self.current_gesture = None
        self.confidence = 0.0
        self.update_video()

    def update_video(self):
        success, frame = self.cap.read()
        if success:
            image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            # Detect hands and recognize gestures
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            recognition_result = self.recognizer.recognize(mp_image)
            recognized_gesture = recognition_result.gestures
            if recognized_gesture:
                for gesture in recognized_gesture[0]:
                    self.gesture_name = gesture.category_name
                    self.confidence = gesture.score
                    self.current_gesture = self.gesture_name
                    self.gesture_label.configure(text=self.gesture_name)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        self.video_label.after(10, self.update_video)
