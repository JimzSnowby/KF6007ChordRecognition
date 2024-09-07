import tkinter as tk
from tkinter import Label, Toplevel, filedialog
from moviepy.editor import VideoFileClip
from gesture_recogniser import GestureRecognizer
from audio_classifier import AudioClassifier

Chords = ['Em', 'Dm', 'C', 'G', 'Am']

def countdown(count, label, callback):
    label.config(text=str(count))
    if count > 0:
        root.after(1000, countdown, count - 1, label, callback)
    else:
        callback()
        
# Get audio from video file
def extract_audio(video_path):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile('temp_audio.wav')
    return 'temp_audio.wav'
        
def open_file():
    file = filedialog.askopenfilename(filetypes=[('Video Files', '*.mp4')])
    if file:
        audio_path = extract_audio(file)
        audio_prediction, audio_confidence = audio_classifier.classify_audio_file(audio_path)
        
        video_app = GestureRecognizer(root, video_label, gesture_label, video_path=file)
        video_app.update_video()
        
        gesture_prediction, gesture_confidence = video_app.current_gesture, video_app.confidence
        
        combined_prediction, combined_confidence = combine_predictions(
            gesture_prediction, gesture_confidence, audio_prediction, audio_confidence
        )
        
        display_prediction(combined_prediction, combined_confidence)

def display_prediction(prediction, confidence):
    popup = Toplevel(root)
    popup.geometry("100x100")
    result_text = (
        f'{prediction}')
    colour = 'green' if confidence > 0.5 else 'red'
    Label(popup, text=result_text, font=large_font, fg=colour).pack()

def on_record():
    countdown(3, gui_label, validate_prediction)

def validate_prediction():
    audio = audio_classifier.record_audio()
    audio_prediction, audio_confidence = audio_classifier.predict_audio(audio)
    
    gesture = app.current_gesture
    gesture_confidence = app.confidence
    
    combined_prediction, combined_confidence = combine_predictions(
        gesture, gesture_confidence, audio_prediction, audio_confidence
    )
    
    popup = Toplevel(root)
    popup.geometry("")

    # Display the prediction results in the popup
    result_text = (
        f'{combined_prediction}')
    if combined_confidence < 0.5:
        Label(popup, text=result_text, font=large_font, fg='red').pack()
    else:
        Label(popup, text=result_text, font=large_font, fg='green').pack()
    
    # Reset gui_label text back to original after displaying results
    gui_label.config(text='Position fretting hand')

def combine_predictions(gesture, gesture_confidence, audio_prediction, audio_confidence):
    # Vision has more weight than audio
    gesture_weight = 0.6
    audio_weight = 0.4
    
    gesture_index = Chords.index(gesture) if gesture in Chords else -1
    
    # Combined score calculation
    combined_scores = [0] * len(Chords)
    if gesture_index != -1:
        combined_scores[gesture_index] += gesture_confidence * gesture_weight
    combined_scores[audio_prediction] += audio_confidence * audio_weight
    
    # Determine final prediction and confidence
    combined_prediction_index = combined_scores.index(max(combined_scores))
    combined_prediction = Chords[combined_prediction_index]
    combined_confidence = combined_scores[combined_prediction_index]
    
    return combined_prediction, combined_confidence

if __name__ == '__main__':
    # Create the main window
    root = tk.Tk()
    root.title('Chord Recognizer')
    root.geometry('600x820')
    
    audio_model = 'src/models/AudioClassification'
    audio_classifier = AudioClassifier(audio_model)
    
    font = ('Verdana', 32)
    large_font = ('Verdana', 70)
    
    gui_label = Label(root, text='Position fretting hand', font=font)
    gui_label.grid(column=0, row=1)
    
    video_label = Label(root)
    video_label.grid(column=0, row=0)
    
    gesture_label = Label(root, font=large_font, fg='green')
    gesture_label.grid(column=0, row=2)
    
    app = GestureRecognizer(root, video_label, gesture_label)
    
    record_button = tk.Button(root, text='Record', command=on_record, font=font)
    record_button.grid(column=0, row=3, pady=10)
    
    upload_button = tk.Button(root, text='Upload Video', command=open_file, font=font)
    upload_button.grid(column=0, row=4)
    
    root.mainloop()
