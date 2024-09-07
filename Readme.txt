Set up a Python virtual environment (venv)

Using 'pip install', get the following packages
mediapipe
mediapipe-model-maker
tensorflow==2.15.0
tensorflow-io==0.36.0
opencv-python
matplotlib
numpy
pandas
sounddevice
scipy
moviepy
librosa

To make a new dataset for new chords use datasetmaker.py. This will take pictures and crop them to your fretting hand.
Make sure you move SLOWLY to avoid blurry images.

Use train_audio.py and train_gestures.py to train new models

Run main.py to launch the program. Pressing 'Record' will take a 3 second clip, play the chord after the countdown for a prediction.
mp4 files can be loaded with the 'Upload Video' button to detect the chord in the video (guitarvid.mp4 is a provided example).
NOTE: video files must be mp4 and 3 seconds in length.