import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write, read
from keras.models import load_model
import tensorflow as tf

class AudioClassifier:
    def __init__(self, model_path):
        self.model = self.load_audio_model(model_path)
        self.labels = ['Em', 'Dm', 'C', 'G', 'Am']
        
    def load_audio_model(self, model_path):
        return load_model(model_path)
    
    def get_spectrogram(self, waveform):
        input_len = 132300 # 3 seconds of audio
        waveform = waveform[:input_len]
        zero_padding = tf.zeros([input_len] - tf.shape(waveform), dtype=tf.float32)
        # Cast the waveform to a float tensor.
        waveform = tf.cast(waveform, dtype=tf.float32)
        # Ensure clips are of the same length.
        equal_length = tf.concat([waveform, zero_padding], 0)
        
        # Convert the waveform to a spectrogram via a STFT.
        spectrogram = tf.signal.stft(
            equal_length, frame_length=1024, frame_step=256)
        # Obtain the magnitude of the STFT.
        spectrogram = tf.abs(spectrogram)
        spectrogram = tf.expand_dims(spectrogram, axis=2)
        
        return spectrogram
    
    def record_audio(self, duration=3, sample_rate=44100):
        print("Recording...")
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()
        print("Recording done.")
        
        return np.squeeze(recording)  # Force to mono audio

    def preprocess_audio(self, audio, sample_rate=44100):
        if len(audio) < sample_rate * 3:  # 3 seconds
            padding = np.zeros((sample_rate * 3 - len(audio),))
            audio = np.concatenate((audio, padding))
            
        return audio

    def predict_audio(self, audio):
        audio = self.preprocess_audio(audio)
        spectrogram = self.get_spectrogram(audio)
        spectrogram = np.expand_dims(spectrogram, axis=0)
        prediction = self.model.predict(spectrogram)
        confidence = np.max(prediction)
        
        return np.argmax(prediction), confidence
    
    def read_audio_file(self, audio_path):
        sample_rate, audio = read(audio_path)
        if len(audio.shape) > 1:
            audio = audio[:, 0]
            
        return self.preprocess_audio(audio, sample_rate)
    
    def classify_audio_file(self, audio_path):
        audio = self.read_audio_file(audio_path)
        
        return self.predict_audio(audio)
