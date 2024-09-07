import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from keras import layers, models, regularizers

DATASET_PATH = 'src/dataset/Audio'
data_dir = pathlib.Path(DATASET_PATH)
training_dir = pathlib.Path(f'{DATASET_PATH}/Train')
test_dir = pathlib.Path(f'{DATASET_PATH}/Test')

# Get labels
chords = np.array(tf.io.gfile.listdir(str(training_dir)))
print('Chords:', chords)

filenames = tf.io.gfile.glob(str(training_dir) + '/*/*')
filenames = tf.random.shuffle(filenames, seed=42)
num_samples = len(filenames)
print('Number of training examples:', num_samples)
print('Number of examples per label:',
        len(tf.io.gfile.listdir(str(training_dir/chords[0]))))
print('Example file tensor:', filenames[0])

# Split the files
train_files = filenames[:1152]
val_files = filenames[-288:]
test_files = tf.io.gfile.glob(str(test_dir) + '/*/*')

print('Training set size', len(train_files))
print('Validation set size', len(val_files))
print('Test set size', len(test_files))

# Get the shape of the audio file for the model input
test_file = tf.io.read_file(test_files[0])
test_audio, _ = tf.audio.decode_wav(contents=test_file, desired_channels=1)
print(test_audio.shape)



def decode_audio(audio_binary):
    # Decode WAV files to `float32` tensors
    audio, _ = tf.audio.decode_wav(contents=audio_binary, desired_channels=1)
    audio = tf.squeeze(audio, axis=-1)
    
    return audio

def get_label(file_path):
    parts = tf.strings.split(
        input=file_path,
        sep=os.path.sep)
    
    return parts[-2]

def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    
    return waveform, label

def augment_audio(waveform, label):
    # Add noise to augment the dataset
    noise = tf.random.normal(tf.shape(waveform), stddev=0.05)
    waveform += noise
    
    return waveform, label

# Tensorflow data pipeline for handling data
AUTOTUNE = tf.data.AUTOTUNE
files_ds = tf.data.Dataset.from_tensor_slices(train_files)
waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
augmented_waveform_ds = waveform_ds.map(augment_audio, num_parallel_calls=AUTOTUNE)

def get_spectrogram(waveform):
    input_len = 132300 # 3 seconds of audio
    waveform = waveform[:input_len]
    zero_padding = tf.zeros([input_len] - tf.shape(waveform), dtype=tf.float32)

    # Convert the waveform to a spectrogram via a STFT
    waveform = tf.cast(waveform, dtype=tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(equal_length, frame_length=1024, frame_step=256)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    
    return spectrogram

for waveform, label in waveform_ds.take(1):
    label = label.numpy().decode('utf-8')
    spectrogram = get_spectrogram(waveform)

def get_spectrogram_and_label_id(audio, label):
    spectrogram = get_spectrogram(audio)
    label_id = tf.argmax(label == chords)
    
    return spectrogram, label_id

spectrogram_ds = waveform_ds.map(
    map_func=get_spectrogram_and_label_id,
    num_parallel_calls=AUTOTUNE)

augmented_spectrogram_ds = augmented_waveform_ds.map(
    map_func=get_spectrogram_and_label_id,
    num_parallel_calls=AUTOTUNE
)

def preprocess_dataset(files):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(
        map_func=get_waveform_and_label,
        num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.map(
        map_func=get_spectrogram_and_label_id,
        num_parallel_calls=AUTOTUNE)
    
    return output_ds


train_ds = spectrogram_ds
val_ds = preprocess_dataset(val_files)
test_ds = preprocess_dataset(test_files)

batch_size = 32
train_ds = train_ds.batch(batch_size)
val_ds = val_ds.batch(batch_size)

train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

for spectrogram, _ in spectrogram_ds.take(2):
    input_shape = spectrogram.shape
    
print('Input shape:', input_shape)
num_labels = len(chords)

train_ds = augmented_spectrogram_ds.batch(batch_size).cache().prefetch(AUTOTUNE)

# Create the model
model = models.Sequential([
    layers.Input(shape=input_shape),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.6),
    layers.Dense(num_labels, activation='softmax'),
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy'],
)

EPOCHS = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
)

# Plot the training and validation loss to a graph
metrics = history.history
plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.show()


# Evaluate the model and test its accuracy
test_audio = []
test_labels = []

for audio, label in test_ds:
    test_audio.append(audio.numpy())
    test_labels.append(label.numpy())

test_audio = np.array(test_audio)
test_labels = np.array(test_labels)
y_pred = np.argmax(model.predict(test_audio), axis=1)
y_true = test_labels

test_acc = sum(y_pred == y_true) / len(y_true)
print(f'Test set accuracy: {test_acc:.0%}')

sample_file = test_files[140]
sample_ds = preprocess_dataset([str(sample_file)])

for spectrogram, label in sample_ds.batch(1):
    prediction = model(spectrogram)
    plt.bar(chords, tf.nn.softmax(prediction[0]))
    plt.title(f'Predictions for "{chords[label[0]]}"')
    plt.show()
    

model.save('src/models/AudioClassification')