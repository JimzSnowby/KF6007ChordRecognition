import os
from mediapipe_model_maker import gesture_recognizer

IMAGES_PATH = 'src/dataset/Gestures'

# Load data and use folder names as labels
labels = [i for i in os.listdir(IMAGES_PATH) if os.path.isdir(os.path.join(IMAGES_PATH, i))]
print(labels)
data = gesture_recognizer.Dataset.from_folder(
    dirname=IMAGES_PATH,
    hparams=gesture_recognizer.HandDataPreprocessingParams()
)

# Split data
train_data, rest_data = data.split(0.8)
validation_data, test_data = rest_data.split(0.5)

# Model options and hyperparameters
model_options = gesture_recognizer.ModelOptions(
    dropout_rate=0.3,
    layer_widths=[256, 128] 
)

hparams = gesture_recognizer.HParams(
    export_dir="src/models/ChordVision",
    learning_rate=0.001,
    batch_size=32,
    epochs=20,
    shuffle=True,
    lr_decay=0.95,
    steps_per_epoch=100
)

options = gesture_recognizer.GestureRecognizerOptions(
    model_options=model_options,
    hparams=hparams
)

# Create and train the model
model = gesture_recognizer.GestureRecognizer.create(
    train_data=train_data,
    validation_data=validation_data,
    options=options
)

# Evaluate the model
loss, acc = model.evaluate(test_data, batch_size=1)
print(f"Test loss: {loss}, Test accuracy: {acc}")

# Save the model
model.export_model(model_name="chord_recognition_vision.task")

