# Emotion and Speaker Recognition from Audio

This project is focused on detecting emotions and recognizing speakers from audio files using deep learning models. We have two pre-trained models for this purpose: one for emotion detection and another for speaker recognition.
The code for the models are in the files: Emotion_Detection_Ravdess_Dataset.ipynb and Speaker_Identification_Ravdess_Dataset.ipynb

## Prerequisites

Before running the code, ensure you have the following prerequisites installed:

- Python 3.x
- TensorFlow
- Keras

## Installation

1. Clone the repository to your local machine:
   ```shell
   git clone https://github.com/pranay-5374/VoiceSculptor.git
   ```

2. Navigate to the project directory:
   ```shell
   cd your-repo
   ```

3. Install the required dependencies:
   ```shell
   pip install tensorflow keras
   ```

4. Download the pre-trained models (if not already provided) and place them in the project directory.

## Usage

You can use the code to perform emotion detection and speaker recognition on audio files.

1. Import the necessary libraries:

   ```python
   from tensorflow import keras
   from tensorflow.keras import layers
   ```

2. Load the pre-trained models:

   ```python
   emotion_model = keras.models.load_model("emotion_detection_model.h5")
   speaker_model = keras.models.load_model("speaker_recognition_model.h5")
   ```

3. Freeze the layers of the loaded models to retain their pre-trained weights:

   ```python
   for layer in emotion_model.layers:
       layer.trainable = False

   for layer in speaker_model.layers:
       layer.trainable = False
   ```

4. Define an input layer for your audio data and connect it to both models:

   ```python
   input_layer = layers.Input(shape=(input_shape,))
   
   emotion_output = emotion_model(input_layer)
   speaker_output = speaker_model(input_layer)
   ```

5. Add additional layers for custom classification:

   ```python
   emotion_output = layers.Dense(128, activation='relu')(emotion_output)
   emotion_output = layers.Dense(num_emotion_classes, activation='softmax', name='emotion_output')(emotion_output)
   
   speaker_output = layers.Dense(128, activation='relu')(speaker_output)
   speaker_output = layers.Dense(num_speaker_classes, activation='softmax', name='speaker_output')(speaker_output)
   ```

6. Create a combined model that outputs both emotion and speaker predictions:

   ```python
   combined_model = keras.Model(inputs=input_layer, outputs=[emotion_output, speaker_output])
   ```

7. Compile the combined model with suitable loss functions, optimizers, and metrics:

   ```python
   combined_model.compile(optimizer='adam',
                         loss={'emotion_output': 'categorical_crossentropy', 'speaker_output': 'categorical_crossentropy'},
                         metrics={'emotion_output': 'accuracy', 'speaker_output': 'accuracy'})
   ```

8. Train and evaluate the combined model using your audio data.

For more details, you can check the code and its comments.

## Summary

The provided code allows you to integrate pre-trained emotion detection and speaker recognition models into a single model for audio analysis. Please make sure to replace `input_shape`, `num_emotion_classes`, and `num_speaker_classes` with appropriate values for your specific use case.

---
