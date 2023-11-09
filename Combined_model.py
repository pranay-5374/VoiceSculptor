from tensorflow import keras
from tensorflow.keras import layers

emotion_model = keras.models.load_model("emotion_detection_model.h5")
speaker_model = keras.models.load_model("speaker_recognition_model.h5")

for layer in emotion_model.layers:
    layer.trainable = False

for layer in speaker_model.layers:
    layer.trainable = False


input_layer = layers.Input(shape=(input_shape,))

emotion_output = emotion_model(input_layer)
speaker_output = speaker_model(input_layer)


emotion_output = layers.Dense(128, activation='relu')(emotion_output)
emotion_output = layers.Dense(num_emotion_classes, activation='softmax', name='emotion_output')(emotion_output)

speaker_output = layers.Dense(128, activation='relu')(speaker_output)
speaker_output = layers.Dense(num_speaker_classes, activation='softmax', name='speaker_output')(speaker_output)

combined_model = keras.Model(inputs=input_layer, outputs=[emotion_output, speaker_output])


combined_model.compile(optimizer='adam',
                      loss={'emotion_output': 'categorical_crossentropy', 'speaker_output': 'categorical_crossentropy'},
                      metrics={'emotion_output': 'accuracy', 'speaker_output': 'accuracy'})


combined_model.summary()