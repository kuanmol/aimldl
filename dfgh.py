import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
from keras_preprocessing import image
import cv2  # OpenCV for camera capture

# Data preprocessing
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

training_set = train_datagen.flow_from_directory('dataset/training_set', target_size=(64, 64),
                                                 batch_size=32, class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

# Building the CNN
cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the CNN
cnn.fit(x=training_set, validation_data=test_set, epochs=25)

# Capturing image from the camera
cap = cv2.VideoCapture(0)  # Open the default camera
print("Press 's' to capture an image.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # Display the live video feed
    cv2.imshow('Camera', frame)

    # Wait for the user to press 's' to capture the image
    if cv2.waitKey(1) & 0xFF == ord('s'):
        captured_image = frame
        break

cap.release()
cv2.destroyAllWindows()

# Preprocessing the captured image
captured_image = cv2.resize(captured_image, (64, 64))  # Resize to match model input
captured_image = np.expand_dims(captured_image, axis=0)
captured_image = captured_image / 255.0  # Normalize

# Prediction
result = cnn.predict(captured_image)
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print("Prediction:", prediction)
