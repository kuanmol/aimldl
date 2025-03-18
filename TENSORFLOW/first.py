import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from keras.api.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras.api.models import Sequential

tf.random.set_seed(42)
train_datagen = ImageDataGenerator(rescale=1. / 255)
valid_datagen = ImageDataGenerator(rescale=1. / 255)

train_dir = "/Artificial/TENSORFLOW/pizza_steak/train/"
test_dir = "/Artificial/TENSORFLOW/pizza_steak/test/"

#import data frm directories and turn it into batches
train_data = train_datagen.flow_from_directory(train_dir,
                                               batch_size=32,
                                               target_size=(224, 224),
                                               class_mode="binary",
                                               seed=42)
valid_data = valid_datagen.flow_from_directory(test_dir,
                                               batch_size=32,
                                               target_size=(224, 224),
                                               class_mode="binary",
                                               seed=42)

model1 = Sequential([
    Conv2D(filters=10, kernel_size=3, activation='relu', input_shape=(224, 224, 3)),
    Conv2D(filters=10, kernel_size=3, activation='relu'),
    MaxPooling2D(pool_size=2, padding='valid'),
    Conv2D(filters=10, kernel_size=3, activation='relu'),
    Conv2D(filters=10, kernel_size=3, activation='relu'),
    MaxPooling2D(pool_size=2),
    Flatten(),
    Dense(1, activation='sigmoid'),
])

model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
his1 = model1.fit(train_data, epochs=5, steps_per_epoch=len(train_data),
                  validation_data=valid_data, validation_steps=len(valid_data))