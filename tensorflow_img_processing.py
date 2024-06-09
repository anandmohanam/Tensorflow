import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Paths to your dataset
train_dir = 'dataset/train'
validation_dir = 'dataset/validation'

# ImageDataGenerator for data augmentation and normalization
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Update class_mode to 'categorical' for multi-class classification
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(224, 224),
                                                    batch_size=32,
                                                    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                              target_size=(224, 224),
                                                              batch_size=32,
                                                              class_mode='categorical')

# Load pre-trained VGG16 model without the top layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze convolutional layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers on top of the convolutional base
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
# Update the number of neurons in the output layer to match the number of classes
num_classes = len(train_generator.class_indices)
predictions = Dense(num_classes, activation='softmax')(x)

# Create the full model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator,
          epochs=10,
          validation_data=validation_generator)

# Evaluate the model
loss, accuracy = model.evaluate(validation_generator)
print(f'Validation accuracy: {accuracy:.2f}')

# Function to process and predict images
def process_and_predict(image_path):
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        prediction = model.predict(img_array)
        class_indices = train_generator.class_indices
        predicted_class = list(class_indices.keys())[np.argmax(prediction)]
        return predicted_class
    except FileNotFoundError:
        print(f"Error: The file {image_path} does not exist.")
        return None

# Example usage
image_path = 'temp.jpg'  # Update this path to the correct image file
result = process_and_predict(image_path)
if result:
    print(f'The image is a {result}.')
