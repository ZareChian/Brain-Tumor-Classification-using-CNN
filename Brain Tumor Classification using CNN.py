# -*- coding: utf-8 -*-
"""
Brain Tumor Classification using CNN
Author: Mohammad Reza Zarechian
Description: CNN model for classifying brain tumors from MRI images
Dataset: Brain Tumor MRI Dataset (https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
"""

import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#-------------------------------------------------
##  Load full dataset into Python lists

# folder_path = "./tumor data"    #address the data is stored
folder_path = r"C:\Users\mreza\Downloads\tumor data"

mat_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.mat')]

images = []
labels = []

# Load each file
for file_path in mat_files:
    try:
        with h5py.File(file_path, 'r') as f:
            cjdata = f['cjdata']

            image = np.array(cjdata['image']).T
            label = int(np.array(cjdata['label'])[0][0])  # nested array

            images.append(image)
            labels.append(label)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
#--------------------------------------------------
## check dataset shape and type after loading
print("check dataset shape and type after loading")
print("Total images loaded:", len(images))
print("images type is:", type(images))
print("Images members shape :", set(images[i].shape for i in range(len(images))))
print("images members type : ", type(images[0]))

print("Total labels loaded:", len(labels))
print("labels type is:", type(labels))
print("Unique labels:", set(labels))
print("labels members type : ", type(labels[0]))

def get_tumor_type(label):
    """Map numeric labels to tumor type names"""
    tumor_types = {
        1: 'Meningioma',
        2: 'Glioma',
        3: 'Pituitary Tumor'
    }
    return tumor_types.get(label, 'Unknown')

unique_labels = set(labels)
example_indices = {label: labels.index(label) for label in unique_labels}


plt.figure(figsize=(15, 5))

for i, (label, idx) in enumerate(example_indices.items()):
    plt.subplot(1, len(unique_labels), i+1)
    plt.imshow(images[idx], cmap='gray')
    plt.title(f'Label: {label}\nTumor Type: {get_tumor_type(label)}')
    plt.axis('off')
    plt.colorbar(label='Intensity')

plt.tight_layout()
plt.show()
#---------------------------------------------------
## Resize all images to the same dimensions (256x256) and numpy arrays
target_size = (256, 256)  
images_resized = [resize(img, target_size, anti_aliasing=True) 
                  for img in images]

images_np = np.array(images_resized)
labels_np = np.array(labels) - 1  #labels change to (0,1,2) from (1,2,3)
#---------------------------------------------------
## check dataset shape after resizing

print("check dataset shape after resizing")
print("Array shape of resized images:", images_np.shape) # Should be (num_samples, 256, 256)
print("Array shape oflabels:" , labels_np.shape)
#----------------------------------------------------
## Standardize the dataset (images)
def standardize_masked(image, threshold=0.1):
    """Ignore near-black pixels when calculating mean/std."""
    mask = image > threshold * np.max(image)  # Exclude background
    mean = np.mean(image[mask])
    std = np.std(image[mask])
    standardized = np.zeros_like(image)
    standardized[mask] = (image[mask] - mean) / (std + 1e-7)
    return standardized

ReadyImages = np.array([standardize_masked(img) for img in images_np])

#the best types for working with CNN

ReadyImages = ReadyImages.astype('float32')
labels_np = np.array(labels_np).astype('int32')
#----------------------------------------------------
## check dataset shape after standardizing
plt.figure(figsize=(15, 5))
plt.suptitle("Images of Every Class After Resizing and Standardizing", fontsize=14, y=1.05) 
for i, (label, idx) in enumerate(example_indices.items()):
    plt.subplot(1, len(unique_labels), i+1)
    plt.imshow(ReadyImages[idx], cmap='gray')
    plt.title(f'Label: {label}\nTumor Type: {get_tumor_type(label)}')  # Subplot title
    plt.axis('off')
    plt.colorbar(label='Intensity')

plt.tight_layout()  # Prevent title overlap
plt.show()

print("check dataset shape after standardizing:")
print("Min:", np.min(ReadyImages))
print("Max:", np.max(ReadyImages))
print("Mean:", np.mean(ReadyImages))
print("Non-zero values:", np.count_nonzero(ReadyImages))
print("Array shape of standard images:", ReadyImages.shape) 
#----------------------------------------------------
## Add channel dimension for CNN input
ReadyImages = np.expand_dims(ReadyImages, axis=-1)  #Now shape is (3064, 128, 128, 1)
#----------------------------------------------------
## splitting the data for CNN
X_train, X_temp, y_train, y_temp = train_test_split(
    ReadyImages, labels_np, test_size=0.3, random_state=42)  
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42)
#----------------------------------------------------
#check dataset shape after spiliting:
print("Shapes of train, validation, and test data:")
print( "X_train:", X_train.shape, "X_val:" , X_val.shape, "X_test:", X_test.shape)
print( "y_train:", y_train.shape, "y_val:" , y_val.shape, "y_test:", y_test.shape)

#----------------------------------------------------
## creating a CNN model
X_train, y_train = shuffle(X_train, y_train, random_state=42)

model = models.Sequential([
    layers.Input(shape=(256, 256, 1)),
    
    # Convolutional layers
    # block 1
    layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    # block 2
    layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    # block 3
    layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    # Classifier head
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')  # 3 output classes
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#----------------------------------------------------
## Train the model
history = model.fit(X_train, y_train,
                    epochs=15,
                    batch_size=32,
                    validation_data=(X_val, y_val))
#----------------------------------------------------
## Evaluate on test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc:.4f}")
#----------------------------------------------------
## Generate predictions and classification report
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test, y_pred_classes)
print("confusion_matrix")
print(cm)

print(classification_report(y_test, y_pred_classes, 
      target_names=['Meningioma', 'Glioma', 'Pituitary Tumor']))


