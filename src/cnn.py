import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt
import os
tf.__version__

# Récupération du chemin du répertoire courant
base_dir = os.getcwd()

tf.keras.utils.get_file(
    origin='https://full-stack-bigdata-datasets.s3.eu-west-3.amazonaws.com/Deep+learning+Images+processing/Transfer+Learning/101_ObjectCategories.tar.gz',
    fname='101_ObjectCategories.tar.gz', 
    untar=True,
    cache_dir=base_dir,
    cache_subdir="content"
)

# Appliquer les augmentations d'images
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomBrightness(factor=(0.0, 0.2)),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomFlip('horizontal_and_vertical'),
    # tf.keras.layers.ChannelShuffle(0.2)
])

# Créer un dataset d'images à partir d'un répertoire
train_dataset = tf.keras.utils.image_dataset_from_directory(
    './content/101_ObjectCategories',
    validation_split=0.3,
    subset='training',  # ou 'validation' pour le dataset de validation
    seed=42,
    image_size=(224, 224),
    batch_size=32
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    './content/101_ObjectCategories',
    validation_split=0.3,
    subset='validation',  # ou 'validation' pour le dataset de validation
    seed=42,
    image_size=(224, 224),
    batch_size=32
)

# Récupérer les noms des classes (les noms des répertoires)
class_names = train_dataset.class_names
print(class_names)

# Récupérer les chemins des fichiers avant d'appliquer la fonction map
val_file_paths = val_dataset.file_paths

def preprocess_image(images, labels):
    # Convertir le label en nom de classe
    images = data_augmentation(images)
    images = tf.cast(images, tf.float32) / 255.0
    return images, labels
    
# Appliquer les augmentations à l'ensemble de données


train_dataset = train_dataset.map(preprocess_image)
val_dataset = val_dataset.map(preprocess_image)

# Visualiser le premier batch
for images, labels in train_dataset.take(1):
    labels_as_string = tf.gather(class_names, labels)
    # Afficher les images du batch
    for img, label in zip(images, labels_as_string):
        # Afficher la première image du batch
        plt.imshow(img.numpy())
        plt.title(label.numpy().decode('utf-8'))
        plt.show()