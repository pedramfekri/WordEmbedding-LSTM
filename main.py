import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_datasets as tfds
import os
DIRECTORY_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
FILE_NAMES = ['cowper.txt', 'derby.txt', 'butler.txt']

for name in FILE_NAMES:
    text_dir = tf.keras.utils.get_file(name, origin=DIRECTORY_URL + name)

parent_dir = os.path.dirname(text_dir)

parent_dir

"""def embedding():
    embedding_layer = layers.Embedding(1000, 5)
"""

dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True,
                          as_supervised=True)