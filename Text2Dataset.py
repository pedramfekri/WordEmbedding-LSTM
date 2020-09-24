import tensorflow as tf

import tensorflow_datasets as tfds
import os
import numpy as np
import pandas as pd

BUFFER_SIZE = 50000
BATCH_SIZE = 64
TAKE_SIZE = 500

counter = -1


def labeler(example, index):
    global counter
    counter += 1
    print("counter:", counter)
    return example, tf.cast(index[counter, :], tf.int64)


def encode(text_tensor, label):
    encoded_text = encoder.encode(text_tensor.numpy())
    return encoded_text, label


def encode_map_fn(text, label):
    # py_func doesn't set the shape of the returned tensors.
    encoded_text, label = tf.py_function(encode,
                                       inp=[text, label],
                                       Tout=(tf.int64, tf.int64))
    # `tf.data.Datasets` work best if all components have a shape set
    #  so set the shapes manually:
    encoded_text.set_shape([None])
    label.set_shape([])

    return encoded_text, label


filename = "dataset.txt"
dir = "C:\\Users\\vabedi\\PycharmProjects\\WordEmeddingLSTM\\"
csv = pd.read_csv("main_dataframe.csv")
txt = tf.data.TextLineDataset(os.path.join(dir, filename))
i = 0


l = csv.iloc[:, 1:]
l = l.values
# l = list(l)

labelled_data = []
for i in range(100):
    temp = txt.map(lambda ex: labeler(ex, l))
    labelled_data.append(temp)
# labelled_data = txt.map(lambda ex: labeler(ex, l))


for ex in labelled_data.take(100):
    print(ex)
'''
tokenizer = tfds.deprecated.text.Tokenizer()

vocabulary_set = set()
for text_tensor, _ in labelled_data:
  some_tokens = tokenizer.tokenize(text_tensor.numpy())
  vocabulary_set.update(some_tokens)

vocab_size = len(vocabulary_set)
print(vocab_size)


encoder = tfds.deprecated.text.TokenTextEncoder(vocabulary_set)

example_text = next(iter(labelled_data))[0].numpy()
print(example_text)

encoded_example = encoder.encode(example_text)
print(encoded_example)

encoded_data = labelled_data.map(encode_map_fn)
print(encoded_data)

train_data = encoded_data.skip(TAKE_SIZE).shuffle(BUFFER_SIZE)
train_data = train_data.padded_batch(BATCH_SIZE)

test_data = encoded_data.take(TAKE_SIZE)
test_data = test_data.padded_batch(BATCH_SIZE)

sample_text, sample_labels = next(iter(train_data))
'''