import pandas as pd
import numpy as np
import tensorflow as tf
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0],True)
import tensorflow_datasets as tfds
import os

BUFFER_SIZE = 50000
BATCH_SIZE = 64
TAKE_SIZE = 500


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
    label.set_shape([20])

    return encoded_text, label


label = pd.read_csv('df_20labels.csv')
text = label.pop('text')

labelled_data = tf.data.Dataset.from_tensor_slices((text, label))

#for txt, cls in labelled_data.take(100):
#    print(txt, cls)

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

label = next(iter(labelled_data))[1].numpy()
print(label)

encoded_data = labelled_data.map(encode_map_fn)
print(encoded_data)

train_data = encoded_data.skip(TAKE_SIZE).shuffle(BUFFER_SIZE)
train_data = train_data.padded_batch(BATCH_SIZE)

test_data = encoded_data.take(TAKE_SIZE)
test_data = test_data.padded_batch(BATCH_SIZE)

sample_text, sample_labels = next(iter(train_data))

print(sample_text[0], sample_labels[0])

vocab_size += 1

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 64),
    # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    # tf.keras.layers.Dense(128, activation='relu'),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(20, activation='sigmoid')
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

# save the model
checkpoint_path = "/home/pedram/modelnlp/"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,


                                                 save_weights_only=True,
                                                 verbose=1)

# Train the model with the new callback
history = model.fit(train_data,
                    validation_data=test_data,
                    validation_steps=30,
                    epochs=200,
                    callbacks=[cp_callback],
                    )

entire_model_path = "/home/pedram/modelnlp/"
model.save(entire_model_path)

"""
embedded = tf.keras.layers.Embedding(vocab_size, 64)
lstm_1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False))(embedded)
# lstm_2 = tf.keras.layers.LSTM(64, return_sequences=False)(lstm_1)
fc_1 = tf.keras.layers.Dense(19, activation='relu')(lstm_1)
fc_2 = tf.keras.layers.BatchNormalization()(fc_1)
fc_3 = tf.keras.layers.Dropout(0.2)(fc_2)

fc_4 = tf.keras.layers.Dense(64, activation='relu')(fc_3)
fc_5 = tf.keras.layers.BatchNormalization()(fc_4)
fc_6 = tf.keras.layers.Dropout(0.2)(fc_5)
OUTPUTS = tf.keras.layers.Dense(20, activation='sigmoid')(fc_6)


model = tf.keras.Model(train_data)

model.summary()


model.compile(loss=tf.keras.losses.binary_crossentropy,
                      optimizer='adam',
                      metrics=['accuracy'])

history = model.fit(train_data,
                    epochs=50)"""