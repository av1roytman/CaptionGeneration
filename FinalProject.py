import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import json

# Parameters
batch_size = 32
img_height = 224
img_width = 224
autotune = tf.data.AUTOTUNE
embedding_dim = 256
lstm_units = 512
num_words = 5000

# Paths
dataset_path = './archive/'
image_dir = os.path.join(dataset_path, 'Images/')
caption_file = 'captions.txt'

# Load Data
df = pd.read_csv(os.path.join(dataset_path, caption_file))
captions = df['caption'].values[0:5000]
image_files = df['image'].values[0:5000]

# Split Data
train_images, val_images, train_captions, val_captions = train_test_split(image_files, captions, test_size=0.2, random_state=42)

# Tokenizer
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=num_words, oov_token='<OOV>')
tokenizer.fit_on_texts(train_captions)

max_caption_length = 30  # adjust as needed

train_captions = tokenizer.texts_to_sequences(train_captions)
train_captions = tf.keras.preprocessing.sequence.pad_sequences(train_captions, maxlen=max_caption_length, padding='post')
val_captions = tokenizer.texts_to_sequences(val_captions)
val_captions = tf.keras.preprocessing.sequence.pad_sequences(val_captions, maxlen=max_caption_length, padding='post')

# Data Generator
def data_generator(image_files, captions):
    for img_file, cap in zip(image_files, captions):
        img = tf.keras.utils.load_img(os.path.join(image_dir, img_file), target_size=(img_height, img_width))
        img = tf.keras.utils.img_to_array(img)
        img = img / 255.0
        input_seq = cap[:-1]
        target_seq = cap[1:]
        yield (img, input_seq), target_seq

# tf.data Dataset
train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(train_images, train_captions),
    output_signature=(
        (tf.TensorSpec(shape=(img_height, img_width, 3), dtype=tf.float32),
         tf.TensorSpec(shape=(max_caption_length-1,), dtype=tf.int32)),
        tf.TensorSpec(shape=(max_caption_length-1,), dtype=tf.int32)
    )
)
train_dataset = train_dataset.batch(batch_size).prefetch(autotune).repeat()

val_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(val_images, val_captions),
    output_signature=(
        (tf.TensorSpec(shape=(img_height, img_width, 3), dtype=tf.float32),
         tf.TensorSpec(shape=(max_caption_length-1,), dtype=tf.int32)),
        tf.TensorSpec(shape=(max_caption_length-1,), dtype=tf.int32)
    )
)
val_dataset = val_dataset.batch(batch_size).prefetch(autotune).repeat()

# Model
encoder = tf.keras.applications.VGG16(include_top=False, weights='imagenet', pooling='avg', input_shape=(img_height, img_width, 3))
encoder.trainable = False

class LSTMDecoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, lstm_units):
        super(LSTMDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)
        self.lstm = tf.keras.layers.LSTM(lstm_units, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, input_sequence, initial_state):
        embedded_sequence = self.embedding(input_sequence)
        lstm_output, _, _ = self.lstm(embedded_sequence, initial_state=initial_state)
        logits = self.dense(lstm_output)
        return logits

vocab_size = len(tokenizer.word_index) + 1
decoder = LSTMDecoder(vocab_size, embedding_dim, lstm_units)

class ImageCaptioningModel(tf.keras.Model):
    def __init__(self, encoder, decoder):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs, training=False):
        images, captions = inputs
        encoded_images = self.encoder(images)
        initial_state = [encoded_images, encoded_images]
        logits = self.decoder(captions, initial_state)
        return logits

model = ImageCaptioningModel(encoder, decoder)
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

# Training
model.fit(train_dataset, validation_data=val_dataset, epochs=5, steps_per_epoch=len(train_images)//batch_size, validation_steps=len(val_images)//batch_size)
model.save('./model')

tokenizer_json = tokenizer.to_json()
with open('./tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))

#Trying to create a caption for an image
def preprocess_image(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
    img = tf.keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

# Function to generate a caption
def generate_caption(image_path):
    input_image = preprocess_image(image_path)
    initial_state = model.encoder(input_image)
    state = [initial_state, initial_state]

    # Start with the first word in the vocabulary if '<start>' is not available
    start_token = tokenizer.word_index.get('<start>', next(iter(tokenizer.word_index.values())))
    input_seq = tf.expand_dims([start_token], 0)
    result = []

    for i in range(max_caption_length):
        predictions = model.decoder(input_seq, state)
        predictions = predictions[:, -1, :]
        predicted_id = tf.argmax(predictions, axis=-1).numpy()[0]
        word = tokenizer.index_word.get(predicted_id, '')
        if word == '<end>' or word == '':
            break
        result.append(word)
        input_seq = tf.expand_dims([predicted_id], 0)

    return ' '.join(result)


# Example usage
image_path = 'dog.jpg'  # Replace with your image path
caption = generate_caption(image_path)
print("Generated Caption:", caption)
