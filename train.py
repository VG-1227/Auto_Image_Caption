import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications.vgg16 import preprocess_input 
from tensorflow.keras.layers import LSTM, Embedding, Dense, Dropout, Input, add
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import corpus_bleu
import numpy as np
import os
import pickle
from tqdm.notebook import tqdm # Progress bar

# Load descriptions
def load_descriptions(filename):
  descriptions = dict()
  with open(filename, 'r') as f:
    for line in f:
      tokens = line.split()
      image_id, image_desc = tokens[0], tokens[1:]
      image_id = image_id.split('.')[0]
      image_desc = ' '.join(image_desc)
      if image_id not in descriptions:
        descriptions[image_id] = list()
      descriptions[image_id].append(image_desc)
  return descriptions

descriptions = load_descriptions('/content/captions.txt')

# Clean descriptions
def clean_descriptions(descriptions):
  import string
  table = str.maketrans('', '', string.punctuation)
  for key, desc_list in descriptions.items():
    for i in range(len(desc_list)):
      desc = desc_list[i]
      desc = desc.split()
      desc = [word.lower() for word in desc]
      desc = [word.translate(table) for word in desc]
      desc = [word for word in desc if len(word)>1]
      desc = [word for word in desc if word.isalpha()]
      desc_list[i] =  ' '.join(desc)

clean_descriptions(descriptions)

# Create vocabulary of all unique words
def to_vocabulary(descriptions):
  all_desc = set()
  for key in descriptions.keys():
    [all_desc.update(d.split()) for d in descriptions[key]]
  return all_desc

vocabulary = to_vocabulary(descriptions)

# Save descriptions to file
def save_descriptions(descriptions, filename):
  lines = list()
  for key, desc_list in descriptions.items():
    for desc in desc_list:
      lines.append(key + ' ' + desc)
  data = '\n'.join(lines)
  with open(filename, 'w') as f:
    f.write(data)

save_descriptions(descriptions, 'descriptions.txt')
# Load VGG16 model
model = VGG16(weights='imagenet')
model = Model(inputs=model.inputs, outputs=model.layers[-2].output) # Remove the last layer

# Extract features from each image
def extract_features(directory):
  features = dict()
  for name in tqdm(os.listdir(directory)):
    filename = directory + '/' + name
    image = load_img(filename, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    image_id = name.split('.')[0]
    features[image_id] = feature
  return features

features = extract_features('/content/Images')

# Save features to file
pickle.dump(features, open('features.pkl', 'wb'))
# Load features from file
def load_features(filename):
  with open(filename, 'rb') as f:
    features = pickle.load(f)
  return features

# Load all features
features = load_features('/content/features.pkl')

# Load all descriptions
def load_descriptions(filename):
    descriptions = dict()
    with open(filename, 'r') as f:
        for line in f:
            # Split the line by space, only once to avoid splitting captions
            tokens = line.split(' ', 1)
            
            # Handle cases where there might be missing descriptions
            if len(tokens) < 2:
                continue
            
            image_id, image_desc = tokens[0], tokens[1].strip()
            
            if image_id not in descriptions:
                descriptions[image_id] = list()
            descriptions[image_id].append('startseq ' + image_desc + ' endseq')  
    return descriptions

# Load all descriptions, using the corrected load_descriptions function
descriptions = load_descriptions('/content/descriptions.txt')



all_words = []
for key, desc_list in descriptions.items():
    for desc in desc_list:
        words = desc.split()
        all_words.extend(words)

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(all_words)
vocab_size = len(tokenizer.word_index) + 1
del all_words # Delete to free memory


# Data generator function
def data_generator(tokenizer, max_length, descriptions, photos, vocab_size, batch_size):
    keys = list(descriptions.keys())
    num_batches = len(keys) // batch_size  # Calculate the number of batches per epoch
    while True:
        for batch_index in range(num_batches):  # Iterate over batches within each epoch
            start = batch_index * batch_size
            end = start + batch_size
            batch_keys = keys[start:end]
            X1, X2, y = list(), list(), list()
            for key in batch_keys:
                if key in photos:
                    for desc in descriptions[key]:
                        seq = tokenizer.texts_to_sequences([desc])[0]
                        for i in range(1, len(seq)):
                            in_seq, out_seq = seq[:i], seq[i]
                            in_seq = tf.keras.preprocessing.sequence.pad_sequences([in_seq], maxlen=max_length)[0]
                            out_seq = tf.keras.utils.to_categorical([out_seq], num_classes=vocab_size)[0]
                            X1.append(photos[key][0])
                            X2.append(in_seq)
                            y.append(out_seq)
            yield [np.array(X1), np.array(X2)], np.array(y)


# Determine the maximum sequence length
def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)


def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

# Prepare sequences
batch_size = 32  # Adjust as needed
max_len = max_length(descriptions)
train_generator = data_generator(tokenizer, max_len, descriptions, features, vocab_size, batch_size)

# Define the CNN-LSTM model
def define_model(vocab_size, max_length):
  inputs1 = Input(shape=(4096,))
  fe1 = Dropout(0.5)(inputs1)
  fe2 = Dense(256, activation='relu')(fe1)
  inputs2 = Input(shape=(max_length,))
  se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
  se2 = Dropout(0.5)(se1)
  se3 = LSTM(256)(se2)
  decoder1 = add([fe2, se3])
  decoder2 = Dense(256, activation='relu')(decoder1)
  outputs = Dense(vocab_size, activation='softmax')(decoder2)
  model = Model(inputs=[inputs1, inputs2], outputs=outputs)
  model.compile(loss='categorical_crossentropy', optimizer='adam',run_eagerly=True)
  return model

# Train the model
model = define_model(vocab_size, max_length(descriptions))
epochs = 10
#for i in range(epochs):
 # generator = data_generator(descriptions, features, tokenizer, max_length(descriptions))
 # model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)
model.fit(train_generator, steps_per_epoch=len(descriptions) // batch_size, epochs=epochs)

# Generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
  in_text = 'startseq'
  for i in range(max_length):
    sequence = tokenizer.texts_to_sequences([in_text])[0]
    sequence = pad_sequences([sequence], maxlen=max_length)
    yhat = model.predict([photo,sequence], verbose=0)
    yhat = np.argmax(yhat)
    word = word_for_id(yhat, tokenizer)
    if word is None:
      break
    in_text += ' ' + word
    if word == 'endseq':
      break
  return in_text

# Map an integer to a word
def word_for_id(integer, tokenizer):
  for word, index in tokenizer.word_index.items():
    if index == integer:
      return word
  return None

# Generate captions for all images
def generate_captions(model, descriptions, photos, tokenizer, max_length):
  actual, predicted = list(), list()
  for key, desc_list in descriptions.items():
    yhat = generate_desc(model, tokenizer, photos[key], max_length)
    references = [d.split() for d in desc_list]
    actual.append(references)
    predicted.append(yhat.split())

  # Calculate BLEU score
  print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
  print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
  print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
  print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

generate_captions(model, descriptions, features, tokenizer, max_length(descriptions))

#save the model
model.save('image_captioning_model.keras')
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)