import glob
from pathlib import Path
import pickle
import random
import sys

import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.losses import CategoricalCrossentropy
from keras.optimizers import RMSprop
from keras.callbacks import LambdaCallback
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau

def read_files():
    corpus = ""
    for filename in glob.glob("data/*.txt"):
        corpus += (Path.cwd()/filename).read_text().lower()
    return corpus

text = read_files()

# Get unique characters
chars = sorted(list(set(text)))
# This can be used to generate the indices
Path('data/chars.pickle').write_bytes(pickle.dumps(chars))

# Map characters to indices and vice versa
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
model = Sequential()
MAXLEN = 20
MINLEN = 5

def generate_x_y(text):
    step = 3
    sentences = []
    next_chars = []
    for i in range(0, len(text) - MAXLEN, step):
        # Add this to have variable lengths
        length = random.randint(0,MAXLEN-MINLEN)
        sentences.append(text[i: i + MAXLEN - length])
        next_chars.append(text[i + MAXLEN - length])

    x = np.zeros((len(sentences), MAXLEN, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1
    return x, y

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(text) - MAXLEN - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + MAXLEN]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, MAXLEN, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

def generate(start):
    model = load_model('data/weights.hdf5')

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = start
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, MAXLEN, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

def main():
    global text
    global model

    x, y = generate_x_y(text)

    model.add(LSTM(128, input_shape=(MAXLEN, len(chars)), unroll=True))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.01)
    lossfn = CategoricalCrossentropy()
    model.compile(loss=lossfn, optimizer=optimizer)
    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

    filepath = "data/weights.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                                  patience=1, min_lr=0.001)
    callbacks = [print_callback, checkpoint, reduce_lr]
    model.fit(x, y, batch_size=128, epochs=2, callbacks=callbacks)


if __name__ == '__main__':
    main()
