from pathlib import Path
import pickle
import sys

import numpy as np
from keras.models import load_model

def _get_map():
    chars = pickle.loads(Path('data/chars.pickle').read_bytes())
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    return chars, char_indices, indices_char

chars, char_indices, indices_char = _get_map()
model = load_model('data/weight.hdf5')
MAXLEN=20

def _sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate(seed, diversity=0.5):
    generated = ""
    sentence = seed
    generated += sentence

    for i in range(40):
        x_pred = np.zeros((1, MAXLEN, len(chars)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_indices[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = _sample(preds, diversity)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char
    return generated

if __name__ == "__main__":
    generate(' '.join(sys.argv[1:]))
