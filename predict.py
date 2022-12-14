import os.path
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np

def predict(teks):
    tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
    input_list = []
    input_list.append(teks)
    sequences_input = tokenizer.texts_to_sequences(input_list)
    padded_input = pad_sequences(sequences_input, maxlen=75, truncating='post')
    model = tf.keras.models.load_model(
        open(os.path.basename('model_ann.h5'), 'rb'))
    predicted = model.predict(processed_input)

    rounded = [np.round(x) for x in predicted]
    for i in rounded:
        if i == 1:
            return "Sentimen Negatif"
        else:
            return "Sentimen Positif"
        break
