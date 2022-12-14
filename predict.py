import os.path
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np

def predict(teks):
    input_list = []
    input_list.append(teks)
    
    tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
    
    tokenizer.fit_on_texts(input_list)
    word_index = tokenizer.word_index
    
    sequences_input = tokenizer.texts_to_sequences(input_list)
    padded_input = pad_sequences(sequences_input, maxlen=75, truncating='post')
    model = tf.keras.models.load_model('model_ann.h5')
    predicted = model.predict(padded_input)

    rounded = [np.round(x) for x in predicted]
    for i in rounded:
        if i == 1:
            return "Sentimen Negatif"
        else:
            return "Sentimen Positif"
        break
