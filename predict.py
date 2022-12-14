import os.path
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('model_ann.h5')

def predict(teks):
    tokenizer = Tokenizer(num_words=500, oov_token="<OOV>")
    
    tokenizer.fit_on_texts(teks)
    word_index = tokenizer.word_index
    
    sequences_input = tokenizer.texts_to_sequences(teks)
    padded_input = pad_sequences(sequences_input, maxlen=75, truncating='post')
    
    predicted = model.predict(padded_input)

    rounded = np.round(predicted)
    for i in rounded:
        return i[0]
#         if i[0] == 0:
#             return ("Sentimen Positif")
#         elif i[0] == 1:
#             return ("Sentimen Negatif")
#         break    
