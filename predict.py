import os.path
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('model_ann.h5')
list_teks = []
word_index=[]
with open("word_index.txt",'r') as indeks:
    for word in indeks:
        word_index.append(word)
        
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(word_index)

def predict(teks):
    list_teks.append(teks)
    sequences_input = tokenizer.texts_to_sequences(list_teks)
    padded_input = pad_sequences(sequences_input, maxlen=75, truncating='post')
    
    predicted = model.predict(padded_input)
#     return predicted
    rounded = np.round(predicted)
    return rounded
#     for i in rounded:
#         if i[0] == 0:
#             return ("Sentimen Positif")
#         elif i[0] == 1:
#             return ("Sentimen Negatif")
#         break    
