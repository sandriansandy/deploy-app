import streamlit as st
import tensorflow as tf
import time
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = tf.keras.models.load_model('model_ann.h5')
def main():
    st.title("Sandy Ganteng")
    st.header('Prediksi Sentimen Ulasan New Star Cineplex Pasuruan')
    text = st.text_input('Masukkan Teks Ulasan', 'Bioskop sangat nyaman')
    if st.button('Mulai Analisis'):
        if text.strip()=='' :
            st.error('Cek kembali teks ulasan', icon="🚨")
        else:
            with st.spinner('Wait for it...'):
                time.sleep(0.1)
                st.success('Success!')
            
            tokenizer = Tokenizer(num_words = 10000, oov_token="<OOV>")    
            sequences_input = tokenizer.texts_to_sequences(text)
            padded_input = pad_sequences(sequences_input,maxlen=75, truncating="post")
            teks_df = model.predict(padded_input)
            rounded = [np.round(x) for x in predicted]
            for i in rounded:
                if i == 1:
                    print("Sentimen Negatif")
                else:
                    print("Sentimen Positif")
                break
            # st.table(teks_df)
    else:
        st.write('')

if __name__=='__main__':
    main()
