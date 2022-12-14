import streamlit as st
import tensorflow as tf
import time
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import predict

model = tf.keras.models.load_model('model_ann.h5')


def main():
    st.header('Prediksi Sentimen Ulasan New Star Cineplex Pasuruan')
    text = st.text_input('Masukkan Teks Ulasan', 'Bioskop sangat nyaman')
    if st.button('Mulai Analisis'):
        if text.strip() == '':
            st.error('Cek kembali teks ulasan', icon="🚨")
        else:
            with st.spinner('Wait for it...'):
                time.sleep(0.1)
                st.success('Success!')
                
            sentimen = predict.predict(text)
            st.write(sentimen)
    else:
        st.write('')


if __name__ == '__main__':
    main()
