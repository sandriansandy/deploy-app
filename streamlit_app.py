import streamlit as st
import tensorflow as tf
import time

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

            teks_df = model.predict(text)
            st.table(teks_df)
    else:
        st.write('')

if __name__=='__main__':
    main()
