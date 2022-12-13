import streamlit as st
import tensorflow as tf

model = tf.keras.models.load_model('model_ann.h5')
def main():
    st.title("Sandy Ganteng")
    st.header('Prediksi Sentimen dan Aspek Ulasan')
    text = st.text_input('Masukkan Teks Ulasan', 'Makanan enak sekali')
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
