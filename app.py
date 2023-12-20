import streamlit as st
import numpy as np
from keras.preprocessing import image
from keras.models import load_model



model = load_model('braintumordetectmodel.h5')



st.title('Classification of Brain Tumor using CNN')

st.text("")

upload_brain_photo = st.file_uploader('Please upload the photo of Brain MRI Image', type=['jpg', 'png'])


if upload_brain_photo is not None:

    brain_photo_uploaded = image.load_img(upload_brain_photo, target_size=(180, 180, 3))

    st.text("")

    col1, col2, col3 = st.columns (3)

    with col1:
        st.write (' ')
    with col2:
        st.image (brain_photo_uploaded, caption='Preview of the uploaded Brain MRI Image', width=250)
    with col3:
        st.write (' ')

    brain_photo_uploaded_to_arr = image.img_to_array(brain_photo_uploaded)

    brain_photo_uploaded_to_arr = brain_photo_uploaded_to_arr / 255

    brain_photo_uploaded_to_arr_expand = np.expand_dims(brain_photo_uploaded_to_arr, axis=0)

    prediction = (model.predict(brain_photo_uploaded_to_arr_expand) > 0.5).astype('int32')

    st.text("")

    if prediction[0][0] == 0:
        st.markdown("<div style='background-color: green; padding: 8px; border-radius: 10px; text-align: center; color: white; font-size: large'>The Brain MRI imaging that you uploaded shows no signs of any tumor.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='background-color: red; padding: 8px; border-radius: 10px; text-align: center; color: white; font-size: large'> The brain imaging that you uploaded indeed has signs of a tumor.</div>", unsafe_allow_html=True)