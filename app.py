## Imported Modules ##
import streamlit as st
from tensorflow import keras
from keras.models import load_model
from keras.utils import load_img, img_to_array
import numpy as np
import cv2
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from  matplotlib.image import imread 

## Functions ##

@st.cache(allow_output_mutation = True) 
def load_VGG():
    model= load_model('model.h5')
    return model

@st.cache(allow_output_mutation = True) 
def load_Inception():
    model= load_model('inception_v3_model.h5')
    return model

 
def import_and_predict(image_data, model):
    size = (224,224)    
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    img_reshape = img[np.newaxis,...]

    prediction = model.predict(img_reshape)
    
    if prediction[0]>=0.5:
        final_result= 'Real'
    else:
        final_result= 'AI Generated'

    return final_result,prediction[0]




       
## Layout##
rad = st.sidebar.radio("Navigation",["Home","Predict AI/Real"])


if rad=="Home":
    st.title("AI_Image_Classifier")
    st.subheader("By Hardik Pahwa")
    st.markdown("""Differentiating between AI-generated content and real content is crucial in today's digital era.<br> As AI algorithms and deepfake technology advance, it becomes increasingly difficult to discern authenticity.<br> This ability is essential for preserving information integrity, combating misinformation, and maintaining trust in various fields.<br> Reliable tools and techniques are necessary to identify AI-generated content, ensuring transparency and protecting against manipulation.""",True)
    st.markdown("Infact the text  above and image below was also generated using AI :wink:")
    st.image("sample.jpg")
    
elif rad=="Predict AI/Real":
    st.write("""
            # Image Classification
    """)
    v1 = st.radio("Choose Model",["VGG16","Inceptionv3"],index=0)
    with st.spinner("Loading Model..."):
        if v1=="VGG16":
            model = load_VGG()
        else:
            model = load_Inception()
    
        
    file = st.file_uploader("Upload an Image")
    st.write(file)
    st.set_option('deprecation.showfileUploaderEncoding', False)
        
    if file != None:
        image = Image.open(file)
        st.image(file,use_column_width=True)
        final_result= import_and_predict(image,model)[0]
        result = import_and_predict(image,model)[1]
        st.write("The image is :")
        st.write(final_result)
        st.write(result)
        
       
        
            
    else:
        st.write("No file uploaded")
