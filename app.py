from deepface import DeepFace
import cv2
import numpy as np
from PIL import Image
import streamlit as st

st.header('AI&DS::RCEE')
st.title('Face Recognition')

image = st.file_uploader('Take any image')

if image:
    img = Image.open(image)
    st.image(img)
    img = np.array(img)
    objs = DeepFace.find(img,'facerecognition1/database',model_name='Facenet512')
    name = objs[0]['identity'].split('/')[-1].split('.')[0]
    st.write(' Hi ' + '  name') 

     
