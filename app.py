from deepface import DeepFace
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import json

st.header('AI&DS::RCEE')
st.title('Face Recognition')

image = st.file_uploader('Take any image')

if image:
    img = Image.open(image)
    #st.image(img)
    img = np.array(img)
    objs = DeepFace.find(img,'database',model_name='Facenet512',enforce_detection=False)
    name = str(objs[0]['identity'][0])
    name = name.split('/')[-1].split('.')[0]
    st.write(' Hi ' + '  ' + name)
    '''information = DeepFace.analyze(img)
    infomation = json.load(information)
    age = information['age']
    emotion = information['dominant_emotion']
    race = information['dominant_race']
    gender = information['gender']
    st.write('Your age is   ',age)
    st.write('You are  ',gender)'''
    st.write('You are from  ', race)
    st.write('You are feeling very ', emotion)
    

     
