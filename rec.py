import numpy as np
import cv2
import sys
import os
#from mtcnn.mtcnn import MTCNN
import tensorflow as tf
import cv2
def detectFaces(img):
        # Convertinto grayscale since it works with grayscale images
        gray_img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#faces = detector.detect_faces(pixels)
        faces = face_cascade.detectMultiScale(gray_img, 1.3, 8)
        if len(faces):
                for f in faces:
                       # x, y, width, height = f['box']
                         frame = cv2.rectangle(img, (f[0], f[1]), (f[0]+f[2], f[1]+f[3]), (0, 255, 0), 2) 
                         img = img[f[1]:f[1]+f[3], f[0]:f[0]+f[2]]    
                         img = img[32:256, 32:256]
                         img = cv2.resize(img, (imgXdim, imgYdim)) 
                         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                         image = img.reshape(48,48,1)
                         image = image.astype(np.float32)/225
                         image = np.array(image,ndmin=4)
                         pri = model.predict(image)
                         argmax = np.argmax(pri)
                         frame = cv2.putText(frame, labels[argmax], (f[0] + 5, f[1] + 15), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 0, 0) , 1)
                return frame
        else:
                return img


labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
face_cascade = cv2.CascadeClassifier('C:/Users/hunter/Desktop/facedectaction.xml')
model =tf.keras.models.load_model("C:/Users/hunter/Desktop/fer-self-made.h5")
# create the detector, using default weights
#detector = MTCNN()
cap = cv2.VideoCapture(0)
imgXdim=48
imgYdim=48
while(True):
    ret, frame = cap.read()
    if ret:
        frame=detectFaces(frame)
        cv2.imshow('image',frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            breakpoint
