#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import PIL.Image


# In[2]:


facedetect= cv2.CascadeClassifier("/Users/macmojave/Downloads/haarcascade_frontalface_default.xml")


# In[4]:


# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture('/Users/macmojave/Downloads/faces01.mp4')


# In[5]:


faces =[]


# Read until video is completed
while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    i = 0
    col= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(
            col,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30,30),
            flags= cv2.CASCADE_SCALE_IMAGE
    )
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        i = i+1
            
        # Draw a rectangle around the text
        cv2.rectangle(frame,(1000,0), (1200,30), (255,255,255), -1)
        # Display the count of faces detected
        cv2.putText(frame, 'Faces detected: '+str(i), (1000,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    
    if ret == True:
 
        # Display the resulting frame
        cv2.imshow('Frame', frame)
 
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
 
    # Break the loop
    else:
        break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()


# In[ ]:




