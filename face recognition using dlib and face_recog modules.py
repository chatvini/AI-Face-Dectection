#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import face_recognition


# In[7]:


# pip install dlib


# In[10]:


# pip install face_recognition


# In[2]:


cap= cv2.VideoCapture("/Users/macmojave/Downloads/faces01.mp4")


# In[5]:


face_locations =[]

while(True):
    
    #reading from frame
    ret, frame = cap.read()
    
    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    #faces = detector.detect_faces(frame)
   
    for (x,y,w,h) in face_locations:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
   
    # Display the resulting frame
    cv2.imshow('Frame', frame)
 
# Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
 
# Break the loop
#else:
 #   break
        
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()
    

