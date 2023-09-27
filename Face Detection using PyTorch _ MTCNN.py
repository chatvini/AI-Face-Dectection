#!/usr/bin/env python
# coding: utf-8

# In[29]:


#Importing Libraries

import numpy as np
import mtcnn
from mtcnn.mtcnn import MTCNN
import torch
import mmcv, cv2
from PIL import Image 
from matplotlib import pyplot as plt


# In[30]:


# Create a VideoCapture object and read from input file

cap= cv2.VideoCapture("/Users/macmojave/Downloads/faces01.mp4")


fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('/Users/macmojave/Downloads/FacesOutput.avi', fourcc, 40.0, (int(cap.get(3)),int(cap.get(4))))


# In[31]:


# Create face detector

detector = MTCNN()


# In[32]:


#Initializing empty array for storing faces
face_locations =[]

while(True):
    
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    i = 0
    
   # if len(results) != 0: x,y,w,h = result[0]['box']

#else: continue
    if ret:
        face_locations = detector.detect_faces(frame)
    
    #checking for 'none' condition (no face detected)
        if face_locations != []:
             # Display the box and faces
            for person in face_locations:
                bounding_box = person['box']
                keypoints = person['keypoints']
    
                cv2.rectangle(frame,
                          (bounding_box[0], bounding_box[1]),
                          (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                          (0,155,255),
                          2)
                i = i+1
                
                
                #fheight = fshape[0]
                #fwidth = fshape[1]
                
            # Draw a rectangle around the text
                cv2.rectangle(frame,(1000,0), (1200,30), (255,255,255), -1)
            # Display the count of faces detected
                cv2.putText(frame, 'Faces Detected: '+str(i), (1000,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
       
            out.write(frame)
        #Display the resulting frame
        #cv2.imshow('Frame', frame)
        


 
# Press 'q' on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
   
    else:
        break
        


        
# When everything done, release the video capture object
cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows()    


# In[ ]:




