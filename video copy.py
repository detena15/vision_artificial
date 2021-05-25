#LIBRERIAS
import cv2
import numpy as np

from face_detector import *

#CARGA DE LOS CLASIFICADORES
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#MAIN
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('directo.mp4')

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
  

# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()

  if ret == True:
    
    #Analizar si frame tiene cara
    face = face_detector(frame,  face_cascade, eye_cascade)
    
    if type(face) == np.ndarray:
      
      start_point = (face[0], face[1])
      end_point = ((face[0] + face[2]), (face[1] + face[3]))
      color = (255, 0, 0)
      thickness = 2
      frame = cv2.rectangle(frame, start_point, end_point, color, thickness)

      # Display the resulting frame
      cv2.imshow('Frame',frame)

      
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'): #a 38 es una velocidad buena sin procesar
      break
  
  # Break the loop
  else: 
    break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()