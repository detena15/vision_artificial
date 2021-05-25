#LIBRERIAS
import cv2
import numpy as np

from face_detector import *

#CARGA DE LOS CLASIFICADORES
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#VARIABLES GENERALES
face = np.array([0, 0, 0, 0])

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
    
    #inicializo el array de face: [0 0 w h].
    face_frame = np.array([0, 0, frame.shape[1], frame.shape[0]])
    
    #Analizar si frame tiene cara. Envio a la funcion un rectangulo inicial para analizar, o justo el que acabo de recibir en la ultima iteraccion
    if type(face) == np.ndarray: #asi evito los face vacios, que darian ademas error
      if face[0] != 0 or face[1] != 0 or face[2] != 0 or face[3] != 0:
        #Face ha cambiado
        face_f = face
      else:
        #Face no ha cambiado
        face_f = face_frame
    
    face = face_detector(frame,  face_cascade, eye_cascade, face_f)

    print("FACE QUE LLEGA A VIDEO.PY: ", face)

    if type(face) == np.ndarray: #si face no es un array vacio, pinta el rectangulo
      
      start_point = (face[0], face[1])
      end_point = ((face[0] + face[2]), (face[1] + face[3]))
      color = (255, 0, 0)
      thickness = 2
      frame = cv2.rectangle(frame, start_point, end_point, color, thickness)

    # Display the resulting frame
    cv2.imshow('Directo',frame)

      
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