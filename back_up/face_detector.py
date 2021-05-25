#LIBRERIAS
import cv2
import numpy as np

#FUNCION: recibe una imagen y te devuelve las coordenadas de las caras
def face_detector(img):

    #cargar los xml (clasificadores en cascada), tanto el de la cara como el de los ojos
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    #pasar la imagen a gris para un mejor tratamiento
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #aplicar el clasificador de caras sobre la imagen y guardo el resultado en faces: seran la x, y, height y width
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.04, minNeighbors=5)
    

    if type(faces) == np.ndarray: #si ha detectado caras, analiza por cada presunta cara si tiene ojos dentro

        flag = -1 #me va a indicar qué rectángulo de faces devuelvo (empieza en -1 porque el primer valor del array de faces es 0)

        c_faces = 0

        for x,y,w,h in faces: #x,y,w,h son del tipo numpy.int32, por lo que cualquier numero que use para operar con ellos deben ser numpy.int32

            c_faces = c_faces + 1

        

            #Region Of Interest
            roi_gray = gray_img[y:y+h, x:x+w]

            #aplico el clasificador de ojos sobre la imagen de interes que se supone que es una cara y guardo el resultado en eyes
            eyes = eye_cascade.detectMultiScale(roi_gray)
            
            flag = flag + 1 #la primera posicion del array de caras es 0 = -1 + 1

            c_eyes = 0

            for ex,ey,ew,eh in eyes:
                
                c_eyes = c_eyes + 1

                #para dibujar los ojos en el sitio correcto, debo sumarle la x e y del rectangulo de la cara. Si no, pinta los ojos con respecto al origen del frame
                start_point = (ex + x, ey +y )
                end_point = ((ex + x + ew),(ey + y + eh))
                color = (0, 255, 0)
                thickness = 2
                cv2.rectangle(img, start_point, end_point, color, thickness)


            if c_eyes >= 2: #si hay mínimo dos ojos (a veces la boca abierta la detecta como un tercer ojo), es una cara
                return faces[flag]
               
    