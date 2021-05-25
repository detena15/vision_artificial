#LIBRERIAS
import cv2
import numpy as np

#FUNCION: recibe una imagen y te devuelve las coordenadas de las caras
def face_detector(img, face_cascade, eye_cascade, face_f):   

    #variables face_f
    xf = face_f[0]
    yf = face_f[1]
    wf = face_f[2]
    hf = face_f[3]
    
    #variables img
    xi = 0
    yi = 0
    wi = img.shape[1]
    hi = img.shape[0]

    #apertura de face_f con relacion a la img
    c = float(0.1) #esto es un 10 %
    
    print("face_f: ", xf, xf + wf, yf, yf + hf)
    #roi_i = img[yf: yf + hf, xf: xf + wf]
    #cv2.imshow("roi_i", roi_i)

    if xf != xi or yf != yi or wf != wi or hf != hi: #(tendre que ver si AND o OR)
        #face_f no es igual a img, hace falta la apertura
    
        y1 = yf - round(c * hf)
        y2 = yf + hf + round(c * hf)
        x1 = xf - round(c * wf)
        x2 = xf + wf + round(c * wf)

        roi_f = img[y1: y2, x1: x2]
        
        print("Face apertura: ", x1, x2, y1, y2)
        cv2.imshow('Face apertura',roi_f)

    else:

        #face_f es igual a img, no hace falta la apertura
    
        roi_f = img[face_f[1] : face_f[1] + face_f[3], face_f[0] : face_f[0] + face_f[2]]

        #cv2.imshow('roi_f',roi_f)



    #paso el roi_f a gris para un mejor tratamiento
    gray_img = cv2.cvtColor(roi_f,cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray_img",gray_img)
    
    #aplicar el clasificador de caras sobre la imagen y guardo el resultado en faces: seran la x, y, height y width
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.04, minNeighbors=5)
    print("Faces: ", faces)

    if type(faces) == np.ndarray:

        flag = -1

        for x,y,w,h in faces:

            flag = flag + 1

            #print("Face: ", x,y,w,h)
            
            if w >= 100 and w <= 125 and h >= 100 and h <= 125:
                print("Entro en el if de tamaño")
                #Region Of Interest
                print("Face: ", x,y,w,h)
                roi_gray = gray_img[y:y+h, x:x+w]
                
                cv2.imshow("roi_gray", roi_gray)

                #aplico el clasificador de ojos sobre la imagen de interes que se supone que es una cara y guardo el resultado en eyes
                eyes = eye_cascade.detectMultiScale(roi_gray)
                
                c_eyes = 0

                for ex,ey,ew,eh in eyes:
                    
                    c_eyes = c_eyes + 1

                if c_eyes >= 2: #si hay mínimo dos ojos (a veces la boca abierta la detecta como un tercer ojo), es una cara
                    print("faces[flag]", faces[flag])
                    return faces[flag]
      
            
            
               
    