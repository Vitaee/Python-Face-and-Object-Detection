import cv2 
import numpy as np

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 

video_capture = cv2.VideoCapture(0) 
img_counter = 0 
while True: 

    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    k = cv2.waitKey(1) 



    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor=1.5,
        minNeighbors=5, 
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
        )

       
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame,"Insan",(x +0, y+ -10),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),2)



    silgi = cv2.imread("silgi.jpg",0)
    w, h = silgi.shape
    res = cv2.matchTemplate(gray,silgi,cv2.TM_CCOEFF_NORMED)
    esik_degeri = 0.8
    loc = np.where(res>esik_degeri)


    
    for x in zip(*loc[::-1]):
        cv2.rectangle(frame, x, (x[0] + h, x[1] + w), (0, 255, 0), 2) 
        cv2.putText(frame, "Silgi", (x[0] + 0, x[1] + -10), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 2)
        


    silgi1 = cv2.imread("silgi1.jpg",0) 
    w, h = silgi1.shape
    res1 = cv2.matchTemplate(gray,silgi1, cv2.TM_CCOEFF_NORMED)
    esik_degeri1 = 0.8
    loc1 = np.where(res1 > esik_degeri1)
    
    for y in zip(*loc1[::-1]):
        cv2.rectangle(frame, y , (y[0] + h, y[1] + w), (0,255,0),2)
        cv2.putText(frame, "Silgi", (x[0] + 0, x[1] + -10), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 2)

    

    usb = cv2.imread("usb.jpg",0)
    w, h = usb.shape
    res_usb = cv2.matchTemplate(gray, usb, cv2.TM_CCOEFF_NORMED)
    esik_degeri2 = 0.8
    loc2 = np.where(res_usb > esik_degeri2)

    for a in zip(*loc2[::-1]):
        cv2.rectangle(frame, a , (a[0] + h , a[1] + w), (0,255,0), 2)
        cv2.putText(frame, "Usb", (a[0] + 0, a[1] + -10), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 2)




    usb1 = cv2.imread("usb1.jpg",0)
    w , h = usb1.shape
    res_usb1 = cv2.matchTemplate(gray, usb1, cv2.TM_CCOEFF_NORMED)
    esik_degeri3 = 0.8
    loc3 = np.where(res_usb1 > esik_degeri3)

    for b in zip(*loc3[::-1]):
        cv2.rectangle(frame, b, (b[0] + h, b[1] + w), (0,255,0), 2)
        cv2.putText(frame,"Usb", (b[0] + 0, b[1] + -10), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 2)




    cv2.imshow('Obje ve Yuz Tanima', frame) 

    if k%256 == 27:  
         break

    elif k%256 == 32: 
       
        img_name = "resim_{}.jpg".format(img_counter)
        cv2.imwrite(img_name, frame) 
        print("{} kayıt edildi!".format(img_name)) 
        img_counter += 1 


video_capture.release()
cv2.destroyAllWindows()
