import numpy as np 
import face_recognition as fr
import cv2 

image= fr.load_image_file('face1_01.jpg')
image2= cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

size= (1000,700)
image_resize= cv2.resize(image2, size)

face1=fr.face_encodings(image_resize)[0]
image_compare2=fr.load_image_file('face1_02.jpg')
image_compare2_encode=fr.face_encodings(image_compare2)[0]
result=fr.compare_faces([face1], image_compare2_encode)

print(result)

face2=fr.face_encodings(image_resize)[0]
image_compare2=fr.load_image_file('face2.jpg')
image_compare2_encode=fr.face_encodings(image_compare2)[0]
result=fr.compare_faces([face1], image_compare2_encode)

print(result)
