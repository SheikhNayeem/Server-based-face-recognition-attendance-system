import face_recognition
import cv2
import numpy as np
import os
from numpy import asarray
from numpy import save
#from PIL import Image
#video_capture = cv2.VideoCapture(0)
path="D:\\python\\ak\\pic"
images =[]
className= []
mylist= os.listdir(path)
for cl in mylist:
    cimg= cv2.imread(f'{path}/{cl}')
    images.append(cimg)
    className.append(os.path.splitext(cl)[0])
print(className)
def findencoding(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
known_face_encodings = findencoding(images)
known_face_names = className
encodingListKnown = findencoding(images)
save('D:\\python\\array\\picnpy.npy', known_face_encodings)
save('D:\\python\\array\\namenpy.npy', known_face_names)
print(len(encodingListKnown))