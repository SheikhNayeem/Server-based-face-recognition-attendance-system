import face_recognition
import cv2
import numpy as np
import datetime
import requests
#import os
#from numpy import asarray
#from numpy import savez_compressed
#from PIL import Image
from numpy import load
video_capture = cv2.VideoCapture('rtsp://*********')

#video_capture = cv2.VideoCapture(0)
known_face_encodings = load('picnpy.npy')
#known_face_encodings = picdict_data['arr_0']
known_face_names = load('namenpy.npy')

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
#frame=cv2.imread("D:\\python\\ak\\pic\\Nayeem.png",1)
#frame=cv2.imread("/var/www/html/uploads/img.jpg",1)
fix ='kk'
latetime=datetime.datetime.now()
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, 0.4)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)
            #print(name)
            if (name != "Unknown"):
                now = datetime.datetime.now()
                eid = name[(name.find("_") + 1):]
                ename = name[:(name.find("_"))]
                if fix != ename:
                    print(ename)
                    fix = ename
                    latetime = now
                    responce = requests.get("http://********?empid=" + str(eid) + "&type=face&name=" + str(ename))
                else:
                    if datetime.timedelta(days=0, seconds=0, microseconds=0, milliseconds=0, minutes=1, hours=0,weeks=0) < (now - latetime):
                        print(ename)
                        latetime = now
                        responce = requests.get("http://********?empid=" + str(eid) + "&type=face&name=" + str(ename))
                """print(name)
                now = datetime.datetime.now()
                print(now.strftime("%Y-%m-%d %H:%M:%S"))
                eid = name [(name.find("_")+1):]
                ename= name[:(name.find("_"))]
                responce = requests.get("http://*********?empid="+ str(eid) +"&type=face&name="+str(ename))
"""
    process_this_frame = not process_this_frame


    # Display the resultsZZ
    """for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)"""

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

