import os
import sys
import pickle
import cv2
from imutils import paths
import face_recognition

# find path of xml file containing haarcascade file
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# load the known faces and embeddings saved in last file
data = pickle.loads(open('face_encodings', "rb").read())

# set webcam
video_capture = cv2.VideoCapture(0)

while True:
    # find path to the image we want to detect face
    ret, frame = video_capture.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # convert frame to Greyscale for haarcascade
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
    # the facial embeddings for face in input
    encodings = face_recognition.face_encodings(rgb)
    names = []

    # loop over the facial embeddings incase
    # we have multiple embeddings for multiple fcaes
    for encoding in encodings:

        # compare encodings with encodings in data["encodings"]
        # matches contain array with boolean values and True for the embeddings it matches closely
        # and False for rest
        matches = face_recognition.compare_faces(data['encodings'], encoding)

        #set name Unknown if no encoding matches
        name = 'Unknown'
        # check to see if we have found a match
        if True in matches:
            #Find positions at which we get True and store them
            matched_indexes = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matched_indexes:
                # check the names at respective indexes we stored in matchedIdxs
                name = data['names'][i]
                # increase count for the name we got
                counts[name] = counts.get(name, 0) + 1
                # set name which has highest count (to increase accuracy)
                name = max(counts, key=counts.get)

        # update the list of names
        names.append(name)

        # loop over the recognized faces
        for ( (x, y, w, h), name) in zip(faces, names):
            # draw reactangle around face and the predicted name on the image
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # open a window to display the results
    cv2.imshow(f'Detected face for Webcam frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()