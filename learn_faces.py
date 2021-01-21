import os
import pickle
import cv2
from imutils import paths
import face_recognition

# get paths of each file in folder named images
# that contains my data(folders of various persons)
image_paths = list(paths.list_images('images'))

known_encodings = []
known_names = []

# loop over the image paths
for (i, image_path) in enumerate(image_paths):
    # extract person name from the image path
    name = image_path.split(os.path.sep)[-2]
    # load the input image and convert it from BGR (OpenCV ordering) to RGB (dlib ordering)
    image = cv2.imread(image_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # use Face Recognition to locate faces
    boxes = face_recognition.face_locations(image, model='hog')
    # compute the facial embedding for the face
    encodings = face_recognition.face_encodings(rgb, boxes)
    # loop over the encodings
    for encoding in encodings:
        known_encodings.append(encoding)
        known_names.append(name)

# save emcodings along with their names in dictionary data
# use pickle to save data into a file for later use
data = {'encodings': known_encodings, 'names': known_names}
with open('face_encodings', 'wb') as f:
    f.write(pickle.dumps(data))
    f.close()
