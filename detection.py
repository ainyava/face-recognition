import cv2

# Open image
path_to_image = 'samples/Lost.jpg'
org = cv2.imread(path_to_image)

if org is not None:

	# Convert image to grayscale
    img = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)

    # Create Cascade Classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

	# Detect faces using the classifier
    detected_faces = face_cascade.detectMultiScale(image=img, scaleFactor=1.2, minNeighbors=2)

    # Draw rectangles around faces on the original, colored image
    for (x,y, width, height) in detected_faces:
        cv2.rectangle(org, (x, y), (x+width, y+height), (0, 0, 255), thickness=2)

    # Open a window to display the results
    cv2.imshow(f'Detected faces in {path_to_image}', org)
    cv2.waitKey(0)
    cv2.destroyAllWindows()