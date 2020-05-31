import numpy as np
import cv2 as cv2
import sys
import tensorflow as tf 

# Get image
imagePath = sys.argv[1]

cascPath = "haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=5,
    minSize=(40, 40),
    flags = cv2.CASCADE_SCALE_IMAGE
)


# transform images for emotion detection
cropped = []
for i in range(0, len(faces)):
    cropped.append(image.copy())
    x, y, w, h = faces[i]
    cropped[i] = cropped[i][y:y+h, x:x+w]
    cropped[i] = cv2.resize(cropped[i], (48,48))
    cropped[i] = cv2.cvtColor(cropped[i], cv2.COLOR_BGR2GRAY)

# labels for prediction
labels = {0:'anger', 1:'disgust', 2:'fear', 3:'happiness', 4: 'sadness', 5: 'surprise', 6: 'neutral'}

# load model
model = tf.keras.models.load_model('finalNetNoGen.h5')
    
# predict emotions using the loaded model
predictions = []
for i in range(0, len(faces)):
    prediction_classes = model.predict(cropped[i].reshape(1,48,48,1))
    prediction = np.argmax(prediction_classes, axis=-1)
    predictions.append(labels[prediction[0]])
    label = labels[prediction[0]]
    x, y, w, h = faces[i]
    # draw rectangles and detected emotions
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(image, label, (x, y+h), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255))

# show image
cv2.imshow("Faces with emotions", image)
cv2.waitKey(0)