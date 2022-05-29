import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import time
import pyttsx3
x = 0
c = 0
m = 0
d = 0

# Get the training data we previously made
data_path = 'data/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

# Create arrays for training data and labels
Training_Data, Labels = [], []

# Open training images in our datapath
# Create a numpy array for training data
for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

# Create a numpy array for both training data and labels
Labels = np.asarray(Labels, dtype=np.int32)

# Initialize facial recognizer
model = cv2.face.LBPHFaceRecognizer_create()

# Let's train our model 
model.train(np.asarray(Training_Data), np.asarray(Labels))

print("Model Training Complete!!!!!")

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Audio according to the user which apears in front og the camera.
def speak(audio):
    engine.say(audio)
    engine.runAndWait()


engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty("voice", voices[0].id)
engine.setProperty("rate", 140)
engine.setProperty("volume", 1000)


def face_detector(img, size=0.5):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if faces is ():
        return img, []

    for(x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))

    return img, roi

# Open Webcam
cap = cv2.VideoCapture(0)
while True:

    ret, frame = cap.read()

    image, face = face_detector(frame)

    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        # Pass face to prediction model
        # "results" comprises of a tuple containing the label and the confidence value
        result = model.predict(face)

        if result[1] < 500:
            confidence = int(100*(1-(result[1])/300))
            display_string = str(confidence)+'% Confidence it is user'
        cv2.putText(image, display_string, (100, 120),cv2.FONT_HERSHEY_COMPLEX, 1, (250, 120, 255), 2)

        if confidence > 75:
            cv2.putText(image, "Unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Cropper', image)
            x=x+1
        else:
            cv2.putText(image, "Locked", (250, 450),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face Cropper', image)
            c=c+1

    except:
        cv2.putText(image, "Face Not Found", (250, 450),cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Face Cropper', image)
        d=d+1
        pass

    if cv2.waitKey(1) == 13 or x == 2 or c == 30 or d == 20:
        break


cap.release()
cv2.destroyAllWindows()
if x >= 2:
    m = 1
    time.sleep(2)
    var = 'a'
    c = var.encode()
    # If it is an, authorized user.
    speak("Face recognition complete..It is matching with database...Welcome back Home...Door is opening for 5 seconds.")
    time.sleep(4)
elif c == 30:
    # If it is an, un-authorized user/intruder.
    speak("Face is not matching..please try again")
elif d == 20:
    # When there's no face, like in case any pet, tries to reach to it: 
    speak("Face is not found please try again ")
if m == 1:
    # After 5 seconds, the lock announces:
    speak("Door is closing")

