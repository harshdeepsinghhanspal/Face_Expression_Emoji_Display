import cv2
from deepface import DeepFace
import numpy as np

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video
cap = cv2.VideoCapture(0)

def draw_emoji(emotion):
    # Create a new black background with a larger size
    emoji_frame = np.zeros((400, 400, 3), dtype="uint8")

    # Common features (eyes)
    # Draw eyes (two larger circles)
    cv2.circle(emoji_frame, (120, 160), 40, (255, 255, 255), -1)  # Left eye white
    cv2.circle(emoji_frame, (280, 160), 40, (255, 255, 255), -1)  # Right eye white
    cv2.circle(emoji_frame, (120, 160), 20, (0, 0, 0), -1)        # Left eye black
    cv2.circle(emoji_frame, (280, 160), 20, (0, 0, 0), -1)        # Right eye black

    # Draw emoji based on emotion
    if emotion == 'happy':
        # Happy (larger curved smile)
        cv2.ellipse(emoji_frame, (200, 280), (100, 50), 0, 0, 180, (0, 255, 0), 10)
    elif emotion == 'sad':
        # Sad (larger curved frown)
        cv2.ellipse(emoji_frame, (200, 320), (100, 50), 0, 180, 360, (255, 0, 0), 10)
    elif emotion == 'angry':
        # Angry (larger sharp eyebrows and straight mouth)
        cv2.line(emoji_frame, (80, 80), (160, 120), (0, 0, 255), 10)  # Left eyebrow
        cv2.line(emoji_frame, (240, 120), (320, 80), (0, 0, 255), 10) # Right eyebrow
        cv2.line(emoji_frame, (140, 280), (260, 280), (0, 0, 255), 10)  # Straight mouth
    elif emotion == 'neutral':
        # Neutral (straight mouth)
        cv2.line(emoji_frame, (140, 280), (260, 280), (255, 255, 255), 10)
    elif emotion == 'fear':
        # Fear (larger open mouth)
        cv2.circle(emoji_frame, (200, 300), 40, (0, 255, 255), 10)

    return emoji_frame

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert grayscale frame to RGB format
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face ROI (Region of Interest)
        face_roi = rgb_frame[y:y + h, x:x + w]

        # Perform emotion analysis on the face ROI
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

        # Determine the dominant emotion
        emotion = result[0]['dominant_emotion']

        # Draw rectangle around face and label with predicted emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Draw the emoji on a black background frame
        emoji_frame = draw_emoji(emotion)

        # Display the larger emoji frame in a separate window
        cv2.imshow('Emoji Representation', emoji_frame)

    # Display the resulting frame
    cv2.imshow('Real-time Emotion Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
