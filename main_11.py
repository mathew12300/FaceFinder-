from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
import time

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
try:
    with open("labels.txt", "r") as file:
        class_names = file.readlines()
except FileNotFoundError:
    print("Error: labels.txt not found.")
    exit()

# Initialize camera
camera = cv2.VideoCapture(0)

# Check if the camera is available
if not camera.isOpened():
    print("Error: Camera not found.")
    time.sleep(2)
    camera.open(0)  # Retry opening the camera
    if not camera.isOpened():
        print("Error: Could not access camera.")
        exit()

# Initialize face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Main loop
while True:
    ret, image = camera.read()

    # Check if the image was captured successfully
    if not ret:
        print("Failed to grab image, retrying...")
        continue  # Retry reading the image

    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # If faces are detected, process each face
    for (x, y, w, h) in faces:
        face = image[y:y + h, x:x + w]
        face_resized = cv2.resize(face, (224, 224))

        # Convert the image to a numpy array and normalize it
        face_array = np.asarray(face_resized, dtype=np.float32).reshape(1, 224, 224, 3)
        face_normalized = (face_array / 127.5) - 1

        # Predict using the model
        try:
            prediction = model.predict(face_normalized)
        except Exception as e:
            print(f"Error during prediction: {e}")
            break  # Exit if an error occurs during prediction

        # Get the predicted class and confidence score
        index = np.argmax(prediction)
        class_name = class_names[index].strip()  # Strip any extra newline characters
        confidence_score = prediction[0][index]

        # Print the prediction to the terminal
        print(f"Class: {class_name}, Confidence Score: {np.round(confidence_score * 100)}%")

        # Display prediction on the video feed
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, f'{class_name}: {np.round(confidence_score * 100)}%', (x, y - 10), font, 0.9, (0, 255, 0), 2)

        # Draw a rectangle around the face
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw the confidence scale
        scale_width = int(w * confidence_score)  # Scale the width based on confidence
        cv2.rectangle(image, (x, y + h + 10), (x + scale_width, y + h + 30), (0, 255, 0), -1)  # Filled rectangle
        cv2.rectangle(image, (x, y + h + 10), (x + w, y + h + 30), (255, 255, 255), 2)  # Outline of the scale

    # Show the video feed
    cv2.imshow("Webcam Image", image)

    # Listen to the keyboard for key presses
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII code for the ESC key
    if keyboard_input == 27:
        break

# Release the camera and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()
