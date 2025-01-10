# FaceFinder-
This repository contains a Python-based face detection project leveraging a pre-trained model from Google Teachable Machine. It offers an easy-to-implement solution for detecting faces in images and video streams using machine learning techniques.

## Features
- Detects faces in images and in real-time too.
- Utilizes a pre-trained model from Google Teachable Machine.
- Beginner-friendly and easy to customize.


## Prerequisites
- Python 3.7 or higher
- OpenCV library
- NumPy library
- A pre-trained face model from Google Teachable Machine

## Step by step process for the make the Project Run

Step 1: Install Python and PyCharm
Install Python:

Download Python from python.org/downloads and install it on your system.
During installation, ensure you check the box to add Python to your PATH.
Install PyCharm:

Download PyCharm from jetbrains.com/pycharm/download and install it. PyCharm will serve as your development environment to write and execute Python programs.
Step 2: Create a New Python Project in PyCharm
Launch PyCharm and select New Project.
Choose a project directory and ensure Python is selected as the interpreter.
Step 3: Install OpenCV Library
Open the terminal within PyCharm (usually located at the bottom of the screen).

Run the following command to install the OpenCV library:



pip install opencv-python
Step 4: Create a Python File
In the project folder, right-click and select New > Python File.
Name the file face_detection.py.
Step 5: Import Required Libraries
Open the face_detection.py file.

Import the OpenCV library:


import cv2
Step 6: Load a Pre-Trained Face Detection Model
OpenCV provides a pre-trained Haar Cascade classifier for face detection.

Load it using the following code:


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
Step 7: Access Webcam Feed
Initialize the webcam to capture video:

python
Copy code
cap = cv2.VideoCapture(0)
Step 8: Detect Faces in Real-Time
Use a loop to process each video frame:

Convert the frame to grayscale for better face detection.
Apply the Haar Cascade model to detect faces.
python
Copy code
while True:
    # Capture video frames
    ret, frame = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Face Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
Step 9: Release Resources and Close Windows
After exiting the loop, release the webcam and close all OpenCV windows:

python
Copy code
cap.release()
cv2.destroyAllWindows()
Full Code:
python
Copy code
import cv2

# Load the pre-trained Haar Cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture video frames
    ret, frame = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Face Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
Step 10: Execute the Program
Run the program by clicking the Run button in PyCharm.
Your webcam will open, and the program will start detecting faces in real-time.
Press the 'q' key to close the video feed and stop the program.


### 1. Install OpenCV
First, you need to install the OpenCV library. You can install it using pip:

```bash
pip install opencv-python
```

### 2. Import Libraries
We need to import the required libraries. In this case, it's OpenCV for computer vision tasks and numpy for handling arrays.

```python
import cv2
import numpy as np
```

### 3. Load the Pre-trained Haar Cascade Classifier
OpenCV provides pre-trained classifiers for detecting faces, eyes, etc. These classifiers are based on Haar features and can be loaded from XML files.

```python
# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
```

This line loads a specific classifier, `haarcascade_frontalface_default.xml`, which is trained to detect faces in images. OpenCV has several other pre-trained classifiers for detecting eyes, smile, etc.

### 4. Load the Image or Video
Next, we need to load the image or video on which we want to perform face detection.

```python
# Load the image from file
image = cv2.imread('image.jpg')

# Convert the image to grayscale (required for Haar Cascade detection)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```

- `cv2.imread()` loads the image from the specified file.
- `cv2.cvtColor()` converts the image from color to grayscale. Grayscale images are easier to process and work better for face detection.

### 5. Detect Faces
Now, we use the `detectMultiScale()` method to detect faces in the grayscale image.

```python
# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
```

Here's what the parameters mean:
- `gray`: The input image in grayscale.
- `scaleFactor`: This compensates for any face scaling in the image. It typically works by resizing the image. A value of 1.1 means the image is resized by 10% at each step.
- `minNeighbors`: This parameter specifies how many neighbors each candidate rectangle should have to retain it. A higher value means fewer detections, but with higher quality.
- `minSize`: The minimum size of the detected face. If faces are smaller than this size, they will be ignored.

`detectMultiScale()` returns a list of rectangles where it believes faces are located. Each rectangle is represented by four values: `(x, y, w, h)` where:
- `x, y` is the top-left corner of the rectangle.
- `w, h` are the width and height of the rectangle.


This loop iterates over each detected face and uses `cv2.rectangle()` to draw a blue rectangle (`(255, 0, 0)` represents blue in BGR format) around the detected face. The `2` represents the thickness of the rectangle.

### 7. Display the Image
After processing the image, we can display it with the detected faces.

```python
# Display the output image with faces highlighted
cv2.imshow('Face Detection', image)

# Wait until a key is pressed to close the image window
cv2.waitKey(0)

# Close all OpenCV windows
cv2.destroyAllWindows()
```

- `cv2.imshow()` displays the image in a window named 'Face Detection'.
- `cv2.waitKey(0)` pauses the program until a key is pressed.
- `cv2.destroyAllWindows()` closes all OpenCV windows after the image is closed.

### Complete Code Example:

```python
from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()



-Installlation of facefinder+ 
-Step1: Open chrome and search for Google Teachable Machines website(https://teachablemachine.withgoogle.com/) and click on Get started. 
![image](https://github.com/user-attachments/assets/2f789832-b844-4643-a1b8-e20919208c40)

-Step2: And under the New Project section, select the Image Model folder.
![image](https://github.com/user-attachments/assets/a935443e-70b4-4426-b2b7-5372348bc88d)

-Step3: Select the Standard Image Model option.
![image](https://github.com/user-attachments/assets/cf425c7c-d88b-4e0b-8825-cd45890af7e3)

-Step4: Determine the Required classes and upload the photos using webcam and google drive, And Train the Model. 
-You could see the comparision percentage for the given classes.
![image](https://github.com/user-attachments/assets/109f6d41-d457-45f4-bc56-a5a22fb85fb2)

-Step5: Click on Export the Model.
-Step6: Under the tensorflow tab, select Download the model option.
-And a .zip file would be downloaded. 
![image](https://github.com/user-attachments/assets/96d79673-80dc-4494-9fb9-1164397d1675)

-Step8: To run it in local system, paste the Given code in pycharm or any other code Interpreter.
-Step9: Open the .zip file and copy the .h5 file and .txt file.
-And the save the files to the project path to Run the code.
![image](https://github.com/user-attachments/assets/c5a57d45-ebb8-4136-8082-e0e275d8e441)

-Step10: Open the terminal and install the required libraries for the project.
-First, open your terminal. 
-Then, run this command: pip install -r requirements.txt 
=This will automatically install all the libraries needed for the project."

Tip-- chech the right python versions and tensorflow versions to avoid the syntax errors.
-Step11: Run the code and, Hence the facefinder+ works.



## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests for enhancements or bug fixes.


## License
This project is licensed under the MIT License.


## Acknowledgments
- [Google Teachable Machine](https://teachablemachine.withgoogle.com/) for providing the pre-trained model.(RECOMMENDED)
- [OpenCV](https://opencv.org/) for real-time image processing.
