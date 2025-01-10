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

--pip install opencv-python


Step 4: Create a Python File
In the project folder, right-click and select New > Python File.
Name the file main_py.
Step 5: Import Required Libraries
Open the main_.py file.


Step 6: Load a Pre-Trained Face Detection Model
use google teacheable Machines to create pre trained  model using the steps
-Step 6.1: Open chrome and search for Google Teachable Machines website(https://teachablemachine.withgoogle.com/) and click on Get started. 
![image](https://github.com/user-attachments/assets/2f789832-b844-4643-a1b8-e20919208c40)

-Step 6.2: And under the New Project section, select the Image Model folder.
![image](https://github.com/user-attachments/assets/a935443e-70b4-4426-b2b7-5372348bc88d)

-Step6.3: Select the Standard Image Model option.
![image](https://github.com/user-attachments/assets/cf425c7c-d88b-4e0b-8825-cd45890af7e3)

-Step6.4: Determine the Required classes and upload the photos using webcam and google drive, And Train the Model. 
-You could see the comparision percentage for the given classes.
![image](https://github.com/user-attachments/assets/109f6d41-d457-45f4-bc56-a5a22fb85fb2)

-Step6.5: Click on Export the Model.
-Step6.6: Under the tensorflow tab, select Download the model option.
-And a .zip file would be downloaded. 
![image](https://github.com/user-attachments/assets/96d79673-80dc-4494-9fb9-1164397d1675)
 Step 7: open the .zip file which is Downloaded from the Google teachable machines.
 --Open the .zip file and extract the "keras_model.h5 and labels.txt file. Add it to pychram project folder where the the project the running.

![image](https://github.com/user-attachments/assets/c5a57d45-ebb8-4136-8082-e0e275d8e441)

Step 8. Install the required libraries using the requirements.txt file()
First, you need to install the OpenCV library. You can install it using pip:

```bash
pip install -r requirements.txt 
```
-Step8: To run it in local system, paste the Given code in pycharm or any other code Interpreter.


### Complete Code Example:

```python
from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
```
```
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)
```

```
# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)
```

```
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

```
Tip-- chech the right python versions and tensorflow versions to avoid the syntax errors.
--download the the complete code by Downloading the main_.py file and paste it in pycharm projects.

Step 9: Execute the Program


Tip-- chech the right python versions and tensorflow versions to avoid the syntax errors.



-Step11: Run the code and, Hence the facefinder+ works.

Run the program by clicking the Run button in PyCharm.
Your webcam will open, and the program will start detecting faces in real-time.
Press the 'q' key to close the video feed and stop the program.


## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests for enhancements or bug fixes.


## License
This project is licensed under the MIT License.


## Acknowledgments
- [Google Teachable Machine](https://teachablemachine.withgoogle.com/) for providing the pre-trained model.(RECOMMENDED)
- [OpenCV](https://opencv.org/) for real-time image processing.
