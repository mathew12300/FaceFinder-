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

1. Import Required Libraries

from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # OpenCV for webcam access and image processing
import numpy as np  # NumPy for numerical operations

Explanation:

load_model: Used to load the pre-trained face recognition model.
cv2: OpenCV allows accessing the webcam and processing image frames.
numpy: Handles numerical operations like reshaping and normalizing image arrays.
2. Load Model and Labels

# Load the pre-trained model
model = load_model("keras_Model.h5", compile=False)

# Load the class labels
class_names = open("labels.txt", "r").readlines()
Explanation:

Model Loading: The model file (keras_Model.h5) contains the trained weights and architecture.
Labels File: labels.txt maps the class indices to their respective names.
3. Initialize Webcam

# Set up the webcam (use 0 for default camera)
camera = cv2.VideoCapture(0)
Explanation:

cv2.VideoCapture(0): Initializes the webcam. Replace 0 with 1 or another index if you have multiple cameras.
4. Process Webcam Input

while True:
    # Capture frame from the webcam
    ret, image = camera.read()

    # Resize the frame to 224x224 pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the frame in a window
    cv2.imshow("Webcam Image", image)
Explanation:

Frame Capture: Captures a single frame from the webcam.
Resize: Resizes the image to the required input dimensions (224x224).
Display: Shows the captured image in a window for real-time feedback.
5. Preprocess the Frame

    # Convert the frame to a NumPy array and reshape
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the pixel values to [-1, 1]
    image = (image / 127.5) - 1
Explanation:

NumPy Conversion: Converts the image to a NumPy array for compatibility with the model.
Reshape: Matches the model’s expected input dimensions: (1, 224, 224, 3).
Normalization: Scales pixel values from [0, 255] to [-1, 1] for improved model performance.
6. Make Predictions

    # Predict the class of the frame
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print the prediction results
    print(f"Class: {class_name[2:]}, Confidence: {confidence_score * 100:.2f}%")
Explanation:

model.predict: Runs the preprocessed image through the model to get predictions.
np.argmax: Identifies the class with the highest confidence score.
Print Results: Outputs the predicted class and its confidence score.
7. Exit on ESC Key
python
Copy code
    # Exit the loop when the ESC key is pressed
    if cv2.waitKey(1) == 27:
        break
Explanation:

Listens for the ESC key (ASCII 27). If pressed, the program exits the loop.
8. Cleanup

# Release the webcam and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()
Explanation:

Release Webcam: Frees up the webcam for other applications.
Close Windows: Closes any OpenCV-created GUI windows.
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
