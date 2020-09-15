# Face Detection Model using a webcam

import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # Load the cascade for the face.

def facedetect(gray, frame): # Create a function that takes the image as input in black and white (gray) and the original image (frame), then return the same image with the detector rectangles.
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) # Apply the detectMultiScale method from the face cascade to locate one or several faces in the image.
    for (x, y, w, h) in faces: # For each detected face:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # Color a bounding box around the face.
        roi_gray = gray[y:y+h, x:x+w] # Get the region of interest in the black and white image.
        roi_color = frame[y:y+h, x:x+w] # Get the region of interest in the colored image.
    return frame # Return the image with the detector rectangles.
    
video_capture = cv2.VideoCapture(0) # Turn the webcam on.

while True: # Repeat inifinitely until process is stopped with stop condition
    
    src, img = video_capture.read() # Capture frame-by-frame
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert colored image to black and white
   
    facedetect(gray, img) # Method call that returns image with the detector rectangles
    cv2.imshow('webcam', img) # Show webcam video
    if cv2.waitKey(1) == ord('q'): # Condition to quit webcam video with the letter q, the stop condition
        break
    
video_capture.release(); # Turn the webcam off
cv2.destroyAllWindows() # Closes all windows that were created
