import os
import cv2

# Get the path of the cascade used for detection
cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
print(cv2_base_dir)
cascadePath = os.path.join(
    cv2_base_dir, 'data/haarcascade_frontalface_default.xml')

# Create cascade object
haar_model = cv2.CascadeClassifier(cascadePath)

# Select the webcam to be used for capture
vidCapture = cv2.VideoCapture(0)

while True:
    # Read the video frame by frame everyloop and also check for the return value if the
    # video runs out of frames(not neccessary for lve detection since the frames never end)
    ret, frame = vidCapture.read()

    # Convert each frame to grayscale
    grayVid = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the images
    faces = haar_model.detectMultiScale(
        grayVid,
        scaleFactor=1.1,
        # Increase this number for a better quality of detection
        minNeighbors=5,
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around any face found using the returned coordinates
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0))

    # Show every frame
    cv2.imshow("CompSight", frame)

    # Stop process if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vidCapture.release()
cv2.destroyAllWindows()
