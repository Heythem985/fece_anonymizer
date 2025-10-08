import cv2

# Load a pre-trained face detection model (Haar Cascade classifier)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
cap = cv2.VideoCapture(0)
# Start an infinite loop to process each video frame in real time
while True:
    # Capture one frame from the webcame
    ret, frame = cap.read()
    # Haar cascades work better on gray images
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.05,
        minNeighbors=2,
        minSize=(60, 60)
        )
    #Loop through all detected faces
    for (x, y, w, h) in faces:
        #Extract the face region from the frame
        face = frame[y:y+h, x:x+w]
        #Apply a strong Gaussian blur to anonymize the face
        blurred_face = cv2.GaussianBlur(face, (99, 99), 30)
        # Replace the original face area in the frame with the blurred version
        frame[y:y+h, x:x+w] = blurred_face

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
