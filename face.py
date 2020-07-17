import cv2 as cv
print(cv.__version__)

cap = cv.VideoCapture(3)

classifier = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:

    ret, frame = cap.read()

    if ret:
        faces = classifier.detectMultiScale(frame)

        for face in faces:
            x, y, w, h = face
            frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)

        cv.imshow("My window", frame)

    key = cv.waitKey(30)

    if key == ord("q"):
        break

cap.release()
cv.destroyAllWindows()