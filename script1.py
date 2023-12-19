import cv2, time

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

vdo=cv2.VideoCapture(0)

a=1 #Frame Counter
while True:
    a=a+1
    check, frame= vdo.read()
    print(check)
    print(frame)
    faces=face_cascade.detectMultiScale(frame, scaleFactor=1.15, minNeighbors=5)
    for x,y,w,h in faces:
        frame=cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255),2)
    #gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow("Video",frame)
    key=cv2.waitKey(1)

    if key==ord('q'):
        break

vdo.release()
cv2.destroyAllWindows()
print(a)
