import cv2, time
import pandas
from datetime import datetime

first_frame=None
#face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
df=pandas.DataFrame(columns=["Start", "End"])

vdo=cv2.VideoCapture(0)
a=0
status_list=[None,None]
change_time_list=[]
while True:
    check, frame= vdo.read()
    status=0
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray, (21,21), 0)

    #if first_frame is None:
    if a<40:        #here the webcam is slow so we need to leave first few frames for the first frame
        first_frame= gray
        a=a+1
        continue

    delta_frame= cv2.absdiff(first_frame, gray)
    thresh_delta_frame=cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1] #for aall the intensity more than 30 make it white
    thresh_delta_frame=cv2.dilate(thresh_delta_frame, None, iterations=5) #dilate the black holes in the white part due to shadows

    (_,cnts,_)= cv2.findContours(thresh_delta_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #if 2 continuoes whites but distinct
     #image then show it as diffrent contoours
    for con in cnts:
        if cv2.contourArea(con)<2000:
            continue
        status=1 #motion detected
        (x,y,w,h)= cv2.boundingRect(con) #if contoours>1000 pux then display ang draw rectanglearound it
        cv2.rectangle(frame, (x,y),(x+w, y+h),(0,0,255),2)

    status_list.append(status)

    status_list=status_list[-2:]

    if (status_list[-1]==0 and status_list[-2]==1) or  (status_list[-1]==1 and status_list[-2]==0):
        change_time_list.append(datetime.now())#neat look
    cv2.imshow("Current Frame",gray)
    cv2.imshow("Delta Frame",delta_frame)
    cv2.imshow("Thresh Delta Frame",thresh_delta_frame)
    cv2.imshow("Frame",frame)

    key=cv2.waitKey(1)

    if key==ord('q'):
        if status_list[-1]==1:
            change_time_list.append(datetime.now()) #time.localtime(time.time())
        break
print(change_time_list)

for i in range(0, len(change_time_list), 2): # i=i+2
    df=df.append({"Start":change_time_list[i] , "End": change_time_list[i+1]}, ignore_index=True)

df.to_csv("Times.csv")
vdo.release()
cv2.destroyAllWindows()
