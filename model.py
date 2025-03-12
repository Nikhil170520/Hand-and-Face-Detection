import cv2
import mediapipe as mp

fac=cv2.CascadeClassifier('D:\haarcascade_frontalface_default.xml')

cap=cv2.VideoCapture(0)
mp_hands=mp.solutions.hands
hands=mp_hands.Hands()
mp_draw=mp.solutions.drawing_utils
while True:
    _,img=cap.read()
    results=hands.process(img)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                height,width,channel=img.shape
                cx,cy=int(lm.x*width),int(lm.y*height)
                cv2.circle(img,(cx,cy),5,(0,0,0))
                mp_draw.draw_landmarks(img,hand_landmarks,mp_hands.HAND_CONNECTIONS)
    face=fac.detectMultiScale(img,1.1,5)
    for x,y,w,h in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,250,0),3)
    cv2.imshow("L  I  V  E",img)
    if cv2.waitKey(1)&0xff==ord('a'):
            break
cap.release()
cv2.destroyAllWindows()