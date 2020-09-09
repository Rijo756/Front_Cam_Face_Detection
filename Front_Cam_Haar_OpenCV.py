import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 200)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    success, img = cap.read()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gauss =  cv2.GaussianBlur(img,(7,7),cv2.BORDER_DEFAULT)
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        img_crop = img[y:y + h, x:x + w]
        gauss[y:y + h, x:x + w] = img_crop
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        roi_gray = gray[y:y + h, x:x + w]

    gauss = cv2.flip(img, 1)
    cv2.imshow("Video", gauss)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
