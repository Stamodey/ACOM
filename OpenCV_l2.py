import cv2
import numpy as np

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Ошибка: Не удалось открыть камеру.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Не удалось получить кадр. Выход...")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    treshold = mask1 + mask2
    
    kernel = np.ones((5, 5), np.uint8)

    opening = cv2.morphologyEx(treshold, cv2.MORPH_OPEN, kernel, iterations=2)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        
        moments = cv2.moments(largest_contour)
        
        area = moments['m00']
        
        if area > 500:
            cx = int(moments['m10'] / area)
            cy = int(moments['m01'] / area)
            
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            
            cv2.putText(frame, f"Area: {int(area)}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Original Frame', frame)
    cv2.imshow('Treshold (Red Filter)', treshold)
    cv2.imshow('Morphological Transformations', closing)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
