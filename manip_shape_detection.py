import cv2 as cv
import numpy as np
cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    blurred = cv.GaussianBlur(gray, (9,9), 0)
    edges = cv.Canny(blurred, 50, 100)
    _, contour = cv.threshold(edges, 140, 255, cv.THRESH_BINARY)

    contours, _ = cv.findContours(contour, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    cv.drawContours(frame, contours, -1, (0, 255, 0), 2)

    cv.imshow('Contour Detection', frame)


    if cv.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv.destroyAllWindows()