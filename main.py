import cv2 as cv

video = cv.VideoCapture("love_story.mp4")
# video = cv.VideoCapture(0)

ret, first_frame = video.read()
ret, second_frame = video.read()
while True:
    diff = cv.absdiff(first_frame, second_frame)

    gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (21,21),0)

    thresh = cv.threshold(gray, 70, 255, cv.THRESH_BINARY)[1]

    thresh = cv.dilate(thresh, None, iterations=2)
    
    contours,_ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for c in contours:
        cv.putText(first_frame, "Status Movement", (10,20), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 3)

    cv.imshow("my video", first_frame)
    # cv.imshow("my video", diff)

    first_frame = second_frame
    ret, second_frame = video.read()

    key = cv.waitKey(1)
    if key == ord('q'):
        break

video.release()
cv.destroyAllWindows()
