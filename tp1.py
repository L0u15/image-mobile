import cv2
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 500)
fontScale = 1
fontColor = (255, 255, 255)
lineType = 2


def save_webcam(outPath, fps, mirror=False):
    # Capturing video from file:
    cap = cv2.VideoCapture('venv/data/bike.avi')

    ret, first_frame = cap.read()

    while (cap.isOpened()):

        ret, current_frame = cap.read()
        sub = cv2.subtract(first_frame, current_frame)

        imgray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(imgray, 100, 255, 0)
        img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            cv2.drawContours(current_frame, c, -1, (0, 255, 0), 3)

        if ret == True:

            cv2.imshow('frame', current_frame)
            cv2.imshow('sub', sub)
            cv2.imshow('img', img)

        else:
            break

        if cv2.waitKey(100) & 0xFF == ord('q'):
            # if 'q' is pressed then quit
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def main():
    save_webcam('output.avi', 30.0, mirror=True)


if __name__ == '__main__':
    main()
